"""Train Triple-stream CFM segmentation with baseline-compatible metrics and checkpoints."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

_SAM = Path(__file__).resolve().parents[2]
if str(_SAM) not in sys.path:
    sys.path.insert(0, str(_SAM))
if str(_SAM / "ablation_study" / "baseline_models") not in sys.path:
    sys.path.insert(0, str(_SAM / "ablation_study" / "baseline_models"))

from common.datasets import build_bijie_split, build_l4s_split
from common.metrics import image_level_metrics_from_logits, pixel_metrics_from_logits

from model_architecture_cfm_landseg.data.datasets import AugmentTripleStream, BijieTripleStreamDataset, L4STripleStreamDataset
from model_architecture_cfm_landseg.losses.composite import TriCFMCompositeLoss
from model_architecture_cfm_landseg.tri_cfm_net import TriEncoderCFMNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def latest_checkpoint(ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def save_checkpoint(path: Path, state: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def prep_batch_triplet(batch: dict, device):
    rg = batch["stream_rgb"].float().to(device, non_blocking=True)
    dem = batch["stream_dem"].float().to(device, non_blocking=True)
    ctx = batch["stream_ctx"].float().to(device, non_blocking=True)
    y = batch["mask"].float().to(device, non_blocking=True)
    if y.dim() == 3:
        y = y.unsqueeze(1)
    return rg, dem, ctx, y


def _accumulate_thresh_sweep_probs(
    prob: torch.Tensor,
    tgt: torch.Tensor,
    thresholds: torch.Tensor,
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
):
    """Vectorized accumulate micro TP/FP/FN across batch for fixed threshold grid (prob: same device dtype)."""
    # prob [B,H,W], tgt bool [B,H,W], thresholds [K]
    tgt = (tgt > 0).to(torch.bool)
    for idx, tau in enumerate(thresholds):
        pred = (prob >= tau).to(torch.bool)
        tp[idx] += (pred & tgt).sum()
        fp[idx] += (pred & ~tgt).sum()
        fn[idx] += (~pred & tgt).sum()


def run_epoch(
    model,
    loader,
    criterion,
    device: torch.device,
    threshold: float,
    training: bool,
    optimizer=None,
    infer_flow_steps: int = 0,
    grad_clip_norm: float = 0.0,
    thresh_sweep: tuple[float, ...] | None = None,
):
    """
    thresh_sweep: if set on validation-only, accumulates pooled micro pixel F1/IoU per threshold across the split
                  (helps match tuned-threshold metrics used in papers; single forward per batch).
    """
    model.train() if training else model.eval()
    losses = []
    pix_hist = {"acc": [], "precision": [], "recall": [], "f1": [], "iou": []}
    img_hist = {"auroc": [], "auprc": [], "best_f1": [], "best_threshold": []}

    sweep_tp = sweep_fp = sweep_fn = sweep_thr_used = None
    if (not training) and thresh_sweep is not None and len(thresh_sweep) > 0:
        k = len(thresh_sweep)
        sweep_tp = torch.zeros(k, device=device, dtype=torch.float64)
        sweep_fp = torch.zeros(k, device=device, dtype=torch.float64)
        sweep_fn = torch.zeros(k, device=device, dtype=torch.float64)
        sweep_thr_used = tuple(float(t) for t in thresh_sweep)
        thresh_tensor = torch.tensor(thresh_sweep, device=device, dtype=torch.float32)

    pbar = tqdm(loader, desc="Train" if training else "Val", leave=False)
    for batch in pbar:
        rg, dem, ctx, y = prep_batch_triplet(batch, device)
        with torch.set_grad_enabled(training):
            if training:
                out = model(rg, dem, ctx, gt_mask=y, inference_flow_steps=None)
                loss = criterion(out, y, dem_chw=dem)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            else:
                out = model(
                    rg,
                    dem,
                    ctx,
                    gt_mask=None,
                    inference_flow_steps=int(infer_flow_steps) if infer_flow_steps > 0 else None,
                )
                loss = criterion(out, y, dem_chw=dem)

        losses.append(float(loss.item()))
        logits = out["logits_aux"]
        pix = pixel_metrics_from_logits(logits, y, threshold=threshold)
        for k in pix_hist:
            pix_hist[k].append(float(pix[k]))
        img = image_level_metrics_from_logits(logits, y, prob_thr_for_instances=threshold, min_area=20)
        for k in img_hist:
            img_hist[k].append(float(img[k]))

        if sweep_tp is not None:
            with torch.no_grad():
                probs = torch.sigmoid(logits)[:, 0]
                yt = y[:, 0]
                _accumulate_thresh_sweep_probs(probs, yt.to(torch.bool), thresh_tensor, sweep_tp, sweep_fp, sweep_fn)

        pbar.set_postfix(loss=f"{losses[-1]:.4f}", f1=f"{pix_hist['f1'][-1]:.4f}", iou=f"{pix_hist['iou'][-1]:.4f}")

    result = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        **{k: float(np.mean(v)) if v else 0.0 for k, v in pix_hist.items()},
        "auroc": float(np.mean(img_hist["auroc"])) if img_hist["auroc"] else 0.0,
        "auprc": float(np.mean(img_hist["auprc"])) if img_hist["auprc"] else 0.0,
        "image_best_f1": float(np.mean(img_hist["best_f1"])) if img_hist["best_f1"] else 0.0,
        "image_best_threshold": float(np.mean(img_hist["best_threshold"])) if img_hist["best_threshold"] else threshold,
    }

    if sweep_tp is not None:
        eps = 1e-6
        f1_micro = (2 * sweep_tp + eps) / (2 * sweep_tp + sweep_fp + sweep_fn + eps)
        iou_micro = (sweep_tp + eps) / (sweep_tp + sweep_fp + sweep_fn + eps)
        k_best_i = int(torch.argmax(f1_micro).item())
        result["pixel_f1_val_micro_thresh_sweep_best"] = float(f1_micro[k_best_i].item())
        result["pixel_iou_val_micro_thresh_sweep_best"] = float(iou_micro[k_best_i].item())
        result["pixel_val_best_threshold_sweep"] = float(sweep_thr_used[k_best_i])
    return result


def score_for_best_checkpoint(val_m: dict, best_metric: str) -> float:
    if best_metric in ("val_global_f1_sweep", "val_global_f1"):  # alias
        return float(val_m.get("pixel_f1_val_micro_thresh_sweep_best", val_m["f1"]))
    if best_metric == "val_iou_batch_mean":
        return float(val_m["iou"])
    if best_metric == "harmonic_mean":
        pf = float(val_m.get("pixel_f1_val_micro_thresh_sweep_best", val_m["f1"]))
        vi = float(val_m.get("pixel_iou_val_micro_thresh_sweep_best", val_m["iou"]))
        return 2.0 * pf * vi / (pf + vi + 1e-12)
    return float(val_m["f1"])  # val_f1_batch_mean at metric_threshold


def package_checkpoint_state(
    epoch_num: int,
    model,
    optimizer,
    scheduler,
    best_score: float,
    best_checkpoint_metric: str,
    metric_threshold: float,
) -> dict:
    return {
        "epoch": epoch_num,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_score": best_score,
        "best_metric": best_checkpoint_metric,
        "metric_threshold": metric_threshold,
    }


def build_default_thresh_sweep(start: float, end: float, step: float) -> tuple[float, ...]:
    """Inclusive-ish grid [start, end] with floating-point safe bound."""
    t = []
    x = float(start)
    guard = int(round((end - start) / step)) + 10
    for _ in range(max(guard, 40)):
        if x > end + step * 0.25:
            break
        t.append(round(float(x), 5))
        x += step
    return tuple(sorted(set(t)))


def train_loop(
    model: TriEncoderCFMNet,
    train_ds,
    val_ds,
    output_dir: Path,
    epochs: int = 200,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    lr_scheduler_patience: int = 12,
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_min_lr: float = 1e-7,
    num_workers: int = 8,
    device_str: str = "cuda",
    metric_threshold: float = 0.6,
    grad_clip_norm: float = 5.0,
    thresh_sweep: tuple[float, ...] | None = None,
    best_checkpoint_metric: str = "val_global_f1_sweep",
    save_every: int = 5,
    resume: bool = False,
    fm_weight: float = 1.0,
    seg_weight: float = 2.0,
    geo_weight: float = 0.15,
    vsmooth_weight: float = 0.05,
    tversky_alpha: float = 0.6,
    tversky_beta: float = 0.4,
    latent_sigma: float = 4.0,
    fm_residual_scale_sq: float | None = None,
    val_infer_flow_steps: int = 0,
    debug_one_batch: bool = False,
    extra_final: dict | None = None,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    criterion = TriCFMCompositeLoss(
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        fm_weight=fm_weight,
        seg_weight=seg_weight,
        geo_weight=geo_weight,
        vsmooth_weight=vsmooth_weight,
        latent_sigma=latent_sigma,
        fm_residual_scale_sq=fm_residual_scale_sq,
    )
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        cooldown=3,
        min_lr=lr_scheduler_min_lr,
    )

    ckpt_dir = output_dir / "checkpoint"
    results_dir = output_dir / "results"
    epoch_csv = results_dir / "epoch_metrics.csv"
    final_csv = results_dir / "final_metrics.csv"

    start_epoch = 1
    best_score = 0.0
    if resume:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if "scheduler" in state:
                scheduler.load_state_dict(state["scheduler"])
            start_epoch = int(state["epoch"]) + 1
            best_score = float(state.get("best_score", state.get("best_f1", 0.0)))

    if debug_one_batch:
        batch = next(iter(train_loader))
        rg, dem, ctx, y = prep_batch_triplet(batch, device)
        out = model(rg, dem, ctx, gt_mask=y)
        crit = criterion(out, y, dem_chw=dem)
        print("debug_one_batch loss", float(crit.item()))
        return

    for epoch in range(start_epoch, epochs + 1):
        train_m = run_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            threshold=metric_threshold,
            training=True,
            optimizer=optimizer,
            infer_flow_steps=0,
            grad_clip_norm=grad_clip_norm,
            thresh_sweep=None,
        )
        val_m = run_epoch(
            model,
            val_loader,
            criterion,
            device=device,
            threshold=metric_threshold,
            training=False,
            optimizer=None,
            infer_flow_steps=val_infer_flow_steps,
            grad_clip_norm=0.0,
            thresh_sweep=thresh_sweep,
        )

        sched_ref = score_for_best_checkpoint(val_m, best_checkpoint_metric)
        scheduler.step(sched_ref)

        row = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_acc": train_m["acc"],
            "train_precision": train_m["precision"],
            "train_recall": train_m["recall"],
            "train_f1": train_m["f1"],
            "train_iou": train_m["iou"],
            "train_auroc": train_m["auroc"],
            "train_auprc": train_m["auprc"],
            "train_image_best_f1": train_m["image_best_f1"],
            "train_image_best_threshold": train_m["image_best_threshold"],
            "val_loss": val_m["loss"],
            "val_acc": val_m["acc"],
            "val_precision": val_m["precision"],
            "val_recall": val_m["recall"],
            "val_f1": val_m["f1"],
            "val_iou": val_m["iou"],
            "val_auroc": val_m["auroc"],
            "val_auprc": val_m["auprc"],
            "val_image_best_f1": val_m["image_best_f1"],
            "val_image_best_threshold": val_m["image_best_threshold"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "val_pixel_f1_micro_thresh_sweep_best": val_m.get("pixel_f1_val_micro_thresh_sweep_best", float("nan")),
            "pixel_iou_val_micro_thresh_sweep_best": val_m.get("pixel_iou_val_micro_thresh_sweep_best", float("nan")),
            "pixel_val_best_threshold_sweep": val_m.get("pixel_val_best_threshold_sweep", float("nan")),
        }
        append_csv(epoch_csv, row)
        print(row)

        if epoch % save_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:04d}.pt",
                package_checkpoint_state(
                    epoch, model, optimizer, scheduler, best_score, best_checkpoint_metric, metric_threshold
                ),
            )
        cand = score_for_best_checkpoint(val_m, best_checkpoint_metric)
        if cand > best_score:
            best_score = cand
            save_checkpoint(
                ckpt_dir / "best.pt",
                package_checkpoint_state(
                    epoch, model, optimizer, scheduler, best_score, best_checkpoint_metric, metric_threshold
                ),
            )

    final = {
        "best_score": best_score,
        "best_checkpoint_metric": best_checkpoint_metric,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "metric_threshold": metric_threshold,
        "tversky_alpha": tversky_alpha,
        "tversky_beta": tversky_beta,
        "fm_weight": fm_weight,
        "seg_weight": seg_weight,
        "geo_weight": geo_weight,
        "vsmooth_weight": vsmooth_weight,
        "latent_sigma": latent_sigma,
        "fm_residual_scale_sq": criterion.fm_residual_scale_sq,
        "val_infer_flow_steps": val_infer_flow_steps,
    }
    if extra_final:
        final.update(extra_final)
    append_csv(final_csv, final)


def parse_args():
    p = argparse.ArgumentParser(description="Train TriEncoderCFMNet (triple-stream + light CFM)")
    p.add_argument("--dataset", type=str, choices=["landslide4sense", "bijie"], required=True)
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--experiment_name", type=str, default="tri_encoder_cfm")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--metric_threshold",
        type=float,
        default=0.6,
        help="Fixed prob threshold for batch-mean pixel metrics (dual_stream_gated default is 0.6).",
    )
    p.add_argument("--val_split_ratio_l4s", type=float, default=0.1)
    p.add_argument("--resize_to", type=int, default=256)
    p.add_argument(
        "--pyramid_width",
        type=int,
        default=64,
        help="Encoder base width; larger helps capacity vs DiGATe EfficientNet backbones.",
    )
    p.add_argument("--flow_combine_scale", type=float, default=0.5)
    p.add_argument("--model_flow_steps", type=int, default=0, help="Default FM integration steps at eval if --val_infer_flow_steps unset")
    p.add_argument("--val_infer_flow_steps", type=int, default=-1, help="If >=0, override eval integration steps; -1 uses model_flow_steps")
    p.add_argument("--fm_weight", type=float, default=1.0)
    p.add_argument("--seg_weight", type=float, default=2.0)
    p.add_argument("--geo_weight", type=float, default=0.15)
    p.add_argument("--vsmooth_weight", type=float, default=0.05)
    p.add_argument(
        "--tversky_alpha",
        type=float,
        default=0.6,
        help="Match dual_stream_gated / paper-style emphasis (higher alpha penalizes FP more).",
    )
    p.add_argument("--tversky_beta", type=float, default=0.4)
    p.add_argument("--grad_clip_norm", type=float, default=5.0, help="0 disables.")
    p.add_argument("--lr_scheduler_patience", type=int, default=12)
    p.add_argument("--lr_scheduler_factor", type=float, default=0.5)
    p.add_argument("--lr_scheduler_min_lr", type=float, default=1e-7)
    p.add_argument(
        "--best_checkpoint_metric",
        type=str,
        default="val_global_f1_sweep",
        choices=("val_global_f1_sweep", "val_f1_batch_mean", "val_iou_batch_mean", "harmonic_mean"),
        help="Score used for best.pt and ReduceLROnPlateau. val_global_f1_sweep = pooled micro-F1 at best prob threshold on val.",
    )
    p.add_argument(
        "--no_val_threshold_sweep",
        action="store_true",
        help="Disable val prob-threshold sweep (disables val_global_f1_sweep / harmonic_mean meaningfully).",
    )
    p.add_argument("--thresh_sweep_start", type=float, default=0.35)
    p.add_argument("--thresh_sweep_end", type=float, default=0.85)
    p.add_argument("--thresh_sweep_step", type=float, default=0.025)
    p.add_argument("--latent_sigma", type=float, default=4.0)
    p.add_argument(
        "--fm_residual_scale_sq",
        type=float,
        default=None,
        help="Divide FM MSE and time-smooth penalties by this (default: (latent_sigma*6)^2 so train_loss is comparable to val_loss scale).",
    )
    p.add_argument("--debug_one_batch", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    ctx_ch = 6 if args.dataset == "landslide4sense" else 4
    model = TriEncoderCFMNet(
        rgb_ch=3,
        dem_ch=1,
        ctx_ch=ctx_ch,
        pyramid_width=args.pyramid_width,
        flow_combine_scale=args.flow_combine_scale,
        inference_flow_steps=args.model_flow_steps,
        latent_sigma=args.latent_sigma,
    )

    output_dir = Path(args.output_dir).resolve()
    model_out = output_dir / args.dataset / args.experiment_name

    val_steps = args.val_infer_flow_steps if args.val_infer_flow_steps >= 0 else args.model_flow_steps

    if args.dataset == "landslide4sense":
        train_ids, val_ids = build_l4s_split(args.dataset_root, val_ratio=args.val_split_ratio_l4s, seed=args.seed)
        train_ds = L4STripleStreamDataset(
            args.dataset_root, ids=train_ids, resize_to=args.resize_to, transform=AugmentTripleStream(p=0.5)
        )
        val_ds = L4STripleStreamDataset(args.dataset_root, ids=val_ids, resize_to=args.resize_to, transform=None)
    else:
        train_raw, val_raw, _ = build_bijie_split(args.dataset_root, seed=args.seed)
        train_ds = BijieTripleStreamDataset(train_raw, resize_to=args.resize_to, transform=AugmentTripleStream(p=0.5))
        val_ds = BijieTripleStreamDataset(val_raw, resize_to=args.resize_to, transform=None)

    thresh_sweep = None
    if not args.no_val_threshold_sweep:
        thresh_sweep = build_default_thresh_sweep(
            args.thresh_sweep_start, args.thresh_sweep_end, args.thresh_sweep_step
        )
    best_metric = args.best_checkpoint_metric
    if args.no_val_threshold_sweep and best_metric in ("val_global_f1_sweep", "harmonic_mean"):
        best_metric = "val_f1_batch_mean"

    train_loop(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        output_dir=model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        lr_scheduler_min_lr=args.lr_scheduler_min_lr,
        num_workers=args.num_workers,
        device_str=args.device,
        metric_threshold=args.metric_threshold,
        grad_clip_norm=args.grad_clip_norm,
        thresh_sweep=thresh_sweep,
        best_checkpoint_metric=best_metric,
        save_every=args.save_every,
        resume=args.resume,
        fm_weight=args.fm_weight,
        seg_weight=args.seg_weight,
        geo_weight=args.geo_weight,
        vsmooth_weight=args.vsmooth_weight,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        latent_sigma=args.latent_sigma,
        fm_residual_scale_sq=args.fm_residual_scale_sq,
        val_infer_flow_steps=val_steps,
        debug_one_batch=args.debug_one_batch,
        extra_final={
            "dataset": args.dataset,
            "experiment_name": args.experiment_name,
            "ctx_ch": ctx_ch,
            "model_flow_steps": args.model_flow_steps,
            "val_infer_flow_steps": val_steps,
            "best_checkpoint_metric": best_metric,
            "val_threshold_sweep_enabled": not args.no_val_threshold_sweep,
        },
    )


if __name__ == "__main__":
    main()
