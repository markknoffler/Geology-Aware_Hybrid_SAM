"""Train Triple-stream CFM segmentation with baseline-compatible metrics and checkpoints."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
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


def run_epoch(
    model,
    loader,
    criterion,
    device: torch.device,
    threshold: float,
    training: bool,
    optimizer=None,
    infer_flow_steps: int = 0,
):
    model.train() if training else model.eval()
    losses = []
    pix_hist = {"acc": [], "precision": [], "recall": [], "f1": [], "iou": []}
    img_hist = {"auroc": [], "auprc": [], "best_f1": [], "best_threshold": []}

    pbar = tqdm(loader, desc="Train" if training else "Val", leave=False)
    for batch in pbar:
        rg, dem, ctx, y = prep_batch_triplet(batch, device)
        with torch.set_grad_enabled(training):
            if training:
                out = model(rg, dem, ctx, gt_mask=y, inference_flow_steps=None)
                loss = criterion(out, y, dem_chw=dem)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
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

        pbar.set_postfix(loss=f"{losses[-1]:.4f}", f1=f"{pix_hist['f1'][-1]:.4f}", iou=f"{pix_hist['iou'][-1]:.4f}")

    result = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        **{k: float(np.mean(v)) if v else 0.0 for k, v in pix_hist.items()},
        "auroc": float(np.mean(img_hist["auroc"])) if img_hist["auroc"] else 0.0,
        "auprc": float(np.mean(img_hist["auprc"])) if img_hist["auprc"] else 0.0,
        "image_best_f1": float(np.mean(img_hist["best_f1"])) if img_hist["best_f1"] else 0.0,
        "image_best_threshold": float(np.mean(img_hist["best_threshold"])) if img_hist["best_threshold"] else threshold,
    }
    return result


def train_loop(
    model: TriEncoderCFMNet,
    train_ds,
    val_ds,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 8,
    device_str: str = "cuda",
    metric_threshold: float = 0.5,
    save_every: int = 5,
    resume: bool = False,
    fm_weight: float = 1.0,
    seg_weight: float = 2.0,
    geo_weight: float = 0.15,
    vsmooth_weight: float = 0.05,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
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
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    ckpt_dir = output_dir / "checkpoint"
    results_dir = output_dir / "results"
    epoch_csv = results_dir / "epoch_metrics.csv"
    final_csv = results_dir / "final_metrics.csv"

    start_epoch = 1
    best_f1 = 0.0
    if resume:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state["epoch"]) + 1
            best_f1 = float(state.get("best_f1", 0.0))

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
        )

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
        }
        append_csv(epoch_csv, row)
        print(row)

        if epoch % save_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:04d}.pt",
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_f1": best_f1},
            )
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            save_checkpoint(
                ckpt_dir / "best.pt",
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_f1": best_f1},
            )

    final = {
        "best_val_f1": best_f1,
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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--metric_threshold", type=float, default=0.5)
    p.add_argument("--val_split_ratio_l4s", type=float, default=0.1)
    p.add_argument("--resize_to", type=int, default=256)
    p.add_argument("--pyramid_width", type=int, default=48)
    p.add_argument("--flow_combine_scale", type=float, default=0.5)
    p.add_argument("--model_flow_steps", type=int, default=0, help="Default FM integration steps at eval if --val_infer_flow_steps unset")
    p.add_argument("--val_infer_flow_steps", type=int, default=-1, help="If >=0, override eval integration steps; -1 uses model_flow_steps")
    p.add_argument("--fm_weight", type=float, default=1.0)
    p.add_argument("--seg_weight", type=float, default=2.0)
    p.add_argument("--geo_weight", type=float, default=0.15)
    p.add_argument("--vsmooth_weight", type=float, default=0.05)
    p.add_argument("--tversky_alpha", type=float, default=0.3)
    p.add_argument("--tversky_beta", type=float, default=0.7)
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

    train_loop(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        output_dir=model_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device_str=args.device,
        metric_threshold=args.metric_threshold,
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
        },
    )


if __name__ == "__main__":
    main()
