import argparse
import csv
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from terrascope.core.model import Terrascope, build_terrascope_b
from terrascope.data.factory import build_datasets
from terrascope.losses.composite import LossWeights, composite_segmentation_loss
from terrascope.utils.checkpointing import latest_checkpoint
from terrascope.utils.metrics import segmentation_metrics_from_logits


class TerrascopeWrapper(nn.Module):
    def __init__(self, model: Terrascope):
        super().__init__()
        self.model = model

    def forward(self, rgb, dem, image_pe, dense_prompt, return_aux: bool = False):
        return self.model(rgb, dem, image_pe, dense_prompt, False, return_aux)


def parse_args():
    p = argparse.ArgumentParser(description="Train Terrascope (dual-stream, from scratch)")
    p.add_argument("--dataset", choices=["landslide4sense", "bijie"], required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--results-dir", default=".")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-size", type=int, default=512)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--resume", default="auto", help="'auto' | path/to/ckpt.pt | 'none'")

    p.add_argument("--w-bce", type=float, default=0.0)
    p.add_argument("--w-dice", type=float, default=0.0)
    p.add_argument("--w-tversky", type=float, default=1.0)
    p.add_argument("--w-focal", type=float, default=0.0)
    p.add_argument("--w-soft-iou", type=float, default=0.0)
    p.add_argument("--w-boundary", type=float, default=0.0)
    p.add_argument("--w-tgbc", type=float, default=0.05)
    p.add_argument("--w-cscd", type=float, default=0.05)
    p.add_argument("--tversky-alpha", type=float, default=0.7)
    p.add_argument("--tversky-beta", type=float, default=0.3)
    return p.parse_args()


def loss_weights_from_args(args) -> LossWeights:
    return LossWeights(
        bce=args.w_bce,
        dice=args.w_dice,
        tversky=args.w_tversky,
        focal=args.w_focal,
        soft_iou=args.w_soft_iou,
        boundary=args.w_boundary,
        tgbc=args.w_tgbc,
        cscd=args.w_cscd,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
    )


def ensure_dirs(results_root: Path, dataset_name: str):
    output_dir = (results_root.resolve() / dataset_name / "terrascope")
    ckpt = output_dir / "checkpoint"
    results = output_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return output_dir, ckpt, results


def _embed_hw(rgb: torch.Tensor):
    return rgb.shape[2] // 16, rgb.shape[3] // 16


def _append_csv(path: Path, row: Dict[str, float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _save_checkpoint(path: Path, epoch: int, model: Terrascope, optimizer, best_val_f1: float, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "args": vars(args),
        },
        path,
    )


def _load_checkpoint(path: Path, model: Terrascope, optimizer) -> tuple[int, float]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(path, map_location="cpu")

    # Support both baseline-style and prior Terrascope checkpoint keys.
    model_key = "model" if "model" in state else "model_state_dict"
    optim_key = "optimizer" if "optimizer" in state else "optimizer_state_dict"
    model.load_state_dict(state[model_key], strict=False)
    optimizer.load_state_dict(state[optim_key])
    start_epoch = int(state.get("epoch", 0)) + 1
    best_val_f1 = float(state.get("best_val_f1", state.get("best_f1", 0.0)))
    return start_epoch, best_val_f1


def run_epoch(
    model,
    loader,
    device,
    net: Terrascope,
    lw: LossWeights,
    training: bool,
    optimizer=None,
):
    model.train() if training else model.eval()
    losses = []
    metric_hist = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "iou": [],
        "auroc": [],
        "auprc": [],
        "best_f1": [],
        "best_threshold": [],
    }
    need_aux = lw.cscd > 0
    prompts = net.prompts

    with torch.set_grad_enabled(training):
        for rgb, dem, mask, _ in tqdm.tqdm(loader, desc="Train" if training else "Val", leave=False):
            rgb = rgb.to(device, non_blocking=True)
            dem = dem.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            b = rgb.size(0)
            eh, ew = _embed_hw(rgb)
            image_pe = prompts.dense_pe((eh, ew), rgb.device).expand(b, -1, eh, ew)
            dense = prompts.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(b, -1, eh, ew)
            logits, _, aux = model(rgb, dem, image_pe, dense, return_aux=need_aux)
            logits = logits[:, 0:1]
            logits = F.interpolate(logits, mask.shape[-2:], mode="bilinear", align_corners=False)
            loss, _ = composite_segmentation_loss(logits, mask, dem, aux, lw)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            losses.append(float(loss.item()))

            batch_m = segmentation_metrics_from_logits(logits, mask)
            for k in metric_hist:
                v = float(batch_m.get(k, 0.0))
                if v == v:  # ignore NaN
                    metric_hist[k].append(v)

    out = {
        "loss": float(sum(losses) / max(1, len(losses))),
        "accuracy": float(sum(metric_hist["accuracy"]) / max(1, len(metric_hist["accuracy"]))),
        "precision": float(sum(metric_hist["precision"]) / max(1, len(metric_hist["precision"]))),
        "recall": float(sum(metric_hist["recall"]) / max(1, len(metric_hist["recall"]))),
        "f1": float(sum(metric_hist["f1"]) / max(1, len(metric_hist["f1"]))),
        "iou": float(sum(metric_hist["iou"]) / max(1, len(metric_hist["iou"]))),
        "auroc": float(sum(metric_hist["auroc"]) / max(1, len(metric_hist["auroc"]))),
        "auprc": float(sum(metric_hist["auprc"]) / max(1, len(metric_hist["auprc"]))),
        "image_best_f1": float(sum(metric_hist["best_f1"]) / max(1, len(metric_hist["best_f1"]))),
        "image_best_threshold": float(
            sum(metric_hist["best_threshold"]) / max(1, len(metric_hist["best_threshold"]))
        ),
    }
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir, ckpt_dir, results_dir = ensure_dirs(Path(args.results_dir), args.dataset)
    epoch_csv = results_dir / "epoch_metrics.csv"
    final_csv = results_dir / "final_metrics.csv"
    lw = loss_weights_from_args(args)

    train_ds, val_ds, test_ds = build_datasets(args.dataset, args.dataset_root, args.target_size, args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if val_ds is not None
        else None
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    base_model = build_terrascope_b().to(device)
    model = TerrascopeWrapper(base_model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_val_f1 = 0.0
    if args.resume != "none":
        if args.resume == "auto":
            ckpt = latest_checkpoint(ckpt_dir)
        else:
            ckpt = Path(args.resume)
        if ckpt is not None and ckpt.exists():
            start_epoch, best_val_f1 = _load_checkpoint(ckpt, base_model, optimizer)
            print(f"Resumed from {ckpt} at epoch {start_epoch}.")

    for epoch in range(start_epoch, args.epochs + 1):
        e0 = time.time()
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            net=base_model,
            lw=lw,
            training=True,
            optimizer=optimizer,
        )
        train_metrics["fps"] = len(train_ds) / max(1e-6, (time.time() - e0))

        if val_loader is not None:
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                net=base_model,
                lw=lw,
                training=False,
                optimizer=None,
            )
        else:
            val_metrics = {k: 0.0 for k in train_metrics.keys() if k != "fps"}
            val_metrics["fps"] = 0.0

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_iou": train_metrics["iou"],
            "train_auroc": train_metrics["auroc"],
            "train_auprc": train_metrics["auprc"],
            "train_image_best_f1": train_metrics["image_best_f1"],
            "train_image_best_threshold": train_metrics["image_best_threshold"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_iou": val_metrics["iou"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_image_best_f1": val_metrics["image_best_f1"],
            "val_image_best_threshold": val_metrics["image_best_threshold"],
            "train_fps": train_metrics["fps"],
            "trainable_params_m": sum(p.numel() for p in base_model.parameters() if p.requires_grad) / 1e6,
        }
        _append_csv(epoch_csv, row)
        print(row)

        if epoch % args.save_every == 0:
            _save_checkpoint(ckpt_dir / f"epoch_{epoch:04d}.pt", epoch, base_model, optimizer, best_val_f1, args)

        if val_loader is not None and val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            _save_checkpoint(ckpt_dir / "best.pt", epoch, base_model, optimizer, best_val_f1, args)

    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        _, best_val_f1 = _load_checkpoint(best_path, base_model, optimizer)
    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        net=base_model,
        lw=lw,
        training=False,
        optimizer=None,
    )
    _append_csv(
        final_csv,
        {
            "best_val_f1": best_val_f1,
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_iou": test_metrics["iou"],
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
            "test_image_best_f1": test_metrics["image_best_f1"],
            "test_image_best_threshold": test_metrics["image_best_threshold"],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dataset": args.dataset,
            "dataset_root": args.dataset_root,
            "target_size": args.target_size,
            "save_every": args.save_every,
            "resume": args.resume,
            "output_dir": str(output_dir),
            "w_bce": args.w_bce,
            "w_dice": args.w_dice,
            "w_tversky": args.w_tversky,
            "w_focal": args.w_focal,
            "w_soft_iou": args.w_soft_iou,
            "w_boundary": args.w_boundary,
            "w_tgbc": args.w_tgbc,
            "w_cscd": args.w_cscd,
            "tversky_alpha": args.tversky_alpha,
            "tversky_beta": args.tversky_beta,
        },
    )


if __name__ == "__main__":
    main()
