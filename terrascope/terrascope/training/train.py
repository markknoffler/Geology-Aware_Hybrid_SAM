import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader

from terrascope.core.model import Terrascope, build_terrascope_b
from terrascope.data.factory import build_datasets
from terrascope.losses.composite import LossWeights, composite_segmentation_loss
from terrascope.utils.checkpointing import latest_checkpoint, load_checkpoint, save_checkpoint
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
    p.add_argument("--results-dir", required=True)
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
    root = results_root / dataset_name
    ckpt = root / "checkpoints"
    metrics = root / "metrics"
    root.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    return root, ckpt, metrics


def _embed_hw(rgb: torch.Tensor):
    return rgb.shape[2] // 16, rgb.shape[3] // 16


def evaluate(model, loader, device, net: Terrascope, lw: LossWeights):
    model.eval()
    losses = []
    all_logits = []
    all_masks = []
    need_aux = lw.cscd > 0
    prompts = net.prompts
    with torch.no_grad():
        for rgb, dem, mask, _ in tqdm.tqdm(loader, desc="eval", leave=False):
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
            losses.append(float(loss.item()))
            all_logits.append(logits.cpu())
            all_masks.append(mask.cpu())

    logits = torch.cat(all_logits, dim=0)
    masks = torch.cat(all_masks, dim=0)
    m = segmentation_metrics_from_logits(logits, masks)
    m["loss"] = float(sum(losses) / max(1, len(losses)))
    return m


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_root, ckpt_dir, metrics_dir = ensure_dirs(Path(args.results_dir), args.dataset)
    metrics_csv = metrics_dir / "metrics.csv"
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

    start_epoch = 0
    if args.resume != "none":
        if args.resume == "auto":
            ckpt = latest_checkpoint(ckpt_dir)
        else:
            ckpt = Path(args.resume)
        if ckpt is not None and ckpt.exists():
            start_epoch = load_checkpoint(ckpt, base_model, optimizer)

    prompts = base_model.prompts

    header = [
        "epoch",
        "phase",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "iou",
        "dice",
        "auroc",
        "auprc",
        "best_f1",
        "best_threshold",
        "fps",
        "peak_memory_mb",
        "gflops",
        "trainable_params_m",
    ]
    if not metrics_csv.exists():
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow(header)

    need_aux = lw.cscd > 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        losses = []
        e0 = time.time()
        for rgb, dem, mask, _ in tqdm.tqdm(train_loader, desc=f"train {epoch:03d}", leave=False):
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = float(sum(losses) / max(1, len(losses)))
        train_fps = len(train_ds) / max(1e-6, (time.time() - e0))

        test_metrics = evaluate(model, test_loader, device, base_model, lw)
        test_metrics["fps"] = train_fps
        test_metrics["trainable_params_m"] = sum(p.numel() for p in base_model.parameters() if p.requires_grad) / 1e6

        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, base_model, lw)
        else:
            val_metrics = None

        with open(metrics_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    "train",
                    train_loss,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    train_fps,
                    "",
                    "",
                    test_metrics["trainable_params_m"],
                ]
            )
            writer.writerow([epoch, "test"] + [test_metrics[k] for k in header[2:]])
            if val_metrics is not None:
                writer.writerow([epoch, "val"] + [val_metrics[k] for k in header[2:]])

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", epoch, base_model, optimizer)


if __name__ == "__main__":
    main()
