from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader, random_split
from tqdm import tqdm

from bijie_dataset import BijieRawDataset, BijieTwoComposites, DualStreamTransformBijie
from losses import DualStreamLoss
from metrics import pixel_metrics_from_logits
from model import DualStreamGateNet


def parse_args():
    p = argparse.ArgumentParser(description="Train DiGATe-UNet (dual-stream) on Bijie dataset.")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./bijie_outputs")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)

    # Paper hyperparameters
    p.add_argument("--tversky_alpha", type=float, default=0.3)
    p.add_argument("--tversky_beta", type=float, default=0.7)
    p.add_argument("--main_weight", type=float, default=1.0)
    p.add_argument("--aux2_weight", type=float, default=0.6)
    p.add_argument("--aux3_weight", type=float, default=0.4)
    p.add_argument("--reg_weight", type=float, default=1e-3)
    p.add_argument("--metric_threshold", type=float, default=0.5)

    # Model settings
    p.add_argument("--backbone", type=str, default="tf_efficientnet_b4")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--freeze_backbone", action="store_true", default=True)
    p.add_argument("--share_backbone", action="store_true", default=True)  # paper says siamese shared weights
    p.add_argument("--use_input_adapter", action="store_true", default=False)
    p.add_argument("--pretrained_path", type=str, default=None)

    # Resume/checkpoint
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in checkpoint/.")
    p.add_argument("--save_every", type=int, default=5)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def latest_checkpoint(ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def save_checkpoint(path: Path, state: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def append_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def prep_batch(batch, device: torch.device):
    x1 = batch["stream_a"].float().to(device, non_blocking=True)
    x2 = batch["stream_b"].float().to(device, non_blocking=True)
    y = batch["mask"].float()
    if y.dtype.is_floating_point:
        y = y.round().long()
    y = y.to(device, non_blocking=True)
    if y.dim() == 3:
        y = y.unsqueeze(1)
    return x1, x2, y


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device, threshold: float):
    model.eval()
    losses = []

    # pixel totals per-image then average (matches your existing logging style)
    metric_accum = {"acc": [], "precision": [], "recall": [], "f1": [], "iou": []}

    for batch in tqdm(loader, desc="Val", leave=False):
        x1, x2, y = prep_batch(batch, device)
        main, aux2, aux3, reg_tuple = model(x1, x2)
        loss_dict = criterion(main, aux2, aux3, reg_tuple, y)
        losses.append(float(loss_dict["loss"].item()))

        pix = pixel_metrics_from_logits(main, y, threshold=threshold)
        for k in metric_accum:
            metric_accum[k].append(float(pix[k]))

    out = {k: float(np.mean(v)) if v else 0.0 for k, v in metric_accum.items()}
    out["loss"] = float(np.mean(losses)) if losses else 0.0
    return out


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device, threshold: float):
    model.train()
    losses = []
    metric_accum = {"acc": [], "precision": [], "recall": [], "f1": [], "iou": []}

    for batch in tqdm(loader, desc="Train", leave=False):
        x1, x2, y = prep_batch(batch, device)

        main, aux2, aux3, reg_tuple = model(x1, x2)
        loss_dict = criterion(main, aux2, aux3, reg_tuple, y)
        loss = loss_dict["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        pix = pixel_metrics_from_logits(main, y, threshold=threshold)
        for k in metric_accum:
            metric_accum[k].append(float(pix[k]))

    out = {k: float(np.mean(v)) if v else 0.0 for k, v in metric_accum.items()}
    out["loss"] = float(np.mean(losses)) if losses else 0.0
    return out


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    ckpt_dir = output_dir / "checkpoint"
    results_dir = output_dir / "results"
    epoch_csv = results_dir / "epoch_metrics.csv"
    final_csv = results_dir / "final_metrics.csv"

    dataset_root = Path(args.dataset_root).resolve()
    # Expected:
    #   dataset_root/landslide/{image,dem,mask}
    #   dataset_root/non-landslide/{image,dem}
    landslide_raw = BijieRawDataset(dataset_root / "landslide", phase="landslide")
    nonlandslide_raw = BijieRawDataset(dataset_root / "non-landslide", phase="non-landslide")

    generator = torch.Generator().manual_seed(args.seed)

    def split(ds):
        n = len(ds)
        ratios = (0.7, 0.2, 0.1)
        sizes = [int(r * n) for r in ratios]
        sizes[2] = n - sum(sizes[:2])
        return random_split(ds, sizes, generator=generator)

    tl, vl, sl = split(landslide_raw)
    tn, vn, sn = split(nonlandslide_raw)

    train_raw = ConcatDataset([tl, tn])
    val_raw = ConcatDataset([vl, vn])
    test_raw = ConcatDataset([sl, sn])

    train_ds = BijieTwoComposites(train_raw, resize_to=256, transform=DualStreamTransformBijie(p=0.5))
    val_ds = BijieTwoComposites(val_raw, resize_to=256, transform=None)
    test_ds = BijieTwoComposites(test_raw, resize_to=256, transform=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = DualStreamGateNet(
        n_classes=1,
        backbone=args.backbone,
        n_channels=3,
        n_channels_b=3,
        pretrained=args.pretrained,
        pretrained_path=args.pretrained_path,
        use_input_adapter=args.use_input_adapter,
        freeze_backbone=args.freeze_backbone,
        share_backbone=args.share_backbone,
    ).to(device)

    criterion = DualStreamLoss(
        alpha=args.tversky_alpha,
        beta=args.tversky_beta,
        main_weight=args.main_weight,
        aux2_weight=args.aux2_weight,
        aux3_weight=args.aux3_weight,
        reg_weight=args.reg_weight,
    )
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_f1 = 0.0

    if args.resume:
        ckpt = latest_checkpoint(ckpt_dir)
        if ckpt is not None:
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state["epoch"]) + 1
            best_f1 = float(state.get("best_f1", 0.0))

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, threshold=args.metric_threshold)
        val_metrics = evaluate(model, val_loader, criterion, device, threshold=args.metric_threshold)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_iou": val_metrics["iou"],
        }
        append_csv(epoch_csv, row)
        print(row)

        if epoch % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:04d}.pt",
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_f1": best_f1},
            )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(
                ckpt_dir / "best.pt",
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "best_f1": best_f1},
            )

    # Optionally evaluate test at the end (not required, but cheap)
    # You can comment this out if you want purely training CSV.
    # test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # test_metrics = evaluate(model, test_loader, criterion, device, threshold=args.metric_threshold)

    append_csv(
        final_csv,
        {
            "best_val_f1": best_f1,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "backbone": args.backbone,
            "tversky_alpha": args.tversky_alpha,
            "tversky_beta": args.tversky_beta,
            "metric_threshold": args.metric_threshold,
            "dataset_root": str(dataset_root),
        },
    )


if __name__ == "__main__":
    main()

