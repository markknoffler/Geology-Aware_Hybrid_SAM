import argparse
import csv
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import DualStreamTransform, Landslide4SenseDualStream
from losses import DualStreamLoss
from metrics import pixel_metrics_from_logits
from model import DualStreamGateNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train paper-aligned DiGATe-UNet on Landslide4Sense.")
    parser.add_argument("--dataset_root", type=str, default="/home/user/Desktop/Deep_learning_projects/4PI/dataset")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize_to", type=int, default=256)
    parser.add_argument("--bands", type=str, default="RGB-NDVI-SLOPE-DEM")
    parser.add_argument("--backbone", type=str, default="tf_efficientnet_b4")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze_backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--share_backbone", action="store_true", default=False)
    parser.add_argument("--no-share_backbone", dest="share_backbone", action="store_false")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--use_input_adapter", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--val_split_ratio", type=float, default=0.1)
    parser.add_argument("--val_split_seed", type=int, default=42)
    parser.add_argument("--tversky_alpha", type=float, default=0.6)
    parser.add_argument("--tversky_beta", type=float, default=0.4)
    parser.add_argument("--main_weight", type=float, default=1.0)
    parser.add_argument("--aux2_weight", type=float, default=0.6)
    parser.add_argument("--aux3_weight", type=float, default=0.4)
    parser.add_argument("--reg_weight", type=float, default=1e-3)
    parser.add_argument("--metric_threshold", type=float, default=0.6)
    return parser.parse_args()


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
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def prep_batch(batch, device: torch.device):
    x1 = batch["stream_a"].float().to(device, non_blocking=True)
    x2 = batch["stream_b"].float().to(device, non_blocking=True)
    y = batch["mask"]
    if y.dtype.is_floating_point:
        y = y.round().long()
    y = y.to(device, non_blocking=True)
    return x1, x2, y


def run_epoch(
    loader,
    model,
    criterion,
    device: torch.device,
    threshold: float = 0.5,
    optimizer=None,
    training: bool = False,
):
    model.train() if training else model.eval()
    prefix = "Train" if training else "Valid"

    losses = []
    metrics = {"acc": [], "precision": [], "recall": [], "f1": [], "iou": []}
    progress = tqdm(loader, desc=prefix, leave=False)

    for batch in progress:
        x1, x2, y = prep_batch(batch, device)

        with torch.set_grad_enabled(training):
            main, aux2, aux3, reg_tuple = model(x1, x2)
            loss_dict = criterion(main, aux2, aux3, reg_tuple, y)
            loss = loss_dict["loss"]

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        pix = pixel_metrics_from_logits(main, y, threshold=threshold)
        losses.append(float(loss.item()))
        for key in metrics:
            metrics[key].append(float(pix[key]))

        progress.set_postfix(
            loss=f"{losses[-1]:.4f}",
            f1=f"{metrics['f1'][-1]:.4f}",
            iou=f"{metrics['iou'][-1]:.4f}",
            reg=f"{float(loss_dict['loss_reg'].item()):.3e}",
        )

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        **{key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()},
    }


def build_dataloaders(args):
    train_ds = Landslide4SenseDualStream(
        data_root=args.dataset_root,
        split="train",
        bands=args.bands,
        resize_to=args.resize_to,
        transform=DualStreamTransform(p=0.5),
    )
    valid_ds = Landslide4SenseDualStream(
        data_root=args.dataset_root,
        split="valid",
        bands=args.bands,
        resize_to=args.resize_to,
        transform=None,
    )

    if valid_ds.has_mask:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        return train_loader, valid_loader

    val_ratio = float(args.val_split_ratio)
    n_total = len(train_ds)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_val = min(n_val, n_total - 1)
    generator = torch.Generator().manual_seed(args.val_split_seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    val_indices = perm[:n_val]
    train_indices = perm[n_val:]

    train_subset = Subset(train_ds, train_indices)
    valid_subset = Subset(
        Landslide4SenseDualStream(
            data_root=args.dataset_root,
            split="train",
            bands=args.bands,
            resize_to=args.resize_to,
            transform=None,
        ),
        val_indices,
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        valid_subset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    return train_loader, valid_loader


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    ckpt_dir = output_dir / "checkpoint"
    results_dir = output_dir / "results"
    epoch_csv = results_dir / "epoch_metrics.csv"
    final_csv = results_dir / "final_metrics.csv"

    train_loader, valid_loader = build_dataloaders(args)

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
        train_metrics = run_epoch(
            train_loader,
            model,
            criterion,
            device,
            threshold=args.metric_threshold,
            optimizer=optimizer,
            training=True,
        )
        val_metrics = run_epoch(
            valid_loader,
            model,
            criterion,
            device,
            threshold=args.metric_threshold,
            optimizer=None,
            training=False,
        )

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

    append_csv(
        final_csv,
        {
            "best_val_f1": best_f1,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "backbone": args.backbone,
            "pretrained": args.pretrained,
            "freeze_backbone": args.freeze_backbone,
            "resize_to": args.resize_to,
            "bands": args.bands,
            "tversky_alpha": args.tversky_alpha,
            "tversky_beta": args.tversky_beta,
            "main_weight": args.main_weight,
            "aux2_weight": args.aux2_weight,
            "aux3_weight": args.aux3_weight,
            "reg_weight": args.reg_weight,
            "metric_threshold": args.metric_threshold,
            "dataset_root": args.dataset_root,
        },
    )


if __name__ == "__main__":
    main()
