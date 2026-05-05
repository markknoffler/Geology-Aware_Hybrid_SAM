import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import Landslide4SenseBiFusion
from losses import BiFusionLoss, dice_loss_from_logits
from metrics import assd_hd_from_logits, expected_calibration_error, metrics_from_confusion, pixel_confusion_from_logits
from model import BiFusionLDSeg


def parse_args():
    p = argparse.ArgumentParser(description="Train BiFusion-LDSeg on Landslide4Sense (ablation setting).")
    p.add_argument("--dataset_root", type=str, default="/home/user/Desktop/Deep_learning_projects/4PI/dataset")
    p.add_argument("--output_dir", type=str, default=".")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=7e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--diffusion_steps", type=int, default=1000)
    p.add_argument("--ddim_steps", type=int, default=10)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--rgb_indices", type=int, nargs=3, default=[3, 2, 1])
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--test_split", type=float, default=0.1)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--gamma", type=float, default=2.0)
    p.add_argument("--lambda_diff", type=float, default=1.0)
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def latest_checkpoint(ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"))
    return ckpts[-1] if ckpts else None


def save_checkpoint(path: Path, state: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    if val_ratio <= 0 or test_ratio <= 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("val_split and test_split must be >0 and sum to <1.")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))
    if n_val + n_test >= n:
        n_test = max(1, n - n_val - 1)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError("Not enough samples to build train/val/test split from TrainData.")

    val_idx = perm[:n_val]
    test_idx = perm[n_val : n_val + n_test]
    train_idx = perm[n_val + n_test :]
    return train_idx, val_idx, test_idx


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        x = batch["image"].to(device)
        y = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(x, y)
        loss_dict = criterion(out, y)
        loss = loss_dict["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, criterion, device: torch.device, ddim_steps: int):
    model.eval()
    bce = torch.nn.BCEWithLogitsLoss()

    sum_tp = sum_fp = sum_fn = sum_tn = 0
    ece_vals: List[float] = []
    assd_vals: List[float] = []
    hd_vals: List[float] = []
    loss_meter = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        out = model(x, sample_steps=ddim_steps)
        logits = out["main"]

        loss = bce(logits, y) + criterion.gamma * dice_loss_from_logits(logits, y)
        loss_meter += float(loss.item())
        n_batches += 1

        conf = pixel_confusion_from_logits(logits, y, threshold=0.5)
        sum_tp += conf["tp"]
        sum_fp += conf["fp"]
        sum_fn += conf["fn"]
        sum_tn += conf["tn"]

        ece_vals.append(expected_calibration_error(logits, y))
        geo = assd_hd_from_logits(logits, y, threshold=0.5)
        if geo["assd"] == geo["assd"]:
            assd_vals.append(geo["assd"])
        if geo["hd"] == geo["hd"]:
            hd_vals.append(geo["hd"])

    stats = metrics_from_confusion(sum_tp, sum_fp, sum_fn, sum_tn)
    stats["loss"] = loss_meter / max(1, n_batches)
    stats["assd"] = float(sum(assd_vals) / max(1, len(assd_vals)))
    stats["hd"] = float(sum(hd_vals) / max(1, len(hd_vals)))
    stats["ece"] = float(sum(ece_vals) / max(1, len(ece_vals)))
    return stats


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).resolve()
    checkpoints_dir = output_dir / "checkpoints"
    results_dir = output_dir / "results"
    epoch_csv = results_dir / "epoch_metrics.csv"
    final_csv = results_dir / "final_metrics.csv"

    train_root = Path(args.dataset_root) / "TrainData"
    image_paths = sorted((train_root / "img").glob("*.h5"))
    mask_dir = train_root / "mask"
    if not image_paths or not mask_dir.exists():
        raise FileNotFoundError("TrainData/img or TrainData/mask not found at dataset_root.")

    full_ds = Landslide4SenseBiFusion(
        image_paths=image_paths,
        mask_dir=mask_dir,
        rgb_indices=tuple(args.rgb_indices),
    )
    train_idx, val_idx, test_idx = split_indices(len(full_ds), args.val_split, args.test_split, args.split_seed)
    print(f"Using TrainData-only split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(Subset(full_ds, test_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = BiFusionLDSeg(
        in_channels=3,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
        diffusion_steps=args.diffusion_steps,
    ).to(device)
    criterion = BiFusionLoss(gamma=args.gamma, lambda_diff=args.lambda_diff)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 1
    best_val_dsc = 0.0
    best_epoch = 0

    if args.resume:
        ckpt = latest_checkpoint(checkpoints_dir)
        if ckpt is not None:
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state["epoch"]) + 1
            best_val_dsc = float(state.get("best_val_dsc", 0.0))
            best_epoch = int(state.get("best_epoch", 0))
            print(f"Resumed from {ckpt} at epoch {start_epoch}.")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, criterion, device, args.ddim_steps)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "val_f1": val["f1"],
            "val_iou": val["iou"],
            "val_dsc": val["dsc"],
            "val_assd": val["assd"],
            "val_hd": val["hd"],
            "val_ece": val["ece"],
        }
        append_csv(epoch_csv, row)
        print(row)

        if epoch % args.save_every == 0:
            save_checkpoint(
                checkpoints_dir / f"epoch_{epoch:04d}.pt",
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_dsc": best_val_dsc,
                    "best_epoch": best_epoch,
                    "args": vars(args),
                },
            )

        if val["dsc"] > best_val_dsc:
            best_val_dsc = val["dsc"]
            best_epoch = epoch
            save_checkpoint(
                checkpoints_dir / "best.pt",
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_dsc": best_val_dsc,
                    "best_epoch": best_epoch,
                    "args": vars(args),
                },
            )

    best_path = checkpoints_dir / "best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"])

    test = evaluate(model, test_loader, criterion, device, args.ddim_steps)
    append_csv(
        final_csv,
        {
            "best_epoch": best_epoch,
            "best_val_dsc": best_val_dsc,
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "test_precision": test["precision"],
            "test_recall": test["recall"],
            "test_f1": test["f1"],
            "test_iou": test["iou"],
            "test_dsc": test["dsc"],
            "test_assd": test["assd"],
            "test_hd": test["hd"],
            "test_ece": test["ece"],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dataset_root": args.dataset_root,
            "split_train": len(train_idx),
            "split_val": len(val_idx),
            "split_test": len(test_idx),
        },
    )
    print("Final test metrics:", test)


if __name__ == "__main__":
    main()
