from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import SegmentationLoss
from .metrics import image_level_metrics_from_logits, pixel_metrics_from_logits


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


def _prep_batch(batch, device):
    x = batch["image"].float().to(device, non_blocking=True)
    y = batch["mask"].float().to(device, non_blocking=True)
    if y.dim() == 3:
        y = y.unsqueeze(1)
    return x, y


def _prep_batch_dual(batch, device):
    x1 = batch["stream_a"].float().to(device, non_blocking=True)
    x2 = batch["stream_b"].float().to(device, non_blocking=True)
    y = batch["mask"].float().to(device, non_blocking=True)
    if y.dim() == 3:
        y = y.unsqueeze(1)
    return x1, x2, y


def _forward(model, x, x2=None):
    return model(x, x2) if x2 is not None else model(x)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    threshold: float,
    training: bool,
    optimizer=None,
    dual_stream: bool = False,
):
    model.train() if training else model.eval()
    losses = []
    pix_hist = {"acc": [], "precision": [], "recall": [], "f1": [], "iou": []}
    img_hist = {"auroc": [], "auprc": [], "best_f1": [], "best_threshold": []}

    pbar = tqdm(loader, desc="Train" if training else "Val", leave=False)
    for batch in pbar:
        if dual_stream:
            x1, x2, y = _prep_batch_dual(batch, device)
            outputs = _forward(model, x1, x2)
        else:
            x, y = _prep_batch(batch, device)
            outputs = _forward(model, x)

        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        else:
            logits = outputs

        with torch.set_grad_enabled(training):
            loss = criterion(outputs, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        losses.append(float(loss.item()))
        pix = pixel_metrics_from_logits(logits, y, threshold=threshold)
        for k in pix_hist:
            pix_hist[k].append(float(pix[k]))
        img = image_level_metrics_from_logits(logits, y, prob_thr_for_instances=threshold, min_area=20)
        for k in img_hist:
            img_hist[k].append(float(img[k]))

        pbar.set_postfix(loss=f"{losses[-1]:.4f}", f1=f"{pix_hist['f1'][-1]:.4f}", iou=f"{pix_hist['iou'][-1]:.4f}")

    out = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        **{k: float(np.mean(v)) if v else 0.0 for k, v in pix_hist.items()},
        "auroc": float(np.mean(img_hist["auroc"])) if img_hist["auroc"] else 0.0,
        "auprc": float(np.mean(img_hist["auprc"])) if img_hist["auprc"] else 0.0,
        "image_best_f1": float(np.mean(img_hist["best_f1"])) if img_hist["best_f1"] else 0.0,
        "image_best_threshold": float(np.mean(img_hist["best_threshold"])) if img_hist["best_threshold"] else threshold,
    }
    return out


def train_model(
    model,
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
    alpha: float = 0.3,
    beta: float = 0.7,
    main_weight: float = 1.0,
    aux2_weight: float = 0.6,
    aux3_weight: float = 0.4,
    dual_stream: bool = False,
    extra_final: Optional[Dict] = None,
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    criterion = SegmentationLoss(
        alpha=alpha,
        beta=beta,
        main_weight=main_weight,
        aux2_weight=aux2_weight,
        aux3_weight=aux3_weight,
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

    for epoch in range(start_epoch, epochs + 1):
        train_m = run_epoch(
            model,
            train_loader,
            criterion,
            device=device,
            threshold=metric_threshold,
            training=True,
            optimizer=optimizer,
            dual_stream=dual_stream,
        )
        val_m = run_epoch(
            model,
            val_loader,
            criterion,
            device=device,
            threshold=metric_threshold,
            training=False,
            optimizer=None,
            dual_stream=dual_stream,
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
        "tversky_alpha": alpha,
        "tversky_beta": beta,
        "main_weight": main_weight,
        "aux2_weight": aux2_weight,
        "aux3_weight": aux3_weight,
    }
    if extra_final:
        final.update(extra_final)
    append_csv(final_csv, final)
