from __future__ import annotations

import argparse
from pathlib import Path

from .architectures import build_model
from .datasets import (
    Augment2D,
    AugmentDual2D,
    BijieSingleStreamDataset,
    BijieTwoComposites,
    L4SBinaryDataset,
    L4SDualStreamDataset,
    build_bijie_split,
    build_l4s_split,
)
from .trainer import set_seed, train_model


def build_parser(default_model_name: str, dual_stream: bool = False):
    p = argparse.ArgumentParser(description=f"Train baseline: {default_model_name}")
    p.add_argument("--dataset", type=str, choices=["landslide4sense", "bijie"], required=True)
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=".")
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

    # paper loss settings
    p.add_argument("--tversky_alpha", type=float, default=0.3)
    p.add_argument("--tversky_beta", type=float, default=0.7)
    p.add_argument("--main_weight", type=float, default=1.0)
    p.add_argument("--aux2_weight", type=float, default=0.6)
    p.add_argument("--aux3_weight", type=float, default=0.4)

    # model/data options
    p.add_argument("--model_name", type=str, default=default_model_name)
    p.add_argument("--input_mode_l4s", type=str, default="rgb")
    p.add_argument("--input_mode_bijie", type=str, default="rgb")
    p.add_argument("--val_split_ratio_l4s", type=float, default=0.1)
    p.add_argument("--resize_to", type=int, default=256)
    p.add_argument("--in_channels", type=int, default=3)
    p.add_argument("--n_classes", type=int, default=1)
    p.add_argument("--dual_stream", action="store_true", default=dual_stream)
    return p


def run_single_stream(args):
    set_seed(args.seed)

    def _infer_channels(dataset_name: str, mode: str, fallback: int) -> int:
        m = mode.lower()
        if dataset_name == "landslide4sense":
            mapping = {
                "rgb": 3,
                "ngb": 3,
                "rgb_swir": 6,
                "all14": 14,
                "rgb_ndvi_slope_dem": 6,
            }
            return mapping.get(m, fallback)
        if dataset_name == "bijie":
            mapping = {
                "rgb": 3,
                "rgb_dem": 6,
            }
            return mapping.get(m, fallback)
        return fallback

    inferred_in_channels = (
        _infer_channels(args.dataset, args.input_mode_l4s, args.in_channels)
        if args.dataset == "landslide4sense"
        else _infer_channels(args.dataset, args.input_mode_bijie, args.in_channels)
    )
    model = build_model(args.model_name, in_channels=inferred_in_channels, n_classes=args.n_classes)
    output_dir = Path(args.output_dir).resolve()
    model_out = output_dir / args.dataset / args.model_name

    if args.dataset == "landslide4sense":
        train_ids, val_ids = build_l4s_split(args.dataset_root, val_ratio=args.val_split_ratio_l4s, seed=args.seed)
        train_ds = L4SBinaryDataset(
            args.dataset_root,
            ids=train_ids,
            resize_to=args.resize_to,
            input_mode=args.input_mode_l4s,
            transform=Augment2D(p=0.5),
        )
        val_ds = L4SBinaryDataset(
            args.dataset_root,
            ids=val_ids,
            resize_to=args.resize_to,
            input_mode=args.input_mode_l4s,
            transform=None,
        )
    else:
        train_raw, val_raw, _ = build_bijie_split(args.dataset_root, seed=args.seed)
        train_ds = BijieSingleStreamDataset(
            train_raw, resize_to=args.resize_to, input_mode=args.input_mode_bijie, transform=Augment2D(p=0.5)
        )
        val_ds = BijieSingleStreamDataset(val_raw, resize_to=args.resize_to, input_mode=args.input_mode_bijie, transform=None)

    train_model(
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
        alpha=args.tversky_alpha,
        beta=args.tversky_beta,
        main_weight=args.main_weight,
        aux2_weight=args.aux2_weight,
        aux3_weight=args.aux3_weight,
        dual_stream=False,
        extra_final={
            "dataset": args.dataset,
            "model_name": args.model_name,
            "in_channels": inferred_in_channels,
            "input_mode_l4s": args.input_mode_l4s,
            "input_mode_bijie": args.input_mode_bijie,
        },
    )


def run_dual_stream(args):
    set_seed(args.seed)
    model = build_model(args.model_name, in_channels=args.in_channels, n_classes=args.n_classes)
    output_dir = Path(args.output_dir).resolve()
    model_out = output_dir / args.dataset / args.model_name

    if args.dataset == "landslide4sense":
        train_ids, val_ids = build_l4s_split(args.dataset_root, val_ratio=args.val_split_ratio_l4s, seed=args.seed)
        train_ds = L4SDualStreamDataset(
            args.dataset_root, ids=train_ids, resize_to=args.resize_to, transform=AugmentDual2D(p=0.5)
        )
        val_ds = L4SDualStreamDataset(args.dataset_root, ids=val_ids, resize_to=args.resize_to, transform=None)
    else:
        train_raw, val_raw, _ = build_bijie_split(args.dataset_root, seed=args.seed)
        train_ds = BijieTwoComposites(train_raw, resize_to=args.resize_to, transform=None)
        val_ds = BijieTwoComposites(val_raw, resize_to=args.resize_to, transform=None)

    train_model(
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
        alpha=args.tversky_alpha,
        beta=args.tversky_beta,
        main_weight=args.main_weight,
        aux2_weight=args.aux2_weight,
        aux3_weight=args.aux3_weight,
        dual_stream=True,
        extra_final={"dataset": args.dataset, "model_name": args.model_name},
    )
