#!/usr/bin/env python3
"""
Image-level landslide vs non-landslide evaluation for TriEncoderCFMNet.

Loads the checkpoint at the best validation epoch (from ablation summary CSV),
runs inference on the **same validation split** used during training, and writes:

  - Combined summary table (Bijie + Landslide4Sense in one CSV)
  - Per-image score dumps (one CSV per dataset)
  - Score-separation figure (histograms)

Bijie: validation concat includes ``landslide/`` (masked positives) and
``non-landslide/`` (implicit all-zero GT masks).

Landslide4Sense: validation tiles are all competition training chips; GT
"non-landslide" means **empty mask** tiles only (no separate negative image folder).

Run on your server (paths are examples):

  python SAM/model_architecture_cfm_landseg/eval/eval_landslide_presence.py \\
    --bijie-root /path/to/Bijie-landslide-dataset \\
    --l4s-root /path/to/Landslide4Sense \\
    --bijie-summary SAM/resources/results/bijie_ablation_report/bijie_best_validation_summary.csv \\
    --l4s-summary SAM/resources/results/l4s_ablation_report/landslide4sense_best_validation_summary.csv \\
    --bijie-run-dir SAM/runs/bijie/tri_encoder_cfm_v2 \\
    --l4s-run-dir SAM/runs/landslide4sense/tri_encoder_cfm_v2
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def _resolve_code_root() -> Path:
    """Directory containing model_architecture_cfm_landseg, ablation_study, and runs."""
    here = Path(__file__).resolve()
    for base in (here.parents[1], here.parents[2]):
        if (base / "ablation_study" / "baseline_models").is_dir():
            return base
        if (base.parent / "ablation_study" / "baseline_models").is_dir():
            return base.parent
    return here.parents[2]


_CODE_ROOT = _resolve_code_root()
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
_BASELINE = _CODE_ROOT / "ablation_study" / "baseline_models"
if str(_BASELINE) not in sys.path:
    sys.path.insert(0, str(_BASELINE))

from common.datasets import build_bijie_split, build_l4s_split
from common.metrics import (
    _binarize,
    _instance_scores,
    _mask_to_instances,
    _pr_curve,
    _roc_curve,
    pixel_metrics_from_logits,
)

from model_architecture_cfm_landseg.data.datasets import BijieTripleStreamDataset, L4STripleStreamDataset
from model_architecture_cfm_landseg.training.train import prep_batch_triplet
from model_architecture_cfm_landseg.tri_cfm_net import TriEncoderCFMNet

DEFAULT_MODEL_ID = "tri_encoder_cfm_v2"


def _default_results_root() -> Path:
    for candidate in (
        _CODE_ROOT / "resources" / "results" / "landslide_presence_report",
        _CODE_ROOT.parent / "resources" / "results" / "landslide_presence_report",
    ):
        if candidate.parent.is_dir():
            return candidate
    return _CODE_ROOT / "resources" / "results" / "landslide_presence_report"


RESULTS_ROOT = _default_results_root()


@dataclass
class ImageRecord:
    dataset: str
    split: str
    index: int
    gt_present: int
    pred_present_instance: int
    pred_present_area: int
    image_score: float
    max_prob: float
    pred_fg_fraction: float
    pixel_f1: float
    pixel_iou: float
    pixel_precision: float
    pixel_recall: float


def read_summary_best_epoch(summary_csv: Path, model_id: str) -> int:
    with summary_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("model_id") or "").strip() == model_id:
                return int(float(row["best_epoch"]))
    raise SystemExit(f"model_id={model_id!r} not found in {summary_csv}")


def resolve_checkpoint(run_dir: Path, epoch: int, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"Checkpoint not found: {p}")
        return p
    # Training saves under ``checkpoint/`` (singular); see training/train.py.
    for sub in ("checkpoint", "checkpoints"):
        ckpt_dir = run_dir / sub
        if not ckpt_dir.is_dir():
            continue
        epoch_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
        if epoch_path.is_file():
            return epoch_path
        best_path = ckpt_dir / "best.pt"
        if best_path.is_file():
            print(
                f"WARNING: {epoch_path} missing; using {best_path} "
                f"(best-metric checkpoint, epoch inside file may differ from summary {epoch})."
            )
            return best_path
    raise SystemExit(
        f"No checkpoint under {run_dir}/checkpoint or {run_dir}/checkpoints for epoch {epoch}"
    )


def load_model_from_checkpoint(
    ckpt_path: Path,
    *,
    ctx_ch: int,
    pyramid_width: int,
    flow_combine_scale: float,
    model_flow_steps: int,
    device: torch.device,
) -> Tuple[TriEncoderCFMNet, dict]:
    state = torch.load(ckpt_path, map_location=device)
    model = TriEncoderCFMNet(
        rgb_ch=3,
        dem_ch=1,
        ctx_ch=ctx_ch,
        pyramid_width=pyramid_width,
        flow_combine_scale=flow_combine_scale,
        inference_flow_steps=model_flow_steps,
    )
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model, state


def per_image_from_prob(
    prob: np.ndarray,
    gt: np.ndarray,
    *,
    threshold: float,
    min_area: int,
) -> Tuple[int, int, int, float, float, float, Dict[str, float]]:
    gt_present = int(gt.sum() > 0)
    max_prob = float(prob.max()) if prob.size else 0.0
    pred_bin = _binarize(prob, threshold)
    fg_pixels = int(pred_bin.sum())
    pred_fg_fraction = fg_pixels / float(prob.size + 1e-12)
    pred_present_area = int(fg_pixels >= min_area)
    insts = _mask_to_instances(pred_bin, min_area=min_area)
    score = float(np.max(_instance_scores(prob, insts))) if insts else 0.0
    pred_present_instance = int(len(insts) > 0)

    logits = torch.from_numpy(prob[None, None, ...].astype(np.float32))
    logits = torch.log(logits.clamp(1e-6, 1 - 1e-6) / (1 - logits.clamp(1e-6, 1 - 1e-6)))
    tgt = torch.from_numpy(gt[None, None, ...].astype(np.float32))
    pix = pixel_metrics_from_logits(logits, tgt, threshold=threshold)

    return (
        gt_present,
        pred_present_instance,
        pred_present_area,
        score,
        max_prob,
        pred_fg_fraction,
        pix,
    )


@torch.no_grad()
def run_split(
    model: TriEncoderCFMNet,
    loader: DataLoader,
    *,
    dataset_name: str,
    split_name: str,
    device: torch.device,
    threshold: float,
    min_area: int,
    infer_flow_steps: int,
) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    idx_global = 0
    for batch in tqdm(loader, desc=f"{dataset_name} val", leave=False):
        rg, dem, ctx, y = prep_batch_triplet(batch, device)
        out = model(
            rg,
            dem,
            ctx,
            gt_mask=None,
            inference_flow_steps=int(infer_flow_steps) if infer_flow_steps > 0 else None,
        )
        probs = torch.sigmoid(out["logits_aux"]).detach().cpu().numpy()
        gts = y.detach().cpu().numpy()
        if gts.ndim == 4:
            gts = gts[:, 0]
        if probs.ndim == 4:
            probs = probs[:, 0]

        b = probs.shape[0]
        for i in range(b):
            gt_present, pred_inst, pred_area, score, mx, fg_frac, pix = per_image_from_prob(
                probs[i],
                (gts[i] > 0).astype(np.uint8),
                threshold=threshold,
                min_area=min_area,
            )
            records.append(
                ImageRecord(
                    dataset=dataset_name,
                    split=split_name,
                    index=idx_global,
                    gt_present=gt_present,
                    pred_present_instance=pred_inst,
                    pred_present_area=pred_area,
                    image_score=score,
                    max_prob=mx,
                    pred_fg_fraction=fg_frac,
                    pixel_f1=float(pix["f1"]),
                    pixel_iou=float(pix["iou"]),
                    pixel_precision=float(pix["precision"]),
                    pixel_recall=float(pix["recall"]),
                )
            )
            idx_global += 1
    return records


def _safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else float("nan")


def aggregate_group(
    records: Sequence[ImageRecord],
    *,
    dataset: str,
    gt_label: str,
    gt_present_value: int,
    use_pred_key: str = "pred_present_instance",
) -> dict:
    image_cols = {
        "image_auroc": float("nan"),
        "image_auprc": float("nan"),
        "image_best_f1": float("nan"),
        "image_best_score_threshold": float("nan"),
    }
    subset = [r for r in records if r.gt_present == gt_present_value]
    n = len(subset)
    if n == 0:
        return {
            "dataset": dataset,
            "gt_class": gt_label,
            "n_images": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "sensitivity_or_specificity": float("nan"),
            "false_alarm_rate": float("nan"),
            "detection_rate": float("nan"),
            "mean_pred_fg_fraction": float("nan"),
            "mean_pixel_f1": float("nan"),
            "mean_pixel_iou": float("nan"),
            "mean_pixel_precision": float("nan"),
            "mean_pixel_recall": float("nan"),
            **image_cols,
        }

    preds = [getattr(r, use_pred_key) for r in subset]
    if gt_present_value == 1:
        tp = sum(preds)
        fn = n - tp
        fp = tn = 0
        sens = _safe_rate(tp, n)
        spec = float("nan")
        far = float("nan")
        det = sens
    else:
        fp = sum(preds)
        tn = n - fp
        tp = fn = 0
        spec = _safe_rate(tn, n)
        far = _safe_rate(fp, n)
        sens = float("nan")
        det = float("nan")

    return {
        "dataset": dataset,
        "gt_class": gt_label,
        "n_images": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity_or_specificity": sens if gt_present_value == 1 else spec,
        "false_alarm_rate": far,
        "detection_rate": det,
        "mean_pred_fg_fraction": float(np.mean([r.pred_fg_fraction for r in subset])),
        "mean_pixel_f1": float(np.mean([r.pixel_f1 for r in subset])),
        "mean_pixel_iou": float(np.mean([r.pixel_iou for r in subset])),
        "mean_pixel_precision": float(np.mean([r.pixel_precision for r in subset])),
        "mean_pixel_recall": float(np.mean([r.pixel_recall for r in subset])),
        **image_cols,
    }


def dataset_level_image_metrics(records: Sequence[ImageRecord]) -> dict:
    base = {
        "dataset": records[0].dataset if records else "",
        "gt_class": "all_images",
        "n_images": len(records),
        "tp": "",
        "fp": "",
        "tn": "",
        "fn": "",
        "sensitivity_or_specificity": float("nan"),
        "false_alarm_rate": float("nan"),
        "detection_rate": float("nan"),
        "mean_pred_fg_fraction": float("nan"),
        "mean_pixel_f1": float("nan"),
        "mean_pixel_iou": float("nan"),
        "mean_pixel_precision": float("nan"),
        "mean_pixel_recall": float("nan"),
        "image_auroc": float("nan"),
        "image_auprc": float("nan"),
        "image_best_f1": float("nan"),
        "image_best_score_threshold": float("nan"),
    }
    scores = np.asarray([r.image_score for r in records], dtype=np.float32)
    labels = np.asarray([r.gt_present for r in records], dtype=np.int32)
    n = len(records)
    if n == 0 or len(np.unique(labels)) < 2:
        return base
    prec, rec, thr = _pr_curve(scores, labels)
    fpr, tpr = _roc_curve(scores, labels)
    auprc = float(np.trapz(prec[np.argsort(rec)], rec[np.argsort(rec)]))
    auroc = float(np.trapz(tpr[np.argsort(fpr)], fpr[np.argsort(fpr)]))
    f1s = 2 * prec * rec / (prec + rec + 1e-6)
    best_idx = int(np.argmax(f1s))
    base.update(
        {
            "image_auroc": auroc,
            "image_auprc": auprc,
            "image_best_f1": float(f1s[best_idx]),
            "image_best_score_threshold": float(thr[best_idx]),
        }
    )
    return base


def write_image_csv(path: Path, records: Sequence[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "split",
        "index",
        "gt_present",
        "pred_present_instance",
        "pred_present_area",
        "image_score",
        "max_prob",
        "pred_fg_fraction",
        "pixel_f1",
        "pixel_iou",
        "pixel_precision",
        "pixel_recall",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: getattr(r, k) for k in fields})


def write_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    preferred = [
        "model_id",
        "best_epoch",
        "prob_threshold",
        "min_connected_area_px",
        "pred_rule",
        "dataset",
        "gt_class",
        "n_images",
        "tp",
        "fp",
        "tn",
        "fn",
        "detection_rate",
        "sensitivity_or_specificity",
        "false_alarm_rate",
        "mean_pred_fg_fraction",
        "mean_pixel_f1",
        "mean_pixel_iou",
        "mean_pixel_precision",
        "mean_pixel_recall",
        "image_auroc",
        "image_auprc",
        "image_best_f1",
        "image_best_score_threshold",
    ]
    all_keys = set().union(*(row.keys() for row in rows))
    fields = [k for k in preferred if k in all_keys]
    fields.extend(sorted(all_keys - set(fields)))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def plot_score_histograms(
    bijie_records: Sequence[ImageRecord],
    l4s_records: Sequence[ImageRecord],
    out_path: Path,
    *,
    threshold: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    def _panel(ax, records, title):
        pos = [r.image_score for r in records if r.gt_present == 1]
        neg = [r.image_score for r in records if r.gt_present == 0]
        bins = np.linspace(0.0, 1.0, 26)
        if pos:
            ax.hist(pos, bins=bins, alpha=0.72, color="#0072B2", label=f"GT landslide (n={len(pos)})")
        if neg:
            ax.hist(neg, bins=bins, alpha=0.72, color="#D55E00", label=f"GT non-landslide (n={len(neg)})")
        ax.axvline(threshold, color="#333333", linestyle="--", linewidth=1.0, label=f"prob thr={threshold}")
        ax.set_xlabel("Image-level score (max instance prob)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=8, frameon=False)
        ax.set_xlim(0.0, 1.0)

    _panel(axes[0], bijie_records, "Bijie validation split")
    _panel(axes[1], l4s_records, "Landslide4Sense validation split")
    fig.suptitle(
        "TriEncoderCFMNet: landslide presence vs absence (image-level score)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def build_combined_table_rows(
    bijie_records: List[ImageRecord],
    l4s_records: List[ImageRecord],
    *,
    model_id: str,
    bijie_epoch: int,
    l4s_epoch: int,
    bijie_threshold: float,
    l4s_threshold: float,
    min_area: int,
) -> List[dict]:
    """Paper-ready long table: both datasets, landslide / non-landslide / pooled image metrics."""
    shared = {
        "model_id": model_id,
        "min_connected_area_px": min_area,
        "pred_rule": "pred_present_instance",
    }
    rows: List[dict] = []
    for recs, ep, thr in (
        (bijie_records, bijie_epoch, bijie_threshold),
        (l4s_records, l4s_epoch, l4s_threshold),
    ):
        ds = recs[0].dataset if recs else "unknown"
        for block in (
            aggregate_group(recs, dataset=ds, gt_label="landslide", gt_present_value=1),
            aggregate_group(recs, dataset=ds, gt_label="non_landslide", gt_present_value=0),
            dataset_level_image_metrics(recs),
        ):
            rows.append({**shared, "best_epoch": ep, "prob_threshold": thr, **block})
    return rows


def build_run_manifest(
    *,
    model_id: str,
    bijie_epoch: int,
    l4s_epoch: int,
    bijie_ckpt: Path,
    l4s_ckpt: Path,
    bijie_threshold: float,
    l4s_threshold: float,
    bijie_summary: Path,
    l4s_summary: Path,
) -> List[dict]:
    return [
        {
            "model_id": model_id,
            "dataset": "bijie",
            "best_epoch_summary_csv": bijie_epoch,
            "summary_csv": str(bijie_summary),
            "checkpoint": str(bijie_ckpt),
            "prob_threshold": bijie_threshold,
        },
        {
            "model_id": model_id,
            "dataset": "landslide4sense",
            "best_epoch_summary_csv": l4s_epoch,
            "summary_csv": str(l4s_summary),
            "checkpoint": str(l4s_ckpt),
            "prob_threshold": l4s_threshold,
        },
    ]


def parse_args():
    p = argparse.ArgumentParser(description="Image-level landslide presence eval (TriEncoderCFMNet).")
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--bijie-root", type=str, required=True, help="Bijie-landslide-dataset root")
    p.add_argument("--l4s-root", type=str, required=True, help="Landslide4Sense root (TrainData/img)")
    p.add_argument(
        "--bijie-summary",
        type=Path,
        default=_CODE_ROOT / "resources/results/bijie_ablation_report/bijie_best_validation_summary.csv",
    )
    p.add_argument(
        "--l4s-summary",
        type=Path,
        default=_CODE_ROOT / "resources/results/l4s_ablation_report/landslide4sense_best_validation_summary.csv",
    )
    p.add_argument("--bijie-run-dir", type=Path, default=_CODE_ROOT / "runs/bijie/tri_encoder_cfm_v2")
    p.add_argument("--l4s-run-dir", type=Path, default=_CODE_ROOT / "runs/landslide4sense/tri_encoder_cfm_v2")
    p.add_argument("--bijie-checkpoint", type=Path, default=None)
    p.add_argument("--l4s-checkpoint", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=RESULTS_ROOT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-split-ratio-l4s", type=float, default=0.1)
    p.add_argument("--resize-to", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--pyramid-width", type=int, default=64)
    p.add_argument("--flow-combine-scale", type=float, default=0.5)
    p.add_argument("--model-flow-steps", type=int, default=0)
    p.add_argument("--val-infer-flow-steps", type=int, default=-1, help="-1: use checkpoint / model default")
    p.add_argument("--metric-threshold", type=float, default=-1.0, help="-1: read from checkpoint")
    p.add_argument("--min-area", type=int, default=20, help="Min connected-component area (matches training image metrics)")
    p.add_argument("--bijie-epoch", type=int, default=-1, help="Override epoch from summary CSV")
    p.add_argument("--l4s-epoch", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    bijie_epoch = args.bijie_epoch if args.bijie_epoch > 0 else read_summary_best_epoch(args.bijie_summary, args.model_id)
    l4s_epoch = args.l4s_epoch if args.l4s_epoch > 0 else read_summary_best_epoch(args.l4s_summary, args.model_id)

    bijie_ckpt = resolve_checkpoint(args.bijie_run_dir, bijie_epoch, args.bijie_checkpoint)
    l4s_ckpt = resolve_checkpoint(args.l4s_run_dir, l4s_epoch, args.l4s_checkpoint)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Bijie val ---
    _, val_bijie_raw, _ = build_bijie_split(args.bijie_root, seed=args.seed)
    val_bijie_ds = BijieTripleStreamDataset(val_bijie_raw, resize_to=args.resize_to, transform=None)
    bijie_loader = DataLoader(
        val_bijie_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    bijie_model, bijie_state = load_model_from_checkpoint(
        bijie_ckpt,
        ctx_ch=4,
        pyramid_width=args.pyramid_width,
        flow_combine_scale=args.flow_combine_scale,
        model_flow_steps=args.model_flow_steps,
        device=device,
    )
    threshold_b = args.metric_threshold if args.metric_threshold >= 0 else float(bijie_state.get("metric_threshold", 0.6))
    infer_b = args.val_infer_flow_steps if args.val_infer_flow_steps >= 0 else int(
        bijie_state.get("val_infer_flow_steps", args.model_flow_steps)
    )

    bijie_records = run_split(
        bijie_model,
        bijie_loader,
        dataset_name="bijie",
        split_name="val",
        device=device,
        threshold=threshold_b,
        min_area=args.min_area,
        infer_flow_steps=infer_b,
    )
    del bijie_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- L4S val ---
    _, val_l4s_ids = build_l4s_split(args.l4s_root, val_ratio=args.val_split_ratio_l4s, seed=args.seed)
    val_l4s_ds = L4STripleStreamDataset(args.l4s_root, ids=val_l4s_ids, resize_to=args.resize_to, transform=None)
    l4s_loader = DataLoader(
        val_l4s_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    l4s_model, l4s_state = load_model_from_checkpoint(
        l4s_ckpt,
        ctx_ch=6,
        pyramid_width=args.pyramid_width,
        flow_combine_scale=args.flow_combine_scale,
        model_flow_steps=args.model_flow_steps,
        device=device,
    )
    threshold_l = args.metric_threshold if args.metric_threshold >= 0 else float(l4s_state.get("metric_threshold", 0.6))
    infer_l = args.val_infer_flow_steps if args.val_infer_flow_steps >= 0 else int(
        l4s_state.get("val_infer_flow_steps", args.model_flow_steps)
    )

    l4s_records = run_split(
        l4s_model,
        l4s_loader,
        dataset_name="landslide4sense",
        split_name="val",
        device=device,
        threshold=threshold_l,
        min_area=args.min_area,
        infer_flow_steps=infer_l,
    )

    write_image_csv(out_dir / "tri_encoder_presence_images_bijie.csv", bijie_records)
    write_image_csv(out_dir / "tri_encoder_presence_images_l4s.csv", l4s_records)

    table_rows = build_combined_table_rows(
        bijie_records,
        l4s_records,
        model_id=args.model_id,
        bijie_epoch=bijie_epoch,
        l4s_epoch=l4s_epoch,
        bijie_threshold=threshold_b,
        l4s_threshold=threshold_l,
        min_area=args.min_area,
    )
    manifest_rows = build_run_manifest(
        model_id=args.model_id,
        bijie_epoch=bijie_epoch,
        l4s_epoch=l4s_epoch,
        bijie_ckpt=bijie_ckpt,
        l4s_ckpt=l4s_ckpt,
        bijie_threshold=threshold_b,
        l4s_threshold=threshold_l,
        bijie_summary=args.bijie_summary.resolve(),
        l4s_summary=args.l4s_summary.resolve(),
    )

    write_summary_csv(out_dir / "tri_encoder_presence_combined_table.csv", table_rows)
    write_summary_csv(out_dir / "tri_encoder_presence_run_manifest.csv", manifest_rows)
    plot_score_histograms(
        bijie_records,
        l4s_records,
        out_dir / "fig_tri_encoder_presence_score_histogram.png",
        threshold=threshold_b,
    )

    print(f"Wrote combined table: {out_dir / 'tri_encoder_presence_combined_table.csv'}")
    print(f"Wrote figure: {out_dir / 'fig_tri_encoder_presence_score_histogram.png'}")
    print(f"Bijie val images: {len(bijie_records)} (GT landslide: {sum(r.gt_present for r in bijie_records)})")
    print(f"L4S val images: {len(l4s_records)} (GT landslide: {sum(r.gt_present for r in l4s_records)})")
    print(f"Checkpoints: bijie={bijie_ckpt} (epoch {bijie_epoch}), l4s={l4s_ckpt} (epoch {l4s_epoch})")


if __name__ == "__main__":
    main()
