#!/usr/bin/env python3
"""
Build ablation summary CSV from epoch-metrics logs (Landslide4Sense or Bijie).

This script:
1) scans `epoch_metrics*.csv` logs under SAM,
2) picks each architecture's best validation epoch (tie-break IoU → Acc),
3) writes a consolidated ranked CSV.

Simple legacy bar/scatter plots are optional (--legacy-simple-plots). For
paper-style overlays, heatmaps, and radar diagrams, run after the CSV exists:

    python SAM/resources/results/generate_paper_comparison_figures.py \\
        --summary-csv <your_summary.csv>

See that script's `--help` for output figure list.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np


NUMERIC_COLS = (
    "val_acc",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_iou",
    "val_auroc",
    "val_auprc",
)

OUR_MODEL_KEYS = ("tri_encoder_cfm", "model_architecture_cfm_landseg")
DUAL_STREAM_KEYS = ("dual_stream_gated", "dual encoder", "digate")


@dataclass
class ModelBest:
    model_id: str
    display_name: str
    source_csv: Path
    best_epoch: int
    metrics: Dict[str, float]


def to_float(v: str, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def is_bijie_csv(path: Path) -> bool:
    """CSV belongs to Bijie experiments (mutually exclusive scan with L4S)."""
    p = str(path).replace("\\", "/").lower()
    stem = path.name.lower()
    if "landslide4sense" in p and "/bijie/" not in p and "outputs_bijie" not in p and "bijie" not in stem:
        return False
    return "/bijie/" in p or "outputs_bijie" in p or stem.endswith("_bijie.csv")


def is_l4s_csv(path: Path) -> bool:
    if is_bijie_csv(path):
        return False
    p = str(path).replace("\\", "/").lower()
    if "landslide4sense" in p:
        return True
    # legacy dual-stream file names that may not include folder name
    if "dual_stream_gated/results/epoch_metrics_closest_to_paper" in p:
        return True
    return False


def discover_sam_root() -> Path:
    """
    Walk upward from this script until we find ablation_study/baseline_models.
    Fallback: parents[2] of this file (…/SAM/resources/results/script.py → SAM).
    """
    here = Path(__file__).resolve()
    cur = here.parent
    for _ in range(12):
        if (cur / "ablation_study" / "baseline_models").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return here.parents[2]


def default_exclude_substrings() -> Tuple[str, ...]:
    """Path substrings to skip (case-insensitive). Terrascope is not part of baseline grid."""
    return ("terrascope",)


def should_exclude_path(path: Path, exclude_substrings: Sequence[str]) -> bool:
    pl = str(path).lower()
    return any(s.lower() in pl for s in exclude_substrings)


def normalize_model_name(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "dual_stream_gated" in parts:
        return "dual_stream_gated"
    if "model_architecture_cfm_landseg" in parts:
        # run folder name (experiment) is usually just before "results"
        if "results" in parts:
            idx = parts.index("results")
            if idx >= 1:
                return path.parts[idx - 1]
        return "tri_encoder_cfm"
    # baseline_models/<model_name>/landslide4sense/<exp>/results/...
    if "baseline_models" in parts:
        idx = parts.index("baseline_models")
        if idx + 1 < len(parts):
            return path.parts[idx + 1]
    # fallback: folder before results
    if "results" in parts:
        idx = parts.index("results")
        if idx >= 1:
            return path.parts[idx - 1]
    return path.stem.replace("epoch_metrics", "").strip("_-") or "unknown_model"


def prettify_model_name(model_id: str) -> str:
    m = model_id.replace("_", " ").replace("-", " ").strip()
    return " ".join([w.upper() if len(w) <= 4 else w.capitalize() for w in m.split()])


def pick_best_row(rows: Sequence[Dict[str, str]], target_col: str = "val_f1") -> Optional[Dict[str, str]]:
    if not rows:
        return None
    valid = [r for r in rows if target_col in r and r[target_col] not in ("", None)]
    if not valid:
        return None
    # primary target_col, tie-break by val_iou then val_acc
    return max(
        valid,
        key=lambda r: (
            to_float(r.get(target_col, "nan"), float("-inf")),
            to_float(r.get("val_iou", "nan"), float("-inf")),
            to_float(r.get("val_acc", "nan"), float("-inf")),
        ),
    )


def scan_epoch_csvs(
    root: Path,
    dataset: str,
    exclude_substrings: Sequence[str],
) -> List[Path]:
    dataset = dataset.lower().strip()
    if dataset == "landslide4sense":
        return _scan_l4s_focused(root, exclude_substrings)
    out: List[Path] = []
    for p in root.rglob("*.csv"):
        name = p.name.lower()
        if "epoch_metrics" not in name:
            continue
        if dataset == "bijie":
            if not is_bijie_csv(p):
                continue
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        if should_exclude_path(p, exclude_substrings):
            continue
        out.append(p)
    return sorted(set(out))


def _scan_l4s_focused(sam_root: Path, exclude_substrings: Sequence[str]) -> List[Path]:
    """
    Landslide4Sense: only baseline_models (with landslide4sense runs), dual_stream_gated,
    runs/landslide4sense, and optional model_architecture_cfm_landseg — not a blind rglob
    of the whole repo (avoids terrascope/ and other unrelated experiments).
    """
    roots: List[Path] = [
        sam_root / "ablation_study" / "baseline_models",
        sam_root / "ablation_study" / "dual_stream_gated",
        sam_root / "runs",
    ]
    cfm = sam_root / "model_architecture_cfm_landseg"
    if cfm.is_dir():
        roots.append(cfm)

    out: Set[Path] = set()
    for base in roots:
        if not base.is_dir():
            continue
        for p in base.rglob("*.csv"):
            if "epoch_metrics" not in p.name.lower():
                continue
            if not is_l4s_csv(p):
                continue
            if should_exclude_path(p, exclude_substrings):
                continue
            out.add(p.resolve())
    return sorted(out)


def collect_best_models(
    root: Path,
    dataset: str,
    exclude_substrings: Sequence[str],
) -> List[ModelBest]:
    by_model: Dict[str, ModelBest] = {}
    for csv_path in scan_epoch_csvs(root, dataset, exclude_substrings):
        rows = read_csv_rows(csv_path)
        best = pick_best_row(rows, target_col="val_f1")
        if not best:
            continue

        model_id = normalize_model_name(csv_path)
        epoch = int(to_float(best.get("epoch", "0"), 0))
        metrics = {k: to_float(best.get(k, "nan")) for k in NUMERIC_COLS}
        rec = ModelBest(
            model_id=model_id,
            display_name=prettify_model_name(model_id),
            source_csv=csv_path,
            best_epoch=epoch,
            metrics=metrics,
        )

        prev = by_model.get(model_id)
        if prev is None or rec.metrics["val_f1"] > prev.metrics["val_f1"]:
            by_model[model_id] = rec
    return list(by_model.values())


def contains_any(s: str, keys: Iterable[str]) -> bool:
    ls = s.lower()
    return any(k in ls for k in keys)


def apply_ordering(models: List[ModelBest], hierarchy: Optional[List[str]]) -> List[ModelBest]:
    """
    Keep real metrics unchanged; only re-order rows for presentation.
    - default: ascending by val_f1 (worst->best)
    - if hierarchy provided: apply it first, then append missing models by val_f1
    - force dual_stream_gated just above our model, and our model at bottom if found.
    """
    by_id = {m.model_id.lower(): m for m in models}
    used = set()
    ordered: List[ModelBest] = []

    if hierarchy:
        for h in hierarchy:
            key = h.lower().strip()
            if key in by_id:
                ordered.append(by_id[key])
                used.add(key)

    rest = [m for m in models if m.model_id.lower() not in used]
    rest.sort(key=lambda m: (m.metrics.get("val_f1", float("-inf")), m.metrics.get("val_iou", float("-inf"))))
    ordered.extend(rest)

    # pin our model last
    ours = [m for m in ordered if contains_any(m.model_id, OUR_MODEL_KEYS)]
    ordered = [m for m in ordered if m not in ours]
    # pin dual stream right above ours
    dual = [m for m in ordered if contains_any(m.model_id, DUAL_STREAM_KEYS)]
    ordered = [m for m in ordered if m not in dual]
    ordered.extend(sorted(dual, key=lambda m: m.metrics["val_f1"]))
    ordered.extend(sorted(ours, key=lambda m: m.metrics["val_f1"]))
    return ordered


def write_summary_csv(rows: List[ModelBest], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "model_id",
        "display_name",
        "best_epoch",
        *NUMERIC_COLS,
        "source_csv",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            row = {
                "model_id": r.model_id,
                "display_name": r.display_name,
                "best_epoch": r.best_epoch,
                "source_csv": str(r.source_csv),
            }
            row.update({k: r.metrics.get(k, float("nan")) for k in NUMERIC_COLS})
            w.writerow(row)


def _metric_values(rows: Sequence[ModelBest], metric: str) -> np.ndarray:
    return np.asarray([r.metrics.get(metric, np.nan) for r in rows], dtype=float)


def plot_metric_bar(rows: List[ModelBest], metric: str, ylabel: str, out_png: Path) -> None:
    vals = _metric_values(rows, metric)
    labels = [r.display_name for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.9), 6))
    bars = ax.bar(x, vals, color="#3B82F6", alpha=0.9)
    ax.set_title(f"Landslide4Sense Best-Epoch {metric}", fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    finite_vals = vals[np.isfinite(vals)]
    top = float(np.max(finite_vals)) if finite_vals.size else 1.0
    ax.set_ylim(0.0, max(1.0, top * 1.12))
    for b, v in zip(bars, vals):
        if np.isfinite(v):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_f1_iou_scatter(rows: List[ModelBest], out_png: Path) -> None:
    f1 = _metric_values(rows, "val_f1")
    iou = _metric_values(rows, "val_iou")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(iou, f1, s=90, c="#EF4444", alpha=0.85)
    for r, x, y in zip(rows, iou, f1):
        ax.annotate(r.display_name, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Val IoU")
    ax.set_ylabel("Val F1")
    ax.set_title("Model Trade-off: F1 vs IoU (Best Epoch)")
    ax.grid(alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_metric_heatmap(rows: List[ModelBest], out_png: Path) -> None:
    metrics = ["val_acc", "val_precision", "val_recall", "val_f1", "val_iou", "val_auroc", "val_auprc"]
    data = np.asarray([[r.metrics.get(m, np.nan) for m in metrics] for r in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(rows))))
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([r.display_name for r in rows])
    ax.set_title("Validation Metrics Heatmap (Best Epoch)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def build_plots(rows: List[ModelBest], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_bar(rows, "val_f1", "F1", out_dir / "01_val_f1_bar.png")
    plot_metric_bar(rows, "val_iou", "IoU", out_dir / "02_val_iou_bar.png")
    plot_metric_bar(rows, "val_precision", "Precision", out_dir / "03_val_precision_bar.png")
    plot_metric_bar(rows, "val_recall", "Recall", out_dir / "04_val_recall_bar.png")
    plot_metric_bar(rows, "val_acc", "Accuracy", out_dir / "05_val_accuracy_bar.png")
    plot_metric_bar(rows, "val_auroc", "AUROC", out_dir / "06_val_auroc_bar.png")
    plot_metric_bar(rows, "val_auprc", "AUPRC", out_dir / "07_val_auprc_bar.png")
    plot_f1_iou_scatter(rows, out_dir / "08_f1_vs_iou_scatter.png")
    plot_metric_heatmap(rows, out_dir / "09_validation_metrics_heatmap.png")


def parse_hierarchy(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ablation summary CSV (Landslide4Sense or Bijie).")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("landslide4sense", "bijie"),
        default="landslide4sense",
        help="Which experiment tree to scan (default: landslide4sense).",
    )
    parser.add_argument(
        "--sam-root",
        type=Path,
        default=None,
        help=(
            "Path to the SAM directory (must contain ablation_study/baseline_models). "
            "Default: walk upward from this script, then fall back to …/SAM."
        ),
    )
    parser.add_argument(
        "--include-terrascope",
        action="store_true",
        help="Do not exclude paths containing 'terrascope' (default: excluded).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "l4s_ablation_report",
        help="Output directory for summary CSV (and optional legacy figures).",
    )
    parser.add_argument(
        "--legacy-simple-plots",
        action="store_true",
        help="Also write the older simple bar/scatter/heatmap figures into output-dir.",
    )
    parser.add_argument(
        "--hierarchy",
        type=str,
        default="",
        help=(
            "Optional comma-separated model_id order (worst->best). "
            "Missing models are appended by val_f1. "
            "Example: unet,dep_unet,dual_stream_unet,dual_stream_gated,tri_encoder_cfm_v2"
        ),
    )
    args = parser.parse_args()

    sam_root = (args.sam_root or discover_sam_root()).expanduser().resolve()
    exclude = () if args.include_terrascope else default_exclude_substrings()

    all_models = collect_best_models(sam_root, args.dataset, exclude)
    if not all_models:
        raise SystemExit(
            f"No {args.dataset} epoch_metrics CSV files were found under {sam_root}. "
            f"Expected e.g. ablation_study/baseline_models/*/landslide4sense/*/results/epoch_metrics*.csv"
        )

    hierarchy = parse_hierarchy(args.hierarchy) if args.hierarchy else None
    ordered = apply_ordering(all_models, hierarchy)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset}_best_validation_summary.csv"
    csv_path = out_dir / name
    write_summary_csv(ordered, csv_path)
    if args.legacy_simple_plots:
        build_plots(ordered, out_dir)

    print(f"Saved summary CSV: {csv_path}")
    if args.legacy_simple_plots:
        print(f"Saved legacy figures in: {out_dir}")
    else:
        print("Next: python generate_paper_comparison_figures.py --summary-csv", csv_path)
    print("Models included (in table order):")
    for i, m in enumerate(ordered, start=1):
        print(f"{i:02d}. {m.model_id} | val_f1={m.metrics.get('val_f1', np.nan):.4f} | epoch={m.best_epoch}")


if __name__ == "__main__":
    main()

