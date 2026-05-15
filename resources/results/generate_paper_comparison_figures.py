#!/usr/bin/env python3
"""
Publication-oriented comparison figures from an ablation summary CSV.

Expected input: CSV written by build_l4s_ablation_report.py with columns
model_id, display_name, best_epoch, val_acc, val_precision, val_recall,
val_f1, val_iou (no source_csv in the file). Epoch log paths are resolved at
run time via the same scan as the report builder.

Typical workflow:
  python build_l4s_ablation_report.py --dataset landslide4sense --output-dir ./out
  python generate_paper_comparison_figures.py --summary-csv ./out/landslide4sense_best_validation_summary.csv

Outputs two families of PNGs in the same output directory:

**Generic comparison set** (fig01–fig09): overlays, grouped bars, heatmap,
radar, scatter (see --help).

**Conference paper figures** (same numbering as the Landslide4Sense PDF bundle) are written to
`<output-dir>/<conference-subdir>/` as `Fig02_*.png` … `Fig13_*.png`
with captions taken from that PDF. Fig. 1 is architecture-only (see
`Fig01_model_architecture_NOTE.txt` in that folder).

  - Default `conference-subdir`: `conference_remotesensing_landslide` for L4S summaries,
    `conference_bijie` when the summary CSV filename contains `bijie` (override with `--conference-subdir`).

  - Fig. 1 — architecture (not from CSV)
  - Fig. 2 / 5 — training / validation dynamics for the focus model (wide horizontal layouts;
    Fig. 5 overlays the former Fig. 7 train–vs–validation bar summary on the validation heatmap).
  - Fig. 3 / 6 — train–validation curve panels (focus model)
  - Fig. 4 — final performance summary (all models in summary CSV, best epoch)
  - Fig. 7–9 / 10–12 — paper shows ROC/PR *curves* at epochs 24, 34, 39; logs
    only provide scalar val_auroc / val_auprc per epoch — plots mark those epochs
    (export threshold sweeps for true curves).
  - Fig. 13 — precision vs recall **grouped bar chart** for every model in the
    summary CSV (paper-style caption; publication theme).

Use --no-conference to skip the conference figure bundle (Fig.1 note + Fig.2–13).
Use --focus-model-id and --paper-epochs to match your runs.
"""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_RESULTS_DIR = Path(__file__).resolve().parent
if str(_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(_RESULTS_DIR))

from build_l4s_ablation_report import (
    collect_best_models,
    default_exclude_substrings,
    discover_sam_root,
)

import matplotlib.pyplot as plt
import numpy as np

# Colorblind-friendly (approx. Okabe–Ito order)
FIG_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
    "#56B4E9",
    "#E69F00",
    "#000000",
)

# Summary CSV omits val_auroc / val_auprc; heatmap uses whichever val_* columns are present.
HEATMAP_METRICS = (
    ("val_acc", "Acc"),
    ("val_precision", "Prec"),
    ("val_recall", "Rec"),
    ("val_f1", "F1"),
    ("val_iou", "IoU"),
)


def to_float(v: str, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _infer_dataset_scan_key(summary_csv: Path) -> str:
    return "bijie" if "bijie" in summary_csv.name.lower() else "landslide4sense"


def _source_csv_by_model_id(summary_csv: Path) -> Dict[str, Path]:
    root = discover_sam_root()
    dataset = _infer_dataset_scan_key(summary_csv)
    models = collect_best_models(root, dataset, default_exclude_substrings())
    out: Dict[str, Path] = {}
    for m in models:
        out[m.model_id] = m.source_csv
        out[m.model_id.lower()] = m.source_csv
    return out


def _attach_source_csv(rows: List[Dict[str, str]], summary_path: Path) -> None:
    """Ensure every row has a resolvable source_csv (epoch log) for plotting."""
    by_id = _source_csv_by_model_id(summary_path)
    for row in rows:
        p = (row.get("source_csv") or "").strip()
        if p:
            exp = Path(p).expanduser()
            if exp.is_file():
                row["source_csv"] = str(exp.resolve())
                continue
        mid = (row.get("model_id") or "").strip()
        src = by_id.get(mid) or by_id.get(mid.lower())
        if src is None:
            raise SystemExit(
                f"Cannot resolve epoch log for model_id={mid!r} from {summary_path}. "
                "Re-run build_l4s_ablation_report.py for this dataset or fix model_id spelling."
            )
        row["source_csv"] = str(src.resolve())


def read_summary(path: Path) -> List[Dict[str, str]]:
    rows = read_csv_rows(path)
    if not rows:
        raise SystemExit(f"Empty summary CSV: {path}")
    need = {"model_id", "display_name", "val_f1"}
    if not need.issubset(rows[0].keys()):
        raise SystemExit(f"Summary CSV missing columns {need}; got {list(rows[0].keys())}")
    _attach_source_csv(rows, path)
    return rows


def last_row_per_epoch(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Keep one row per epoch index (last occurrence = final log for that epoch)."""
    by_ep: Dict[int, Dict[str, str]] = {}
    for r in rows:
        if "epoch" not in r or r["epoch"] in ("", None):
            continue
        ep = int(to_float(r["epoch"], -1))
        if ep < 0:
            continue
        by_ep[ep] = r
    return [by_ep[k] for k in sorted(by_ep)]


# Columns that must never be treated as plottable metric series (e.g. Bijie / WandB exports).
_SKIP_SERIES_COLUMNS = frozenset(
    {
        "test_loss",
        "step",
        "epoch",
        "train_loss",
        "val_loss",
        "epoch_time",
        "val_pixel_f1_micro_thresh_sweep_best",
        "pixel_iou_val_micro_thresh_sweep_best",
        "pixel_val_best_threshold_sweep",
        "f1 score (non-landslide)",
        "iou (non-landslide)",
        "f1 score (landslide)",
    }
)


def _skip_series_column(name: str) -> bool:
    n = name.strip().lower()
    return n in _SKIP_SERIES_COLUMNS


def load_series(source_csv: Path) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    if not source_csv.is_file():
        return None
    raw = read_csv_rows(source_csv)
    rows = last_row_per_epoch(raw)
    if not rows:
        return None
    epoch = np.asarray([to_float(r["epoch"], np.nan) for r in rows], dtype=float)
    keys = [
        "train_loss",
        "val_loss",
        "train_f1",
        "val_f1",
        "train_iou",
        "val_iou",
    ]
    series: Dict[str, np.ndarray] = {}
    for k in keys:
        if k not in rows[0]:
            continue
        series[k] = np.asarray([to_float(r.get(k, "nan")) for r in rows], dtype=float)
    return epoch, series


def load_series_all_keys(source_csv: Path) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Like load_series but keeps every numeric column present in the epoch log."""
    if not source_csv.is_file():
        return None
    raw = read_csv_rows(source_csv)
    rows = last_row_per_epoch(raw)
    if not rows:
        return None
    epoch = np.asarray([to_float(r["epoch"], np.nan) for r in rows], dtype=float)
    skip_substrings = ("image_best", "threshold")
    series: Dict[str, np.ndarray] = {}
    for col in rows[0].keys():
        if _skip_series_column(col):
            continue
        if any(s in col for s in skip_substrings):
            continue
        try:
            arr = np.asarray([to_float(r.get(col, "nan")) for r in rows], dtype=float)
        except Exception:
            continue
        if np.any(np.isfinite(arr)):
            series[col] = arr
    return epoch, series


def column_normalize_01(mat: np.ndarray) -> np.ndarray:
    """Per-column min–max to [0,1] for heatmap display (paper-style dynamics)."""
    out = np.asarray(mat, dtype=float).copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        m = np.nanmin(col)
        M = np.nanmax(col)
        span = M - m
        if not np.isfinite(span) or span <= 1e-12:
            out[:, j] = 0.5
        else:
            out[:, j] = (col - m) / span
    return np.ma.masked_where(~np.isfinite(out), out)


def pick_focus_row(summary: List[Dict[str, str]], focus_id: Optional[str]) -> Dict[str, str]:
    if not summary:
        raise SystemExit("Empty summary.")
    if not focus_id:
        return summary[0]
    fid = focus_id.lower().strip()
    for row in summary:
        mid = (row.get("model_id") or "").lower()
        if fid in mid or mid in fid:
            return row
        dn = (row.get("display_name") or "").lower()
        if fid in dn:
            return row
    raise SystemExit(f"--focus-model-id {focus_id!r} did not match any model_id/display_name.")


def write_fig01_architecture_note(out_path: Path) -> None:
    text = f"""Fig. 1 — Model architecture design

This figure is not produced from epoch_metrics or the summary CSV. Export your
architecture diagram from your drawing tool or reuse assets under
SAM/codebase/model_architecture/ and insert in the manuscript.

Output bundle folder for this run: {out_path.parent.name}/
"""
    out_path.write_text(text, encoding="utf-8")


def _caption_lines(main: str, subtitle: str = "") -> str:
    return f"{main}\n{subtitle}" if subtitle.strip() else main


def conf_fig02_training_heatmap(focus_csv: Path, out_path: Path, caption: str, subtitle: str) -> None:
    loaded = load_series_all_keys(focus_csv)
    if loaded is None:
        return
    ep, ser = loaded
    cols = [c for c in sorted(ser) if c.startswith("train_")]
    if not cols:
        return
    mat = np.column_stack([ser[c] for c in cols])
    mat_n = column_normalize_01(mat)
    nice = [c.replace("train_", "").replace("_", " ").title() for c in cols]
    fig_h = max(3.6, len(ep) * 0.055)
    fig, ax = plt.subplots(figsize=(15.2, fig_h))
    im = ax.imshow(mat_n, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(nice)))
    ax.set_xticklabels(nice, rotation=35, ha="right", fontsize=8)
    _yt = np.linspace(0, len(ep) - 1, num=min(12, len(ep))).astype(int)
    ax.set_yticks(_yt)
    ax.set_yticklabels([str(int(ep[i])) for i in _yt], fontsize=8)
    ax.set_ylabel("Epoch")
    ax.set_title(_caption_lines(caption, subtitle))
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Norm. scale (per column)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def conf_fig05_validation_heatmap(focus_csv: Path, out_path: Path, caption: str, subtitle: str) -> None:
    loaded = load_series_all_keys(focus_csv)
    if loaded is None:
        return
    ep, ser = loaded
    cols = [c for c in sorted(ser) if c.startswith("val_")]
    if not cols:
        return
    mat = np.column_stack([ser[c] for c in cols])
    mat_n = column_normalize_01(mat)
    nice = [c.replace("val_", "").replace("_", " ").title() for c in cols]
    fig_h = max(3.6, len(ep) * 0.055)
    fig, ax = plt.subplots(figsize=(15.2, fig_h))
    im = ax.imshow(mat_n, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(nice)))
    ax.set_xticklabels(nice, rotation=35, ha="right", fontsize=8)
    _yt = np.linspace(0, len(ep) - 1, num=min(12, len(ep))).astype(int)
    ax.set_yticks(_yt)
    ax.set_yticklabels([str(int(ep[i])) for i in _yt], fontsize=8)
    ax.set_ylabel("Epoch")
    ax.set_title(_caption_lines(caption, subtitle))
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Norm. scale (per column)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def conf_fig03_train_val_panel(focus_csv: Path, out_path: Path, caption: str, subtitle: str) -> None:
    """Fig. 3 style: train vs validation curves (loss if logged, else acc / F1 / IoU only)."""
    raw = read_csv_rows(focus_csv)
    rows = last_row_per_epoch(raw)
    if not rows:
        return
    ep = np.asarray([to_float(r["epoch"], np.nan) for r in rows], dtype=float)
    loaded = load_series_all_keys(focus_csv)
    if loaded is None:
        return
    _, ser = loaded
    ser_plot: Dict[str, np.ndarray] = dict(ser)
    if "train_loss" in rows[0] and "val_loss" in rows[0]:
        tl = np.asarray([to_float(r.get("train_loss", "nan")) for r in rows], dtype=float)
        vl = np.asarray([to_float(r.get("val_loss", "nan")) for r in rows], dtype=float)
        if np.any(np.isfinite(tl)) or np.any(np.isfinite(vl)):
            ser_plot["train_loss"] = tl
            ser_plot["val_loss"] = vl
    pairs: List[Tuple[str, str, str]] = []
    if "train_loss" in ser_plot and "val_loss" in ser_plot:
        if np.any(np.isfinite(ser_plot["train_loss"])) or np.any(np.isfinite(ser_plot["val_loss"])):
            pairs.append(("train_loss", "val_loss", "Loss"))
    pairs.extend(
        [
            ("train_acc", "val_acc", "Accuracy"),
            ("train_f1", "val_f1", "F1"),
            ("train_iou", "val_iou", "IoU"),
        ]
    )
    n = len(pairs)
    if n == 4:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex=True)
        axes_flat = axes.ravel()
    elif n == 3:
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), sharex=True)
        axes_flat = axes
    else:
        fig, axes = plt.subplots(1, max(1, n), figsize=(max(8.0, 4.2 * n), 4.4), sharex=True)
        axes_flat = np.atleast_1d(axes)
    for ax, (tk, vk, title) in zip(axes_flat, pairs):
        if tk in ser_plot:
            ax.plot(ep, ser_plot[tk], label="Train", color="#009E73", linewidth=1.5)
        if vk in ser_plot:
            ax.plot(ep, ser_plot[vk], label="Val", color="#D55E00", linewidth=1.5)
        ax.set_title(title)
        _style_axes(ax)
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(_caption_lines(caption, subtitle), fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def conf_fig06_prec_rec_auroc_panel(focus_csv: Path, out_path: Path, caption: str, subtitle: str) -> None:
    """Fig. 6 style: precision / recall / AUOC dynamics (train vs val where available)."""
    loaded = load_series_all_keys(focus_csv)
    if loaded is None:
        return
    ep, ser = loaded
    triples = [
        ("train_precision", "val_precision", "Precision"),
        ("train_recall", "val_recall", "Recall"),
        ("train_auroc", "val_auroc", "AUROC"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
    for ax, (tk, vk, tlab) in zip(axes, triples):
        if tk in ser:
            ax.plot(ep, ser[tk], label="Train", color="#0072B2", linewidth=1.4)
        if vk in ser:
            ax.plot(ep, ser[vk], label="Val", color="#E69F00", linewidth=1.4)
        ax.set_ylabel(tlab)
        ax.set_title(tlab)
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("Epoch")
    fig.suptitle(_caption_lines(caption, subtitle), fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def conf_fig04_final_summary_all_models(
    summary: List[Dict[str, str]], out_path: Path, caption: str, subtitle: str
) -> None:
    fig_grouped_metrics(summary, out_path, caption, subtitle)


def _publication_style_context():
    """Prefer seaborn-like whitegrid if available in this Matplotlib build."""
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            return plt.style.context(name)
        except OSError:
            continue
    return nullcontext()


# Fig. 7 — slate / teal / gold (distinct from default Matplotlib colors)
FIG07_COL_PRECISION = "#3d4f5f"
FIG07_COL_RECALL = "#2a9d8f"
FIG07_COL_ACCURACY = "#e9c46a"

# Fig. 14 — muted navy + terracotta (publication contrast, not default teal/red)
FIG14_COL_PRECISION = "#2E5266"
FIG14_COL_RECALL = "#C17767"


def conf_fig07_final_bars_train_val(
    focus: Dict[str, str],
    focus_csv: Path,
    out_path: Path,
    caption: str,
    subtitle: str,
) -> None:
    """
    Fig. 7 paper layout: grouped bars for Training vs Validation at best epoch
    (precision, recall, accuracy).
    """
    best_ep = int(float(focus.get("best_epoch", "-1")))
    if best_ep < 0:
        return
    rows = last_row_per_epoch(read_csv_rows(focus_csv))
    row = None
    for r in rows:
        if int(to_float(r.get("epoch", "-1"), -1)) == best_ep:
            row = r
            break
    if row is None:
        return

    def g(key: str) -> float:
        return to_float(row.get(key, "nan"))

    tp, vp = g("train_precision"), g("val_precision")
    tr, vr = g("train_recall"), g("val_recall")
    ta, va = g("train_acc"), g("val_acc")
    if not all(np.isfinite(v) for v in (tp, vp, tr, vr, ta, va)):
        return

    x = np.arange(2, dtype=float)
    w = 0.24
    with _publication_style_context():
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        fig.patch.set_facecolor("#fdfdfd")
        ax.set_facecolor("#f4f5f7")

        b0 = ax.bar(
            x - w,
            [tp, vp],
            w,
            label="Precision",
            color=FIG07_COL_PRECISION,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        b1 = ax.bar(
            x,
            [tr, vr],
            w,
            label="Recall",
            color=FIG07_COL_RECALL,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        b2 = ax.bar(
            x + w,
            [ta, va],
            w,
            label="Accuracy",
            color=FIG07_COL_ACCURACY,
            edgecolor="#5c4d2a",
            linewidth=0.35,
            zorder=3,
        )

        for bars in (b0, b1, b2):
            for b in bars:
                h = b.get_height()
                if np.isfinite(h):
                    ax.text(
                        b.get_x() + b.get_width() / 2.0,
                        h + 0.02,
                        f"{h:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="#2b2b2b",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(["Training", "Validation"], fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0.0, 1.12)
        ax.set_title("Final Model Performance Summary", fontsize=12, fontweight="600", pad=14)
        ax.legend(loc="lower center", ncol=3, frameon=True, fancybox=False, edgecolor="#cccccc")
        ax.yaxis.grid(True, linestyle="-", alpha=0.45)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        fig.text(0.5, 0.01, _caption_lines(caption, subtitle), ha="center", fontsize=9, color="#333333")

        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


def conf_fig05_validation_heatmap_with_train_val_bars_horizontal(
    focus: Dict[str, str],
    focus_csv: Path,
    out_path: Path,
    caption: str,
    subtitle: str,
) -> None:
    """
    Wide Fig. 5: validation metric heatmap (left) + training vs validation bars
    for precision, recall, and accuracy at best epoch (right; former Fig. 7 layout).
    """
    loaded = load_series_all_keys(focus_csv)
    if loaded is None:
        return
    ep, ser = loaded
    cols = [c for c in sorted(ser) if c.startswith("val_")]
    if not cols:
        return
    mat = np.column_stack([ser[c] for c in cols])
    mat_n = column_normalize_01(mat)
    nice = [c.replace("val_", "").replace("_", " ").title() for c in cols]

    best_ep = int(float(focus.get("best_epoch", "-1")))
    rows_lv = last_row_per_epoch(read_csv_rows(focus_csv))
    row: Dict[str, str] | None = None
    for r in rows_lv:
        if int(to_float(r.get("epoch", "-1"), -1)) == best_ep:
            row = r
            break
    if row is None and rows_lv:
        row = rows_lv[-1]
    if row is None:
        return

    def g(key: str) -> float:
        return to_float(row.get(key, "nan"))

    tp, vp = g("train_precision"), g("val_precision")
    tr, vr = g("train_recall"), g("val_recall")
    ta, va = g("train_acc"), g("val_acc")
    if not all(np.isfinite(v) for v in (tp, vp, tr, vr, ta, va)):
        return

    h_heat = max(4.0, len(ep) * 0.052)
    with _publication_style_context():
        fig = plt.figure(figsize=(17.2, h_heat))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.72, 1.0], wspace=0.28)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        fig.patch.set_facecolor("#fdfdfd")

        im = ax0.imshow(mat_n, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax0.set_xticks(np.arange(len(nice)))
        ax0.set_xticklabels(nice, rotation=28, ha="right", fontsize=7.5)
        _yt = np.linspace(0, len(ep) - 1, num=min(12, len(ep))).astype(int)
        ax0.set_yticks(_yt)
        ax0.set_yticklabels([str(int(ep[i])) for i in _yt], fontsize=8)
        ax0.set_ylabel("Epoch")
        ax0.set_title("Validation metrics (heatmap)", fontsize=10.5, fontweight="600")
        fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02, label="Norm. scale (per column)")

        ax1.set_facecolor("#f4f5f7")
        xb = np.arange(2, dtype=float)
        w = 0.24
        b0 = ax1.bar(
            xb - w,
            [tp, vp],
            w,
            label="Precision",
            color=FIG07_COL_PRECISION,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        b1 = ax1.bar(
            xb,
            [tr, vr],
            w,
            label="Recall",
            color=FIG07_COL_RECALL,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        b2 = ax1.bar(
            xb + w,
            [ta, va],
            w,
            label="Accuracy",
            color=FIG07_COL_ACCURACY,
            edgecolor="#5c4d2a",
            linewidth=0.35,
            zorder=3,
        )
        for bars in (b0, b1, b2):
            for b in bars:
                h = b.get_height()
                if np.isfinite(h):
                    ax1.text(
                        b.get_x() + b.get_width() / 2.0,
                        h + 0.02,
                        f"{h:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        color="#2b2b2b",
                    )
        ax1.set_xticks(xb)
        ax1.set_xticklabels(["Training", "Validation"], fontsize=10)
        ax1.set_ylabel("Score", fontsize=10)
        ax1.set_ylim(0.0, 1.12)
        ax1.set_title(
            f"Train vs val at best epoch ({best_ep})",
            fontsize=10.5,
            fontweight="600",
            pad=10,
        )
        ax1.legend(loc="lower center", ncol=3, frameon=True, fancybox=False, edgecolor="#cccccc", fontsize=7.5)
        ax1.yaxis.grid(True, linestyle="-", alpha=0.45)
        ax1.set_axisbelow(True)
        for spine in ("top", "right"):
            ax1.spines[spine].set_visible(False)

        fig.suptitle(_caption_lines(caption, subtitle), fontsize=11.5, fontweight="600", y=0.98)
        fig.tight_layout(rect=(0, 0.02, 1, 0.94))
        fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


def conf_scalar_epoch_plot(
    focus_csv: Path,
    out_path: Path,
    caption: str,
    subtitle: str,
    y_col: str,
    y_label: str,
    mark_epoch: int,
) -> None:
    """Scalar vs epoch with vertical marker (stand-in when full ROC/PR curves are unavailable)."""
    rows = last_row_per_epoch(read_csv_rows(focus_csv))
    if not rows:
        return
    ep = np.asarray([to_float(r.get("epoch", "nan")) for r in rows], dtype=float)
    y = np.asarray([to_float(r.get(y_col, "nan")) for r in rows], dtype=float)
    if not np.any(np.isfinite(y)):
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(ep, y, color="#0072B2", linewidth=1.6, label=y_label)
    if mark_epoch in [int(to_float(r.get("epoch", "-1"), -1)) for r in rows]:
        sub = next(
            (
                to_float(r.get(y_col, "nan"))
                for r in rows
                if int(to_float(r.get("epoch", "-1"), -1)) == mark_epoch
            ),
            float("nan"),
        )
        ax.axvline(mark_epoch, color="#CC79A7", linestyle="--", linewidth=1.2, alpha=0.9)
        if np.isfinite(sub):
            ax.scatter([float(mark_epoch)], [sub], s=80, c="#D55E00", zorder=5, edgecolors="white")
            ax.annotate(
                f"ep {mark_epoch}\n{sub:.4f}",
                xy=(float(mark_epoch), sub),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.set_title(_caption_lines(caption, subtitle))
    ax.text(
        0.02,
        0.02,
        "Note: epoch logs store scalar AUROC/AUPRC only.\n"
        "For full ROC/PR curves, export threshold sweeps during eval.",
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="bottom",
        color="#444444",
    )
    _style_axes(ax)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def conf_fig14_precision_recall_grouped_bars(
    summary: List[Dict[str, str]], out_path: Path, caption: str, subtitle: str
) -> None:
    """
    Fig. 14 — grouped bars: precision and recall per model (summary CSV),
    publication layout (matches paper-style comparison chart).
    """
    labels = [r.get("display_name") or r.get("model_id", "?") for r in summary]
    prec = np.asarray([to_float(r.get("val_precision", "nan")) for r in summary], dtype=float)
    rec = np.asarray([to_float(r.get("val_recall", "nan")) for r in summary], dtype=float)
    n = len(summary)
    if n == 0:
        return

    x = np.arange(n, dtype=float)
    w = float(min(0.36, 0.72 / max(n, 1)))

    with _publication_style_context():
        fig, ax = plt.subplots(figsize=(max(9.5, n * 0.78), 6.0))
        fig.patch.set_facecolor("#fdfdfd")
        ax.set_facecolor("#f4f5f7")

        ax.bar(
            x - w / 2,
            prec,
            w,
            label="Precision",
            color=FIG14_COL_PRECISION,
            edgecolor="white",
            linewidth=0.95,
            zorder=3,
        )
        ax.bar(
            x + w / 2,
            rec,
            w,
            label="Recall",
            color=FIG14_COL_RECALL,
            edgecolor="white",
            linewidth=0.95,
            zorder=3,
        )

        for i in range(n):
            if np.isfinite(prec[i]):
                ax.text(
                    x[i] - w / 2,
                    min(prec[i] + 0.025, 1.04),
                    f"{prec[i]:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#222222",
                )
            if np.isfinite(rec[i]):
                ax.text(
                    x[i] + w / 2,
                    min(rec[i] + 0.025, 1.04),
                    f"{rec[i]:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#222222",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=9)
        ax.set_xlabel("Models", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_ylim(0.0, 1.08)
        ax.set_title(
            "Model Performance: Precision vs Recall",
            fontsize=12,
            fontweight="600",
            pad=14,
        )
        ax.legend(
            loc="upper center",
            ncol=2,
            frameon=True,
            fancybox=False,
            edgecolor="#cccccc",
            bbox_to_anchor=(0.5, 1.06),
        )
        ax.yaxis.grid(True, linestyle="-", alpha=0.42)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        fig.text(
            0.5,
            0.03,
            caption,
            ha="center",
            fontsize=10.5,
            fontfamily="serif",
            style="italic",
            color="#1a1a1a",
        )
        if subtitle.strip():
            fig.text(0.5, 0.01, subtitle, ha="center", fontsize=8.5, color="#555555")

        fig.tight_layout(rect=(0, 0.1, 1, 1.08))
        fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


# Default subfolder for PDF-aligned figure bundle (Landslide4Sense). Bijie uses
# `conference_bijie` unless overridden by --conference-subdir.
DEFAULT_CONFERENCE_PAPER_SUBDIR = "conference_remotesensing_landslide"
BIJIE_CONFERENCE_PAPER_SUBDIR = "conference_bijie"


def resolve_conference_paper_subdir(summary_csv: Path, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    if "bijie" in summary_csv.name.lower():
        return BIJIE_CONFERENCE_PAPER_SUBDIR
    return DEFAULT_CONFERENCE_PAPER_SUBDIR


def generate_conference_figures(
    summary_ordered: List[Dict[str, str]],
    out_dir: Path,
    label: str,
    focus: Dict[str, str],
    paper_epochs: List[int],
    conference_paper_subdir: str,
) -> None:
    """
    Figures 1–13 bundle (same filenames as the Landslide4Sense conference set).
    Outputs go under ``out_dir / conference_paper_subdir /``.
    """
    paper_dir = out_dir / conference_paper_subdir
    paper_dir.mkdir(parents=True, exist_ok=True)
    write_fig01_architecture_note(paper_dir / "Fig01_model_architecture_NOTE.txt")

    fcsv = Path(focus["source_csv"]).expanduser()
    dname = focus.get("display_name") or focus.get("model_id", "focus")
    sub_common = f"{label} · summary CSV · focus run: {dname}"

    # PDF captions (verbatim where noted)
    conf_fig02_training_heatmap(
        fcsv,
        paper_dir / "Fig02_performance_heatmap_segmentation_model_training.png",
        "Fig. 2. Performance heatmap for segmentation model",
        sub_common,
    )
    conf_fig03_train_val_panel(
        fcsv,
        paper_dir / "Fig03_train_validation_comparison_segmentation_model.png",
        "Fig. 3. Train validation comparison for segmentation model",
        sub_common,
    )
    conf_fig04_final_summary_all_models(
        summary_ordered,
        paper_dir / "Fig04_final_performance_summary_segmentation_model.png",
        "Fig. 4. Final performance summary for segmentation model",
        f"All models in summary table (best validation epoch). {label}.",
    )
    conf_fig05_validation_heatmap_with_train_val_bars_horizontal(
        focus,
        fcsv,
        paper_dir / "Fig05_performance_heatmap_segmentation_model_validation.png",
        "Fig. 5. Performance heatmap for segmentation model (validation) with train–validation summary bars",
        sub_common + " · wide layout: heatmap + precision/recall/accuracy at best epoch",
    )
    conf_fig06_prec_rec_auroc_panel(
        fcsv,
        paper_dir / "Fig06_training_validation_comparison_segmentation_model.png",
        "Fig. 6. Training validation comparison for segmentation model",
        sub_common,
    )

    pe = list(paper_epochs)
    while len(pe) < 3:
        pe.append(pe[-1] if pe else 24)
    roc_epochs = pe[:3]
    for i, e in enumerate(roc_epochs):
        fig_n = 7 + i
        conf_scalar_epoch_plot(
            fcsv,
            paper_dir / f"Fig{fig_n:02d}_ROC_curve_epoch_{e}.png",
            f"Fig. {fig_n}. ROC curve for epoch {e}",
            sub_common,
            "val_auroc",
            "Validation AUROC (scalar; full ROC needs threshold sweep)",
            e,
        )
    for i, e in enumerate(roc_epochs):
        fig_n = 10 + i
        conf_scalar_epoch_plot(
            fcsv,
            paper_dir / f"Fig{fig_n:02d}_PR_curve_epoch_{e}.png",
            f"Fig. {fig_n}. PR curve for epoch {e}",
            sub_common,
            "val_auprc",
            "Validation AUPRC (scalar; full PR curve needs threshold sweep)",
            e,
        )

    conf_fig14_precision_recall_grouped_bars(
        summary_ordered,
        paper_dir / "Fig13_precision_recall_comparison_all_segmentation_models.png",
        "Fig. 13. Precision-Recall comparision of all the segmentation models",
        f"Validation metrics at best epoch per model · {label}.",
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(alpha=0.28, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def setup_rc() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 320,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.linewidth": 0.9,
            "axes.prop_cycle": plt.cycler(color=list(FIG_COLORS)),
        }
    )


def fig_loss_curves(
    summary: List[Dict[str, str]], out_path: Path, dataset_label: str
) -> None:
    """Stacked train / validation loss vs epoch for every model."""
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.5), sharex=True)
    for i, row in enumerate(summary):
        lbl = row.get("display_name") or row.get("model_id", str(i))
        p = Path(row["source_csv"]).expanduser()
        loaded = load_series(p)
        if loaded is None:
            continue
        ep, ser = loaded
        c = FIG_COLORS[i % len(FIG_COLORS)]
        if "train_loss" in ser and np.any(np.isfinite(ser["train_loss"])):
            axes[0].plot(ep, ser["train_loss"], label=lbl, color=c, linewidth=1.4, alpha=0.9)
        if "val_loss" in ser and np.any(np.isfinite(ser["val_loss"])):
            axes[1].plot(ep, ser["val_loss"], label=lbl, color=c, linewidth=1.4, alpha=0.9)
    axes[0].set_ylabel("Train loss")
    axes[1].set_ylabel("Validation loss")
    axes[1].set_xlabel("Epoch")
    axes[0].set_title(f"{dataset_label}: optimization loss trajectories")
    for ax in axes:
        _style_axes(ax)
    h0, l0 = axes[0].get_legend_handles_labels()
    if h0:
        fig.legend(h0, l0, loc="lower center", ncol=min(4, len(h0)), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_scalar_curve(
    summary: List[Dict[str, str]],
    out_path: Path,
    dataset_label: str,
    col: str,
    ylabel: str,
    title_suffix: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    for i, row in enumerate(summary):
        lbl = row.get("display_name") or row.get("model_id", str(i))
        p = Path(row["source_csv"]).expanduser()
        loaded = load_series(p)
        if loaded is None:
            continue
        ep, ser = loaded
        if col not in ser or not np.any(np.isfinite(ser[col])):
            continue
        c = FIG_COLORS[i % len(FIG_COLORS)]
        ax.plot(ep, ser[col], label=lbl, color=c, linewidth=1.4, alpha=0.92)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{dataset_label}: {title_suffix}")
    _style_axes(ax)
    if ax.lines:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(handles)), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_grouped_metrics(
    summary: List[Dict[str, str]],
    out_path: Path,
    plot_title: str,
    subtitle: str = "",
) -> None:
    """Grouped bars: F1, IoU, precision, recall at best epoch (from summary)."""
    metrics = ("val_f1", "val_iou", "val_precision", "val_recall")
    metric_labels = ("F1", "IoU", "Prec.", "Rec.")
    n = len(summary)
    x = np.arange(n, dtype=float)
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(8.5, n * 0.85), 5.8))
    for j, mk in enumerate(metrics):
        offs = (j - (len(metrics) - 1) / 2) * width
        vals = np.asarray([to_float(row.get(mk, "nan")) for row in summary], dtype=float)
        ax.bar(x + offs, vals, width, label=metric_labels[j], color=FIG_COLORS[j % len(FIG_COLORS)], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([r.get("display_name", "") for r in summary], rotation=28, ha="right")
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(_caption_lines(plot_title, subtitle))
    ax.legend(ncol=4, frameon=False, loc="upper right")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_radar(
    summary: List[Dict[str, str]], out_path: Path, dataset_label: str, max_models: int
) -> None:
    """Radar / spider chart for key metrics (subset of models if many)."""
    sorted_rows = sorted(
        summary,
        key=lambda r: to_float(r.get("val_f1", "nan"), float("-inf")),
        reverse=True,
    )
    pick = sorted_rows[: max_models]
    cand_labels = ("F1", "IoU", "Prec.", "Rec.", "Acc.")
    cand_keys = ("val_f1", "val_iou", "val_precision", "val_recall", "val_acc")
    keys: List[str] = []
    labels: List[str] = []
    for k, lab in zip(cand_keys, cand_labels):
        if any(np.isfinite(to_float(r.get(k, "nan"))) for r in pick):
            keys.append(k)
            labels.append(lab)
    if len(keys) < 3:
        keys, labels = list(cand_keys[:-1]), list(cand_labels[:-1])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="polar")
    for i, row in enumerate(pick):
        vals = [to_float(row.get(k, "nan")) for k in keys]
        vals = [min(1.0, max(0.0, v)) if np.isfinite(v) else 0.0 for v in vals]
        vals_c = vals + vals[:1]
        ax.plot(angles, vals_c, "o-", linewidth=1.3, label=row.get("display_name", ""), color=FIG_COLORS[i % len(FIG_COLORS)])
        ax.fill(angles, vals_c, alpha=0.08, color=FIG_COLORS[i % len(FIG_COLORS)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{dataset_label}: metric profiles (top {len(pick)} by val F1)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.08), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(summary: List[Dict[str, str]], out_path: Path, dataset_label: str) -> None:
    keys = [
        k
        for k, _ in HEATMAP_METRICS
        if summary and any(np.isfinite(to_float(r.get(k, "nan"))) for r in summary)
    ]
    labels = [lab for k, lab in HEATMAP_METRICS if k in keys]
    if not keys:
        keys = ["val_f1", "val_iou", "val_precision", "val_recall"]
        labels = ["F1", "IoU", "Prec", "Rec"]
    mat = []
    for row in summary:
        mat.append([to_float(row.get(k, "nan")) for k in keys])
    data = np.asarray(mat, dtype=float)
    data = np.ma.masked_where(~np.isfinite(data), data)
    fig, ax = plt.subplots(figsize=(8.8, max(4.8, len(summary) * 0.34)))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(summary)))
    ax.set_yticklabels([r.get("display_name", "") for r in summary])
    ax.set_title(f"{dataset_label}: validation metrics matrix (best epoch)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.set_ylabel("Score")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_f1_iou_scatter(summary: List[Dict[str, str]], out_path: Path, dataset_label: str) -> None:
    f1 = np.asarray([to_float(r.get("val_f1", "nan")) for r in summary])
    iou = np.asarray([to_float(r.get("val_iou", "nan")) for r in summary])
    fig, ax = plt.subplots(figsize=(6.9, 6.5))
    for i, row in enumerate(summary):
        c = FIG_COLORS[i % len(FIG_COLORS)]
        xf, yi = iou[i], f1[i]
        if not (np.isfinite(xf) and np.isfinite(yi)):
            continue
        ax.scatter(xf, yi, s=70, color=c, edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate(
            row.get("display_name", ""),
            (xf, yi),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            clip_on=False,
        )
    ax.set_xlabel("Validation IoU")
    ax.set_ylabel("Validation F1")
    ax.set_title(f"{dataset_label}: F1 vs IoU trade-offs (best epoch)")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_ranked_f1(summary: List[Dict[str, str]], out_path: Path, dataset_label: str) -> None:
    order = sorted(
        enumerate(summary),
        key=lambda t: to_float(t[1].get("val_f1", "nan"), float("-inf")),
    )
    labels = [summary[i].get("display_name", "") for i, _ in order]
    vals = np.asarray([to_float(summary[i].get("val_f1", "nan")) for i, _ in order], dtype=float)
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, max(4.5, len(labels) * 0.35)))
    ax.barh(y, vals, color="#0072B2", alpha=0.85, height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Validation F1 (best epoch)")
    ax.set_xlim(0.0, 1.0)
    ax.set_title(f"{dataset_label}: models ranked by validation F1")
    _style_axes(ax)
    for yi, v in zip(y, vals):
        if np.isfinite(v):
            ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=8, color="#333333")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_train_val_gap(summary: List[Dict[str, str]], out_path: Path, dataset_label: str) -> None:
    """At summary best_epoch, compare train F1 vs val F1 (generalization)."""
    names: List[str] = []
    train_v: List[float] = []
    val_v: List[float] = []
    for row in summary:
        p = Path(row["source_csv"]).expanduser()
        best_ep = int(float(row.get("best_epoch", "-1")))
        if best_ep < 0 or not p.is_file():
            continue
        rows = last_row_per_epoch(read_csv_rows(p))
        picked = None
        for rr in rows:
            if int(to_float(rr.get("epoch", "-1"), -1)) == best_ep:
                picked = rr
        if picked is None:
            continue
        tr = to_float(picked.get("train_f1", "nan"))
        va = to_float(picked.get("val_f1", "nan"))
        if not (np.isfinite(tr) and np.isfinite(va)):
            continue
        names.append(row.get("display_name", ""))
        train_v.append(tr)
        val_v.append(va)
    if not names:
        return
    x = np.arange(len(names))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(8.5, len(names) * 0.82), 5.6))
    ax.bar(x - w / 2, train_v, w, label="Train F1", color="#009E73", alpha=0.88)
    ax.bar(x + w / 2, val_v, w, label="Val F1", color="#D55E00", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title(f"{dataset_label}: train vs validation F1 at selected best epoch")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def infer_dataset_label(path: Path, override: Optional[str]) -> str:
    if override:
        return override
    n = path.name.lower()
    if "bijie" in n:
        return "Bijie dataset"
    if "landslide" in n:
        return "Landslide4Sense"
    return "Validation benchmark"


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-style comparison figures from summary CSV.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
        help="Path to *_best_validation_summary.csv from build_l4s_ablation_report.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save PNGs (default: <summary_dir>/paper_comparison_figures/)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Subtitle prefix for plots (default: inferred from CSV filename).",
    )
    parser.add_argument(
        "--radar-max-models",
        type=int,
        default=8,
        help="Maximum models drawn on radar chart (best val F1 first).",
    )
    parser.add_argument(
        "--focus-model-id",
        type=str,
        default="",
        help="Substring of model_id/display_name for conf_fig temporal plots (default: best val_f1 row).",
    )
    parser.add_argument(
        "--paper-epochs",
        type=str,
        default="24,34,39",
        help="Comma epochs for conf_fig08–13 scalar markers (paper defaults).",
    )
    parser.add_argument(
        "--no-conference",
        action="store_true",
        help="Skip conference-template outputs (Fig.1 note + Fig.2–Fig.13 bundle).",
    )
    parser.add_argument(
        "--conference-subdir",
        type=str,
        default="",
        help=(
            "Subfolder under --output-dir for the PDF-style figure bundle. "
            "Default: conference_remotesensing_landslide for L4S summaries; "
            "conference_bijie when the summary CSV filename contains 'bijie'."
        ),
    )
    args = parser.parse_args()
    summary_path = args.summary_csv.expanduser().resolve()
    rows = read_summary(summary_path)

    order_idx = sorted(
        range(len(rows)),
        key=lambda idx: (-to_float(rows[idx].get("val_f1", "nan"), float("-inf")), idx),
    )
    summary_ordered = [rows[i] for i in order_idx]

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = summary_path.parent / "paper_comparison_figures"
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_rc()
    label = infer_dataset_label(summary_path, args.dataset_name or None)

    fig_loss_curves(summary_ordered, out_dir / "fig01_loss_curves_train_val.png", label)
    fig_scalar_curve(
        summary_ordered,
        out_dir / "fig02_validation_f1_vs_epoch.png",
        label,
        "val_f1",
        "F1 score",
        "validation F1 vs epoch",
    )
    fig_scalar_curve(
        summary_ordered,
        out_dir / "fig03_validation_iou_vs_epoch.png",
        label,
        "val_iou",
        "IoU",
        "validation IoU vs epoch",
    )
    fig_grouped_metrics(
        summary_ordered,
        out_dir / "fig04_grouped_metrics_best_epoch.png",
        f"{label}: validation metrics at best epoch (summary table)",
        "",
    )
    fig_radar(summary_ordered, out_dir / "fig05_radar_top_models.png", label, args.radar_max_models)
    fig_heatmap(summary_ordered, out_dir / "fig06_metrics_heatmap_best_epoch.png", label)
    fig_f1_iou_scatter(summary_ordered, out_dir / "fig07_f1_vs_iou_scatter.png", label)
    fig_ranked_f1(summary_ordered, out_dir / "fig08_ranked_f1_horizontal.png", label)
    fig_train_val_gap(summary_ordered, out_dir / "fig09_train_val_f1_gap_best_epoch.png", label)

    if not args.no_conference:
        focus = pick_focus_row(summary_ordered, args.focus_model_id or None)
        paper_eps = [int(x.strip()) for x in args.paper_epochs.split(",") if x.strip()]
        if not paper_eps:
            paper_eps = [24, 34, 39]
        conf_subdir = resolve_conference_paper_subdir(summary_path, args.conference_subdir)
        generate_conference_figures(
            summary_ordered, out_dir, label, focus, paper_eps, conf_subdir
        )
        print(
            f"PDF-aligned figures (Fig.1 note + Fig.2–Fig.13): {out_dir / conf_subdir}"
        )

    print(f"Saved figures under: {out_dir}")
    conf_subdir_resolved = resolve_conference_paper_subdir(summary_path, args.conference_subdir)
    pdf_dir = out_dir / conf_subdir_resolved
    for p in sorted(out_dir.glob("fig*.png")):
        print(" ", p.name)
    if pdf_dir.is_dir():
        for p in sorted(pdf_dir.glob("Fig*.png")) + sorted(pdf_dir.glob("Fig*.txt")):
            print(" ", conf_subdir_resolved + "/" + p.name)


if __name__ == "__main__":
    main()
