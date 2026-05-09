from typing import Dict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _safe_div(num, den):
    return float(num / (den + 1e-8))


def confusion_counts_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> tuple[float, float, float, float]:
    """Binary segmentation confusion summed over flattened spatial batch (streaming-friendly)."""
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).long().clamp(0, 1).view(-1)
    tgt = targets.long().clamp(0, 1).view(-1)
    tp = float(((pred == 1) & (tgt == 1)).sum().item())
    fp = float(((pred == 1) & (tgt == 0)).sum().item())
    tn = float(((pred == 0) & (tgt == 0)).sum().item())
    fn = float(((pred == 0) & (tgt == 1)).sum().item())
    return tp, fp, tn, fn


def pixel_metrics_from_confusion(tp: float, fp: float, tn: float, fn: float) -> dict[str, float]:
    """Paper Eq. (8)-style ratios from summed counts across the evaluated set."""
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": f1,
    }


def segmentation_metrics_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    probs = torch.sigmoid(logits).detach().cpu().flatten().numpy()
    y = targets.detach().cpu().flatten().numpy().astype(np.uint8)
    pred = (probs >= threshold).astype(np.uint8)

    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    # sklearn ranking metrics are undefined for one-class targets and emit warnings.
    has_pos = bool((y == 1).any())
    has_neg = bool((y == 0).any())
    if has_pos and has_neg:
        auroc = float(roc_auc_score(y, probs))
        auprc = float(average_precision_score(y, probs))
        p_curve, r_curve, th = precision_recall_curve(y, probs)
        f1_curve = (2 * p_curve * r_curve) / (p_curve + r_curve + 1e-8)
        best_idx = int(np.nanargmax(f1_curve))
        best_f1 = float(f1_curve[best_idx])
        best_threshold = float(th[min(best_idx, max(len(th) - 1, 0))]) if len(th) > 0 else 0.5
    else:
        auroc = float("nan")
        auprc = float("nan")
        best_f1 = f1
        best_threshold = 0.5

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": f1,
        "auroc": auroc,
        "auprc": auprc,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "fps": float("nan"),
        "peak_memory_mb": float("nan"),
        "gflops": float("nan"),
        "trainable_params_m": float("nan"),
    }


def ranking_metrics_from_prob_arrays(
    probs: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """Dataset-level AUROC / AUPRC / PR-curve best-F1 (paper Eq. 9 style on pooled pixels)."""
    y = y.astype(np.uint8)
    has_pos = bool((y == 1).any())
    has_neg = bool((y == 0).any())
    if not (has_pos and has_neg):
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "best_f1": float("nan"),
            "best_threshold": 0.5,
        }
    auroc = float(roc_auc_score(y, probs))
    auprc = float(average_precision_score(y, probs))
    p_curve, r_curve, th = precision_recall_curve(y, probs)
    f1_curve = (2 * p_curve * r_curve) / (p_curve + r_curve + 1e-8)
    best_idx = int(np.nanargmax(f1_curve))
    best_f1 = float(f1_curve[best_idx])
    best_threshold = float(th[min(best_idx, max(len(th) - 1, 0))]) if len(th) > 0 else 0.5
    return {
        "auroc": auroc,
        "auprc": auprc,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
    }
