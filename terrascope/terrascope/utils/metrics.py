from typing import Dict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _safe_div(num, den):
    return float(num / (den + 1e-8))


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
