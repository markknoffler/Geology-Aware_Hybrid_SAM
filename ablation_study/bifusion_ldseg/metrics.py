from typing import Dict, List

import numpy as np
import torch

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def pixel_confusion_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, int]:
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    gt = (target.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.uint8)
    pred = (probs >= threshold).astype(np.uint8)
    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    dsc = _safe_div(2 * tp, 2 * tp + fp + fn)
    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dsc": dsc,
    }


def expected_calibration_error(logits: torch.Tensor, target: torch.Tensor, n_bins: int = 15) -> float:
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    labels = (target.detach().cpu().numpy().reshape(-1) > 0.5).astype(np.float32)
    pred = (probs >= 0.5).astype(np.float32)
    conf = np.where(pred > 0.0, probs, 1.0 - probs)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        idx = (conf > lo) & (conf <= hi)
        if idx.sum() == 0:
            continue
        acc_bin = np.mean(pred[idx] == labels[idx])
        conf_bin = np.mean(conf[idx])
        ece += np.abs(acc_bin - conf_bin) * (idx.sum() / conf.shape[0])
    return float(ece)


def _surface_distances(pred: np.ndarray, gt: np.ndarray):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
    if not pred.any() or not gt.any():
        return np.array([np.inf], dtype=np.float32), np.array([np.inf], dtype=np.float32)

    pred_edge = pred ^ binary_erosion(pred)
    gt_edge = gt ^ binary_erosion(gt)
    dt_pred = distance_transform_edt(~pred_edge)
    dt_gt = distance_transform_edt(~gt_edge)
    dist_pred_to_gt = dt_gt[pred_edge]
    dist_gt_to_pred = dt_pred[gt_edge]
    return dist_pred_to_gt, dist_gt_to_pred


def assd_hd_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    if not SCIPY_OK:
        return {"assd": float("nan"), "hd": float("nan")}

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    gt = target.detach().cpu().numpy()
    preds = (probs >= threshold).astype(np.uint8)
    gts = (gt > 0.5).astype(np.uint8)

    assd_vals: List[float] = []
    hd_vals: List[float] = []
    for i in range(preds.shape[0]):
        p = preds[i, 0]
        g = gts[i, 0]
        d1, d2 = _surface_distances(p, g)
        if np.isinf(d1).any() or np.isinf(d2).any():
            continue
        assd_vals.append(float((d1.mean() + d2.mean()) / 2.0))
        hd_vals.append(float(max(d1.max(), d2.max())))

    if len(assd_vals) == 0:
        return {"assd": float("nan"), "hd": float("nan")}
    return {"assd": float(np.mean(assd_vals)), "hd": float(np.mean(hd_vals))}
