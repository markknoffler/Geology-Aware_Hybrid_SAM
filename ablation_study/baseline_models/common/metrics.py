import cv2
import numpy as np
import torch


def pixel_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).to(torch.int64)
    tgt = (target > 0).to(torch.int64)
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if tgt.dim() == 3:
        tgt = tgt.unsqueeze(1)

    dims = (1, 2, 3)
    tp = (pred * tgt).sum(dim=dims).float()
    fp = (pred * (1 - tgt)).sum(dim=dims).float()
    fn = ((1 - pred) * tgt).sum(dim=dims).float()
    tn = ((1 - pred) * (1 - tgt)).sum(dim=dims).float()

    acc = ((tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6)).mean().item()
    prec = ((tp + 1e-6) / (tp + fp + 1e-6)).mean().item()
    rec = ((tp + 1e-6) / (tp + fn + 1e-6)).mean().item()
    f1 = ((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).mean().item()
    iou = ((tp + 1e-6) / (tp + fp + fn + 1e-6)).mean().item()
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "iou": float(iou)}


def _binarize(prob: np.ndarray, threshold: float) -> np.ndarray:
    return (prob >= threshold).astype(np.uint8)


def _mask_to_instances(mask: np.ndarray, min_area: int = 20):
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    instances = []
    for comp_id in range(1, num_labels):
        comp = (labels == comp_id).astype(np.uint8)
        if int(comp.sum()) >= min_area:
            instances.append(comp)
    return instances


def _instance_scores(prob: np.ndarray, instances):
    scores = []
    for inst in instances:
        vals = prob[inst > 0]
        scores.append(float(vals.max()) if vals.size else 0.0)
    return scores


def _pr_curve(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]
    tp = 0
    fp = 0
    p = labels.sum()
    precisions = [1.0]
    recalls = [0.0]
    thresholds = [1.0]
    last = None
    for score, label in zip(scores, labels):
        tp += int(label == 1)
        fp += int(label == 0)
        if last is None or score != last:
            precisions.append(tp / (tp + fp + 1e-6))
            recalls.append(tp / (p + 1e-6))
            thresholds.append(float(score))
            last = score
    precisions.append(tp / (tp + fp + 1e-6))
    recalls.append(tp / (p + 1e-6))
    thresholds.append(0.0)
    return np.asarray(precisions), np.asarray(recalls), np.asarray(thresholds)


def _roc_curve(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]
    tp = 0
    fp = 0
    p = labels.sum()
    n = len(labels) - p + 1e-6
    tpr = [0.0]
    fpr = [0.0]
    last = None
    for score, label in zip(scores, labels):
        tp += int(label == 1)
        fp += int(label == 0)
        if last is None or score != last:
            tpr.append(tp / (p + 1e-6))
            fpr.append(fp / n)
            last = score
    tpr.append(1.0)
    fpr.append(1.0)
    return np.asarray(fpr), np.asarray(tpr)


@torch.no_grad()
def image_level_metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    prob_thr_for_instances: float = 0.5,
    min_area: int = 20,
):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    gt = target.detach().cpu().numpy().astype(np.uint8)
    if gt.ndim == 4:
        gt = gt[:, 0]
    if probs.ndim == 4:
        probs = probs[:, 0]

    img_scores = []
    img_labels = []
    for prob, mask in zip(probs, gt):
        label = 1 if mask.sum() > 0 else 0
        pred_bin = _binarize(prob, prob_thr_for_instances)
        insts = _mask_to_instances(pred_bin, min_area=min_area)
        score = float(np.max(_instance_scores(prob, insts))) if len(insts) else 0.0
        img_scores.append(score)
        img_labels.append(label)

    scores = np.asarray(img_scores, dtype=np.float32)
    labels = np.asarray(img_labels, dtype=np.int32)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {"auroc": 0.0, "auprc": 0.0, "best_f1": 0.0, "best_threshold": 0.5}

    prec, rec, thr = _pr_curve(scores, labels)
    auprc = float(np.trapz(prec[np.argsort(rec)], rec[np.argsort(rec)]))

    fpr, tpr = _roc_curve(scores, labels)
    auroc = float(np.trapz(tpr[np.argsort(fpr)], fpr[np.argsort(fpr)]))

    f1s = 2 * prec * rec / (prec + rec + 1e-6)
    best_idx = int(np.argmax(f1s))
    return {
        "auroc": auroc,
        "auprc": auprc,
        "best_f1": float(f1s[best_idx]),
        "best_threshold": float(thr[best_idx]),
    }
