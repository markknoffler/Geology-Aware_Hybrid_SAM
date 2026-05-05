import cv2
import numpy as np
import torch


def get_statistics(output: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    pred = (output >= threshold).to(torch.int64)
    tgt = (target > 0).to(torch.int64)
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if tgt.dim() == 3:
        tgt = tgt.unsqueeze(1)

    dims = (1, 2, 3)
    tp = (pred * tgt).sum(dim=dims)
    fp = (pred * (1 - tgt)).sum(dim=dims)
    fn = ((1 - pred) * tgt).sum(dim=dims)
    tn = ((1 - pred) * (1 - tgt)).sum(dim=dims)
    return tp, fp, fn, tn


def iou(tp, fp, fn, tn, reduction: str = "micro-imagewise"):
    del tn
    score = (tp.float() + 1e-6) / (tp.float() + fp.float() + fn.float() + 1e-6)
    return score.mean() if reduction == "micro-imagewise" else score.sum()


def f1(tp, fp, fn, tn, reduction: str = "micro-imagewise"):
    del tn
    score = (2.0 * tp.float() + 1e-6) / (2.0 * tp.float() + fp.float() + fn.float() + 1e-6)
    return score.mean() if reduction == "micro-imagewise" else score.sum()


def acc(tp, fp, fn, tn, reduction: str = "micro-imagewise"):
    score = (tp.float() + tn.float() + 1e-6) / (tp.float() + fp.float() + fn.float() + tn.float() + 1e-6)
    return score.mean() if reduction == "micro-imagewise" else score.sum()


def recall(tp, fp, fn, tn, reduction: str = "micro-imagewise"):
    del fp, tn
    score = (tp.float() + 1e-6) / (tp.float() + fn.float() + 1e-6)
    return score.mean() if reduction == "micro-imagewise" else score.sum()


def precision(tp, fp, fn, tn, reduction: str = "micro-imagewise"):
    del fn, tn
    score = (tp.float() + 1e-6) / (tp.float() + fp.float() + 1e-6)
    return score.mean() if reduction == "micro-imagewise" else score.sum()


def pixel_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    tp, fp, fn, tn = get_statistics(probs, target, threshold=threshold)
    return {
        "acc": float(acc(tp, fp, fn, tn).item()),
        "precision": float(precision(tp, fp, fn, tn).item()),
        "recall": float(recall(tp, fp, fn, tn).item()),
        "f1": float(f1(tp, fp, fn, tn).item()),
        "iou": float(iou(tp, fp, fn, tn).item()),
        "tp": int(tp.sum().item()),
        "fp": int(fp.sum().item()),
        "fn": int(fn.sum().item()),
        "tn": int(tn.sum().item()),
    }


def _binarize(prob: np.ndarray, threshold: float) -> np.ndarray:
    return (prob >= threshold).astype(np.uint8)


def mask_to_instances(mask: np.ndarray, min_area: int = 20):
    num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
    instances = []
    for component_id in range(1, num_labels):
        component = (labels == component_id).astype(np.uint8)
        if int(component.sum()) >= min_area:
            instances.append(component)
    return instances


def instance_scores(prob: np.ndarray, instances):
    scores = []
    for instance in instances:
        pixels = prob[instance > 0]
        scores.append(float(pixels.max()) if pixels.size > 0 else 0.0)
    return scores


def _pr_curve(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]
    tp = 0
    fp = 0
    positives = labels.sum()
    precisions = [1.0]
    recalls = [0.0]
    thresholds = [1.0]
    last = None
    for score, label in zip(scores, labels):
        if label == 1:
            tp += 1
        else:
            fp += 1
        if last is None or score != last:
            precisions.append(tp / (tp + fp + 1e-6))
            recalls.append(tp / (positives + 1e-6))
            thresholds.append(float(score))
            last = score
    precisions.append(tp / (tp + fp + 1e-6))
    recalls.append(tp / (positives + 1e-6))
    thresholds.append(0.0)
    return np.array(precisions), np.array(recalls), np.array(thresholds)


def _auprc(precisions: np.ndarray, recalls: np.ndarray) -> float:
    order = np.argsort(recalls)
    return float(np.trapz(precisions[order], recalls[order]))


def _roc_curve(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]
    tp = 0
    fp = 0
    positives = labels.sum()
    negatives = len(labels) - positives + 1e-6
    tpr = [0.0]
    fpr = [0.0]
    last = None
    for score, label in zip(scores, labels):
        if label == 1:
            tp += 1
        else:
            fp += 1
        if last is None or score != last:
            tpr.append(tp / (positives + 1e-6))
            fpr.append(fp / negatives)
            last = score
    tpr.append(1.0)
    fpr.append(1.0)
    return np.array(fpr), np.array(tpr)


def _auroc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


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

    image_scores = []
    image_labels = []
    for prob, mask in zip(probs, gt):
        label = 1 if mask.sum() > 0 else 0
        pred_bin = _binarize(prob, prob_thr_for_instances)
        instances = mask_to_instances(pred_bin, min_area=min_area)
        score = float(np.max(instance_scores(prob, instances))) if len(instances) > 0 else 0.0
        image_scores.append(score)
        image_labels.append(label)

    scores = np.asarray(image_scores, dtype=np.float32)
    labels = np.asarray(image_labels, dtype=np.int32)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {"auroc": 0.0, "auprc": 0.0, "best_f1": 0.0, "best_threshold": 0.5}

    prec, rec, thr = _pr_curve(scores, labels)
    auprc = _auprc(prec, rec)
    fpr, tpr = _roc_curve(scores, labels)
    auroc = _auroc(fpr, tpr)
    f1s = 2 * prec * rec / (prec + rec + 1e-6)
    best_idx = int(np.argmax(f1s))
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "best_f1": float(f1s[best_idx]),
        "best_threshold": float(thr[best_idx]),
    }
