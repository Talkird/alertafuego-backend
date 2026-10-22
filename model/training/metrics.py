"""Pixel-wise segmentation metrics computed from thresholded predictions."""

import torch
import torch.nn.functional as F


def confusion_counts(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    return tp, fp, fn, tn


def tolerant_counts(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float, radius: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Counts for scoring with a tolerance of `radius` pixels (1: one 2 km GOES pixel,
    about GOES-VIIRS geolocation error), as (predicted fire pixels with a true fire
    within radius, predicted fire pixels, true fire pixels with a prediction within
    radius, true fire pixels). Precision = first/second, recall = third/fourth.

    Strict counts call a fire predicted one pixel off both a false positive and a
    miss; for an alert, that prediction is right."""
    preds = (torch.sigmoid(logits) >= threshold).float()
    size = 2 * radius + 1
    near_target = F.max_pool2d(targets, size, stride=1, padding=radius)
    near_pred = F.max_pool2d(preds, size, stride=1, padding=radius)
    return (preds * near_target).sum(), preds.sum(), (targets * near_pred).sum(), targets.sum()


def fire_counts(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float, radius: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample counts, as (fire samples with a prediction within `radius` of a fire
    pixel, fire samples, no-fire samples with any prediction, no-fire samples).

    Each fire sample is one VIIRS detection, so first/second is the share of fires the
    model alerts on (fire recall), the metric that matters for missing a fire; third/
    fourth is the false alarm rate on fire-free patches."""
    preds = (torch.sigmoid(logits) >= threshold).float()
    near_pred = F.max_pool2d(preds, 2 * radius + 1, stride=1, padding=radius)
    has_fire = targets.flatten(1).amax(dim=1) > 0
    detected = (targets * near_pred).flatten(1).amax(dim=1) > 0
    alarm = preds.flatten(1).amax(dim=1) > 0
    counts = [(detected & has_fire), has_fire, (alarm & ~has_fire), ~has_fire]
    return tuple(c.float().sum() for c in counts)


def precision(tp: float, fp: float) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp: float, fn: float) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(precision_value: float, recall_value: float) -> float:
    denom = precision_value + recall_value
    return 2 * precision_value * recall_value / denom if denom > 0 else 0.0


def f_beta(precision_value: float, recall_value: float, beta: float) -> float:
    """F-score weighting recall beta times as much as precision (F2: beta=2)."""
    denom = beta**2 * precision_value + recall_value
    return (1 + beta**2) * precision_value * recall_value / denom if denom > 0 else 0.0


def iou(tp: float, fp: float, fn: float) -> float:
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0
