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


def precision(tp: float, fp: float) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp: float, fn: float) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1(precision_value: float, recall_value: float) -> float:
    denom = precision_value + recall_value
    return 2 * precision_value * recall_value / denom if denom > 0 else 0.0


def iou(tp: float, fp: float, fn: float) -> float:
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0
