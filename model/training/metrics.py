"""Pixel-wise segmentation metrics computed from thresholded predictions."""

import torch


def confusion_counts(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()
    return tp, fp, fn, tn


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
