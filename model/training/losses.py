"""BCE + Tversky loss for imbalanced pixel-wise fire segmentation, weighted against misses."""

import torch
from torch import nn


def tversky_loss(logits: torch.Tensor, targets: torch.Tensor, fn_weight: float, eps: float = 1.0) -> torch.Tensor:
    """Soft Tversky loss: Dice with false negatives weighted fn_weight and false
    positives 1 - fn_weight (0.5 is exactly Dice). Above 0.5 a missed fire pixel costs
    more than a false alarm, pushing the model toward recall.

    Computed per sample and averaged, so a small fire counts as much as a large one.
    eps in both numerator and denominator so an all-background sample (mask all zero)
    with an all-background prediction scores as a perfect match instead of 0/0 -
    relevant since half the dataset is exactly this case."""
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    tp = (probs * targets).sum(dim=1)
    fp = (probs * (1 - targets)).sum(dim=1)
    fn = ((1 - probs) * targets).sum(dim=1)
    tversky = (tp + eps) / (tp + (1 - fn_weight) * fp + fn_weight * fn + eps)
    return 1 - tversky.mean()


class BCETverskyLoss(nn.Module):
    """Combines BCE's stable early-training gradient with Tversky's robustness to the
    heavy per-pixel class imbalance within a patch (a fire front is a small fraction
    of a 32x32 patch even for a "positive" sample)."""

    def __init__(self, tversky_weight: float = 0.5, fn_weight: float = 0.5) -> None:
        super().__init__()
        self.tversky_weight = tversky_weight
        self.fn_weight = fn_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.tversky_weight * tversky_loss(logits, targets, self.fn_weight) + (
            1 - self.tversky_weight
        ) * self.bce(logits, targets)
