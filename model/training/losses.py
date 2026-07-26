"""BCE + Dice loss for imbalanced pixel-wise fire segmentation."""

import torch
from torch import nn


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss. eps in both numerator and denominator so an all-background
    sample (mask all zero) with an all-background prediction scores as a perfect
    match instead of 0/0 - relevant since half the dataset is exactly this case."""
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combines BCE's stable early-training gradient with Dice's robustness to the
    heavy per-pixel class imbalance within a patch (a fire front is a small fraction
    of a 32x32 patch even for a "positive" sample)."""

    def __init__(self, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * dice_loss(logits, targets) + (1 - self.dice_weight) * self.bce(logits, targets)
