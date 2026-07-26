"""Configuration for the wildfire segmentation model and its training loop."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    crop_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    base_channels: int
    depth: int
    dice_weight: float
    seg_threshold: float
    num_workers: int
    seed: int
    checkpoint_dir: Path


def default_config(checkpoint_dir: Path | None = None) -> TrainingConfig:
    """Reasonable phase-2 defaults, sized for a 32x32x16 input on CPU-only training."""
    return TrainingConfig(
        crop_size=32,
        batch_size=32,
        epochs=30,
        learning_rate=1e-3,
        weight_decay=1e-4,
        base_channels=16,
        depth=3,
        dice_weight=0.5,
        seg_threshold=0.5,
        num_workers=0,
        seed=0,
        checkpoint_dir=checkpoint_dir if checkpoint_dir is not None else Path("model/checkpoints"),
    )
