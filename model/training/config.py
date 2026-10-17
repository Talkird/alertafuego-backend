"""Configuration for the wildfire segmentation model and its training loop."""

import json
from dataclasses import asdict, dataclass
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
    #: Patches are centered on the fire; shifting them up to this many pixels (train:
    #: random, val/test: fixed per sample) keeps the model from learning "the fire is
    #: in the middle", which never holds on the inference tiles. 0 disables it.
    max_shift: int
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
        max_shift=8,
        num_workers=0,
        seed=0,
        checkpoint_dir=checkpoint_dir if checkpoint_dir is not None else Path("model/checkpoints"),
    )


@dataclass(frozen=True)
class ModelConfig:
    """What inference needs to rebuild and use a checkpoint: its size and the
    threshold chosen on val. Saved next to model_best.pt as model_config.json."""

    base_channels: int
    depth: int
    threshold: float


MODEL_CONFIG_FILENAME = "model_config.json"


def save_model_config(model_cfg: ModelConfig, checkpoint_dir: Path) -> None:
    (checkpoint_dir / MODEL_CONFIG_FILENAME).write_text(json.dumps(asdict(model_cfg), indent=2) + "\n")


def load_model_config(checkpoint_dir: Path) -> ModelConfig:
    """The checkpoint's model_config.json, or the defaults for checkpoints trained
    before it existed (which used them)."""
    path = checkpoint_dir / MODEL_CONFIG_FILENAME
    if path.exists():
        return ModelConfig(**json.loads(path.read_text()))
    cfg = default_config()
    return ModelConfig(base_channels=cfg.base_channels, depth=cfg.depth, threshold=cfg.seg_threshold)
