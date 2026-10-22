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
    tversky_weight: float
    #: Tversky weight of a missed fire pixel vs a false alarm (losses.tversky_loss).
    #: Above 0.5 favors recall; 0.5 is plain Dice.
    fn_weight: float
    seg_threshold: float
    #: Share of val fires the chosen threshold must detect (metrics.fire_counts): the
    #: highest threshold reaching it is kept. Missing a fire is worse than a false alarm.
    recall_target: float
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
        tversky_weight=0.5,
        fn_weight=0.7,
        seg_threshold=0.5,
        recall_target=0.95,
        max_shift=8,
        num_workers=0,
        seed=0,
        checkpoint_dir=checkpoint_dir if checkpoint_dir is not None else Path("model/checkpoints"),
    )


#: Candidates for the segmentation threshold, chosen on val. Fire pixels are rare,
#: so the model's fire probabilities run low: the best threshold can sit well
#: under 0.1, hence the fine steps at the bottom (and lower when chasing recall).
THRESHOLD_CANDIDATES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08] + [round(0.05 * i, 2) for i in range(2, 19)]


@dataclass(frozen=True)
class ModelConfig:
    """What inference needs to rebuild and use a checkpoint: its size and the
    threshold chosen on val. Saved next to model_best.pt as model_config.json.

    night_threshold, when set, applies where the sun is below the horizon and
    threshold everywhere else (evaluate_scenes calibrate picks both)."""

    base_channels: int
    depth: int
    threshold: float
    night_threshold: float | None = None


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


def pick_threshold(sweep: list[dict], recall_target: float) -> dict:
    """The highest-threshold result in sweep whose fire recall reaches recall_target
    (the fewest false alarms without missing more fires), or, if none does, the one
    with the highest fire recall."""
    reaching = [m for m in sweep if m["fire_recall"] >= recall_target]
    if reaching:
        return max(reaching, key=lambda m: m["threshold"])
    return max(sweep, key=lambda m: (m["fire_recall"], m["threshold"]))
