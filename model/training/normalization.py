"""Per-band normalization statistics for GOES-19 patches.

Stats must be computed from the train split only (never val/test, to avoid leakage)
and reused unchanged at inference time - hence they are persisted to disk rather than
recomputed ad hoc.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model.data_pipeline.config import PATCH_FILL_VALUE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BandStats:
    mean: np.ndarray
    std: np.ndarray


def compute_band_stats(paths: list[Path]) -> BandStats:
    """Per-band mean/std over paths, excluding PATCH_FILL_VALUE sentinel pixels."""
    n_bands = np.load(paths[0])["patch"].shape[0]
    total = np.zeros(n_bands, dtype=np.float64)
    total_sq = np.zeros(n_bands, dtype=np.float64)
    count = np.zeros(n_bands, dtype=np.float64)

    for path in paths:
        patch = np.load(path)["patch"].astype(np.float64)
        valid = patch != PATCH_FILL_VALUE
        total += (patch * valid).sum(axis=(1, 2))
        total_sq += ((patch**2) * valid).sum(axis=(1, 2))
        count += valid.sum(axis=(1, 2))

    mean = total / count
    variance = total_sq / count - mean**2
    std = np.sqrt(np.clip(variance, a_min=1e-8, a_max=None))
    logger.info("Computed band stats from %d files", len(paths))
    return BandStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


def save_band_stats(stats: BandStats, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mean": stats.mean.tolist(), "std": stats.std.tolist()}))


def load_band_stats(path: Path) -> BandStats:
    data = json.loads(path.read_text())
    return BandStats(mean=np.array(data["mean"], dtype=np.float32), std=np.array(data["std"], dtype=np.float32))


def normalize_patch(patch: np.ndarray, stats: BandStats) -> np.ndarray:
    """Z-score normalize per band; sentinel-filled pixels are zeroed after normalizing."""
    was_fill = patch == PATCH_FILL_VALUE
    normalized = (patch - stats.mean[:, None, None]) / stats.std[:, None, None]
    normalized[was_fill] = 0.0
    return normalized.astype(np.float32)
