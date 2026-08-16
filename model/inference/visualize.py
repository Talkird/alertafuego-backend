"""Debug utility: render a detection run's raster + predicted fire pixels as a PNG,
for visually sanity-checking the model. Temporary - not the long-term image storage
path (that'll be a DB column + object storage, once there's time to design it)."""

import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model.data_pipeline.config import PATCH_FILL_VALUE
from model.data_pipeline.patches import CMI_BANDS

logger = logging.getLogger(__name__)

#: 3.9um shortwave IR - fire hotspots saturate this band, easiest single band to
#: visually correlate against the model's predicted pixels.
FIRE_BAND = "CMI_C07"

#: ABI has no native green band - synthetic green per NOAA's published true-color
#: recipe (Bah et al. 2018): a blend of red, veggie/NIR and blue.
_TRUE_COLOR_RED = "CMI_C02"
_TRUE_COLOR_VEGGIE = "CMI_C03"
_TRUE_COLOR_BLUE = "CMI_C01"


def save_debug_image(
    raster: np.ndarray,
    pixel_detections: list[tuple[int, int, float]],
    image_time: datetime,
    threshold: float,
    output_dir: Path,
) -> Path:
    """pixel_detections is (row, col, probability) in raster pixel coordinates
    (unfiltered by the Argentina polygon - shows the model's raw output)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    band = raster[CMI_BANDS.index(FIRE_BAND)]
    band = np.ma.masked_where(band == PATCH_FILL_VALUE, band)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(band, cmap="inferno")
    if pixel_detections:
        rows, cols, _probs = zip(*pixel_detections)
        ax.scatter(cols, rows, s=20, facecolors="none", edgecolors="cyan", linewidths=1)
    ax.set_title(f"{image_time.isoformat()}  threshold={threshold}  {len(pixel_detections)} detections")
    ax.axis("off")

    output_path = output_dir / f"{image_time.strftime('%Y%m%dT%H%M%S')}_ir.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved debug image to %s", output_path)
    return output_path


def save_true_color_image(raster: np.ndarray, image_time: datetime, output_dir: Path) -> Path:
    """Reflectance-only composite - useless at night (visible bands read ~0 without
    sunlight), included anyway since that's a real constraint of this satellite,
    not a rendering bug."""
    output_dir.mkdir(parents=True, exist_ok=True)
    red = raster[CMI_BANDS.index(_TRUE_COLOR_RED)]
    veggie = raster[CMI_BANDS.index(_TRUE_COLOR_VEGGIE)]
    blue = raster[CMI_BANDS.index(_TRUE_COLOR_BLUE)]
    green = 0.45 * red + 0.10 * veggie + 0.45 * blue

    rgb = np.clip(np.stack([red, green, blue], axis=-1), 0, 1) ** 0.7  # gamma so it isn't flat/dark
    mask = raster[CMI_BANDS.index(_TRUE_COLOR_RED)] == PATCH_FILL_VALUE
    rgb[mask] = 0

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(rgb)
    ax.set_title(f"{image_time.isoformat()}  true color")
    ax.axis("off")

    output_path = output_dir / f"{image_time.strftime('%Y%m%dT%H%M%S')}_true_color.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved true-color image to %s", output_path)
    return output_path
