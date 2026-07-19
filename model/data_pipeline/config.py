"""Configuration for the GOES-19 / VIIRS dataset export pipeline."""

from dataclasses import dataclass
from pathlib import Path

GOES_COLLECTION_ID = "NOAA/GOES/19/MCMIPF"
VIIRS_COLLECTION_IDS = (
    "NASA/LANCE/NOAA20_VIIRS/C2",
    "NASA/LANCE/SNPP_VIIRS/C2",
)


@dataclass(frozen=True)
class BBox:
    west: float
    south: float
    east: float
    north: float


#: VIIRS "confidence" band values, per the NASA/LANCE C2 collections: 0=low, 1=nominal, 2=high.
VIIRS_CONFIDENCE_NOMINAL = 1

#: GOES-19 native pixel resolution for the MCMIPF CMI bands.
GOES_SCALE_METERS = 2000

#: Fill value written into patch pixels masked out by GOES DQF quality flags.
PATCH_FILL_VALUE = -1.0


@dataclass(frozen=True)
class PipelineConfig:
    argentina_bbox: BBox
    patch_size_px: int
    match_window_minutes: int
    min_viirs_confidence: int
    negative_to_positive_ratio: float
    min_negative_distance_km: float
    output_dir: Path


def default_config(output_dir: Path | None = None) -> PipelineConfig:
    """Reasonable phase-1 defaults. Bbox, patch size and window are tunable."""
    return PipelineConfig(
        argentina_bbox=BBox(west=-73.6, south=-55.1, east=-53.6, north=-21.8),
        patch_size_px=32,
        match_window_minutes=15,
        min_viirs_confidence=VIIRS_CONFIDENCE_NOMINAL,
        negative_to_positive_ratio=1.0,
        min_negative_distance_km=50.0,
        output_dir=output_dir if output_dir is not None else Path("model/dataset"),
    )
