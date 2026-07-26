"""Configuration for on-demand model serving."""

from dataclasses import dataclass
from pathlib import Path

from model.data_pipeline.config import BBox, default_config as default_pipeline_config

#: La Rioja/Catamarca/San Juan cluster - real fire activity observed here during
#: phase 1 smoke testing. ~110x95px at GOES's scale, well under any Earth Engine
#: pixel cap, so this endpoint never needs chunking.
DEMO_BBOX = BBox(west=-68.0, south=-31.0, east=-66.0, north=-29.0)

#: 481x481 = 231,361px, ~12% under sampleRectangle's 262,144-pixel cap, leaving
#: margin for the known +1 bbox-rounding off-by-one (see docs/DECISIONS.md).
CHUNK_SIZE_PX = 480


@dataclass(frozen=True)
class InferenceConfig:
    demo_bbox: BBox
    argentina_bbox: BBox
    chunk_size_px: int
    patch_size_px: int
    checkpoint_dir: Path


def default_config(checkpoint_dir: Path | None = None) -> InferenceConfig:
    return InferenceConfig(
        demo_bbox=DEMO_BBOX,
        argentina_bbox=default_pipeline_config().argentina_bbox,
        chunk_size_px=CHUNK_SIZE_PX,
        patch_size_px=32,
        checkpoint_dir=checkpoint_dir if checkpoint_dir is not None else Path("model/checkpoints"),
    )
