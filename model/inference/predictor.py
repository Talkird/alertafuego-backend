"""Load the trained model and run batched inference over tiles."""

from pathlib import Path

import numpy as np
import torch

from model.training.config import load_model_config
from model.training.model import TinyUNet
from model.training.normalization import BandStats, normalize_patch

#: Physical constant: number of GOES-19 CMI bands the model was trained on - not a
#: tunable hyperparameter, unlike base_channels/depth below.
IN_CHANNELS = 16


def load_model(checkpoint_path: Path, device: torch.device) -> TinyUNet:
    """Builds the model at the size it was trained with (the checkpoint's
    model_config.json), so a larger model deploys by uploading files alone."""
    model_cfg = load_model_config(checkpoint_path.parent)
    model = TinyUNet(in_channels=IN_CHANNELS, base_channels=model_cfg.base_channels, depth=model_cfg.depth)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model.to(device)


#: predict_raster() runs the model on BLOCK_SIZE x BLOCK_SIZE pieces of the raster
#: plus BLOCK_MARGIN pixels of real context on each side, and keeps each block's
#: core. One pass over the whole country peaks at ~2.4 GB, too close to the
#: detector Lambda's 3 GB; the margin covers the model's receptive field, so the
#: result matches a single pass.
BLOCK_SIZE = 512
BLOCK_MARGIN = 64


def _forward(model: TinyUNet, batch: torch.Tensor, tta: bool) -> torch.Tensor:
    """Fire probabilities for batch (1, 16, H, W). tta: average over the 8 flips and
    rotations, each mapped back, so the result doesn't depend on orientation."""
    if not tta:
        return torch.sigmoid(model(batch))
    total = torch.zeros_like(batch[:, :1])
    for k in range(4):
        for flip in (False, True):
            x = torch.rot90(batch, k, dims=(2, 3))
            x = torch.flip(x, dims=(3,)) if flip else x
            y = torch.sigmoid(model(x))
            y = torch.flip(y, dims=(3,)) if flip else y
            total += torch.rot90(y, -k, dims=(2, 3))
    return total / 8


def predict_raster(
    model: TinyUNet, band_stats: BandStats, raster: np.ndarray, device: torch.device, tta: bool = False
) -> np.ndarray:
    """raster: (16, H, W) calibrated, a whole scene. Returns (H, W) fire probabilities.

    The model runs on the scene itself, not on separate 32x32 tiles: on tiles it
    detected fires almost only at tile centers, its training patches' fire position
    (docs/DECISIONS.md, 2026-10-22)."""
    normalized = normalize_patch(raster, band_stats)
    _, height, width = normalized.shape
    # The U-Net halves the size `depth` times; padding with 0 (the band mean, as
    # normalize_patch fills missing data) keeps every block divisible.
    multiple = 2 ** len(model.encoders)
    probabilities = np.zeros((height, width), dtype=np.float32)
    with torch.no_grad():
        for top in range(0, height, BLOCK_SIZE):
            for left in range(0, width, BLOCK_SIZE):
                y0, x0 = max(0, top - BLOCK_MARGIN), max(0, left - BLOCK_MARGIN)
                y1 = min(height, top + BLOCK_SIZE + BLOCK_MARGIN)
                x1 = min(width, left + BLOCK_SIZE + BLOCK_MARGIN)
                block = normalized[:, y0:y1, x0:x1]
                padded = np.zeros((16, -(-(y1 - y0) // multiple) * multiple, -(-(x1 - x0) // multiple) * multiple), np.float32)
                padded[:, : y1 - y0, : x1 - x0] = block
                probs = _forward(model, torch.from_numpy(padded)[None].to(device), tta)[0, 0].cpu().numpy()
                core_bottom, core_right = min(height, top + BLOCK_SIZE), min(width, left + BLOCK_SIZE)
                probabilities[top:core_bottom, left:core_right] = probs[top - y0 : core_bottom - y0, left - x0 : core_right - x0]
    return probabilities
