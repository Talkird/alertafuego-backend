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


def predict_tiles(model: TinyUNet, band_stats: BandStats, tiles: np.ndarray, device: torch.device) -> np.ndarray:
    """tiles: (N, 16, H, W) calibrated. Returns (N, H, W) fire probabilities."""
    normalized = np.stack([normalize_patch(tile, band_stats) for tile in tiles], axis=0)
    batch = torch.from_numpy(normalized).float().to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch))
    return probs.squeeze(1).cpu().numpy()
