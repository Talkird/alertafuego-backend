"""PyTorch Dataset for GOES-19/VIIRS fire patches."""

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from model.training.normalization import BandStats, normalize_patch


def center_crop(array: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """Deterministic crop to (size, size) or (height, width), taken from the array's
    spatial center."""
    target_height, target_width = (size, size) if isinstance(size, int) else size
    height, width = array.shape[-2:]
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    if array.ndim == 3:
        return array[:, top : top + target_height, left : left + target_width]
    return array[top : top + target_height, left : left + target_width]


def shift(patch: np.ndarray, mask: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    """Move patch and mask by (dy, dx) pixels, filling the uncovered border with 0
    (the band mean after normalization, as normalize_patch also fills missing data)."""
    height, width = mask.shape
    shifted_patch = np.zeros_like(patch)
    shifted_mask = np.zeros_like(mask)
    src_y, dst_y = slice(max(0, -dy), height - max(0, dy)), slice(max(0, dy), height - max(0, -dy))
    src_x, dst_x = slice(max(0, -dx), width - max(0, dx)), slice(max(0, dx), width - max(0, -dx))
    shifted_patch[:, dst_y, dst_x] = patch[:, src_y, src_x]
    shifted_mask[dst_y, dst_x] = mask[src_y, src_x]
    return shifted_patch, shifted_mask


def _random_augment(patch: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same random flip/rotation to patch and mask - both are geometric,
    physically valid transforms for a satellite patch with no canonical orientation."""
    if rng.random() < 0.5:
        patch, mask = patch[:, :, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        patch, mask = patch[:, ::-1, :], mask[::-1, :]
    k = rng.randint(0, 3)
    if k:
        patch = np.rot90(patch, k, axes=(1, 2))
        mask = np.rot90(mask, k, axes=(0, 1))
    return np.ascontiguousarray(patch), np.ascontiguousarray(mask)


def list_split_files(
    manifest_path: Path, dataset_dir: Path, split: str, limit: int | None = None, seed: int = 0
) -> list[Path]:
    """File paths for one manifest split, optionally randomly subsampled to limit."""
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    files = [dataset_dir / row["filename"] for row in rows if row["split"] == split]
    if limit is not None and len(files) > limit:
        files = random.Random(seed).sample(files, limit)
    return files


class FirePatchDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        dataset_dir: Path,
        split: str,
        band_stats: BandStats,
        crop_size: int,
        augment: bool,
        limit: int | None = None,
        seed: int = 0,
        max_shift: int = 0,
    ) -> None:
        """augment: random flips/rotations and a random shift of up to max_shift
        pixels (train). Without it, max_shift still applies, as a fixed shift per
        sample, so val/test scores are repeatable."""
        self.dataset_dir = dataset_dir
        self.band_stats = band_stats
        self.crop_size = crop_size
        self.augment = augment
        self.max_shift = max_shift
        self.seed = seed
        self.files = list_split_files(manifest_path, dataset_dir, split, limit, seed)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[index])
        patch = center_crop(data["patch"], self.crop_size)
        mask = center_crop(data["mask"], self.crop_size).astype(np.float32)
        patch = normalize_patch(patch, self.band_stats)
        # Training draws from the `random` module, which DataLoader seeds differently
        # in each worker; a per-dataset Random would repeat the same sequence in all.
        rng = random if self.augment else random.Random(self.seed * 1_000_003 + index)
        if self.augment:
            patch, mask = _random_augment(patch, mask, rng)
        if self.max_shift:
            dy, dx = (rng.randint(-self.max_shift, self.max_shift) for _ in range(2))
            patch, mask = shift(patch, mask, dy, dx)
        return torch.from_numpy(patch).float(), torch.from_numpy(mask).float()
