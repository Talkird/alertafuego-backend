"""Unit tests for model/training: segmentation metrics, BCE+Dice loss, per-band
normalization and the dataset's crop/shift helpers. No training is run."""

import numpy as np
import pytest
import torch

from model.data_pipeline.config import PATCH_FILL_VALUE
from model.training.dataset import center_crop, shift
from model.training.losses import BCEDiceLoss, dice_loss
from model.training.metrics import confusion_counts, f1, iou, precision, recall, tolerant_counts
from model.training.normalization import (
    BandStats,
    compute_band_stats,
    load_band_stats,
    normalize_patch,
    save_band_stats,
)

BIG = 20.0  # sigmoid(20) ~ 1, sigmoid(-20) ~ 0


def _logits(pred: list[list[int]]) -> torch.Tensor:
    return (torch.tensor(pred, dtype=torch.float32) * 2 - 1).mul(BIG)[None, None]


def _targets(mask: list[list[int]]) -> torch.Tensor:
    return torch.tensor(mask, dtype=torch.float32)[None, None]


def test_confusion_counts():
    logits = _logits([[1, 1], [0, 0]])
    targets = _targets([[1, 0], [1, 0]])
    tp, fp, fn, tn = (float(x) for x in confusion_counts(logits, targets, threshold=0.5))
    assert (tp, fp, fn, tn) == (1, 1, 1, 1)


def test_tolerant_counts_accept_a_prediction_one_pixel_off():
    logits = _logits([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    targets = _targets([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    strict_tp, _, _, _ = confusion_counts(logits, targets, 0.5)
    hit_preds, preds, hit_targets, positives = (float(x) for x in tolerant_counts(logits, targets, 0.5, radius=1))
    assert float(strict_tp) == 0
    assert (hit_preds, preds, hit_targets, positives) == (1, 1, 1, 1)


def test_tolerant_counts_reject_a_prediction_two_pixels_off():
    logits = _logits([[1, 0, 0]])
    targets = _targets([[0, 0, 1]])
    hit_preds, preds, hit_targets, positives = (float(x) for x in tolerant_counts(logits, targets, 0.5, radius=1))
    assert (hit_preds, hit_targets) == (0, 0)
    assert (preds, positives) == (1, 1)


def test_scalar_metrics():
    assert precision(3, 1) == pytest.approx(0.75)
    assert recall(3, 3) == pytest.approx(0.5)
    assert f1(0.75, 0.5) == pytest.approx(0.6)
    assert iou(3, 1, 2) == pytest.approx(0.5)


@pytest.mark.parametrize("metric,args", [(precision, (0, 0)), (recall, (0, 0)), (f1, (0, 0)), (iou, (0, 0, 0))])
def test_scalar_metrics_are_zero_without_division_by_zero(metric, args):
    assert metric(*args) == 0.0


def test_dice_loss_perfect_on_all_background():
    """Half the dataset is all-background: predicting none of it must not be a 0/0."""
    logits = torch.full((2, 1, 4, 4), -BIG)
    targets = torch.zeros((2, 1, 4, 4))
    assert float(dice_loss(logits, targets)) == pytest.approx(0.0, abs=1e-3)


def test_dice_loss_high_when_prediction_misses():
    logits = torch.full((1, 1, 4, 4), -BIG)
    targets = torch.zeros((1, 1, 4, 4))
    targets[0, 0, 1, 1] = 1
    assert float(dice_loss(logits, targets)) > 0.4


def test_bce_dice_loss_mixes_both_terms():
    logits = torch.randn(2, 1, 8, 8, generator=torch.Generator().manual_seed(0))
    targets = (torch.rand(2, 1, 8, 8, generator=torch.Generator().manual_seed(1)) > 0.8).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss(logits, targets)
    loss = BCEDiceLoss(dice_weight=0.25)(logits, targets)
    assert float(loss) == pytest.approx(float(0.25 * dice + 0.75 * bce), rel=1e-5)


def _save_patch(path, patch):
    np.savez(path, patch=patch.astype(np.float32), mask=np.zeros(patch.shape[1:], dtype=np.uint8))
    return path


def test_compute_band_stats_ignores_fill_pixels(tmp_path):
    a = np.array([[[1.0, 3.0], [PATCH_FILL_VALUE, PATCH_FILL_VALUE]], [[10.0, 10.0], [10.0, 10.0]]])
    b = np.array([[[5.0, PATCH_FILL_VALUE], [PATCH_FILL_VALUE, PATCH_FILL_VALUE]], [[20.0, 20.0], [20.0, 20.0]]])
    stats = compute_band_stats([_save_patch(tmp_path / "a.npz", a), _save_patch(tmp_path / "b.npz", b)])
    np.testing.assert_allclose(stats.mean, [3.0, 15.0])
    np.testing.assert_allclose(stats.std, [np.std([1, 3, 5]), 5.0], rtol=1e-5)


def test_compute_band_stats_constant_band_has_positive_std(tmp_path):
    patch = np.ones((1, 2, 2))
    stats = compute_band_stats([_save_patch(tmp_path / "a.npz", patch)])
    assert stats.std[0] > 0


def test_band_stats_round_trip(tmp_path):
    stats = BandStats(mean=np.array([1.5, 2.5], dtype=np.float32), std=np.array([0.5, 4.0], dtype=np.float32))
    path = tmp_path / "nested" / "norm_stats.json"
    save_band_stats(stats, path)
    loaded = load_band_stats(path)
    np.testing.assert_array_equal(loaded.mean, stats.mean)
    np.testing.assert_array_equal(loaded.std, stats.std)


def test_normalize_patch_zscores_and_zeroes_fill():
    stats = BandStats(mean=np.array([2.0], dtype=np.float32), std=np.array([2.0], dtype=np.float32))
    patch = np.array([[[4.0, PATCH_FILL_VALUE]]], dtype=np.float32)
    normalized = normalize_patch(patch, stats)
    np.testing.assert_array_equal(normalized, [[[1.0, 0.0]]])
    assert normalized.dtype == np.float32


def test_center_crop_2d_and_3d():
    array = np.arange(36).reshape(6, 6)
    np.testing.assert_array_equal(center_crop(array, 2), array[2:4, 2:4])
    stacked = np.stack([array, array])
    assert center_crop(stacked, (4, 2)).shape == (2, 4, 2)


def test_shift_moves_patch_and_mask_together():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 1
    patch = mask[None].astype(np.float32) * 7
    shifted_patch, shifted_mask = shift(patch, mask, dy=1, dx=2)
    assert shifted_mask[2, 3] == 1 and shifted_mask.sum() == 1
    assert shifted_patch[0, 2, 3] == 7 and shifted_patch.sum() == 7


def test_shift_fills_uncovered_border_with_zero():
    patch = np.ones((1, 3, 3), dtype=np.float32)
    mask = np.ones((3, 3), dtype=np.uint8)
    shifted_patch, shifted_mask = shift(patch, mask, dy=-1, dx=0)
    assert (shifted_patch[0, 2] == 0).all() and (shifted_mask[2] == 0).all()
    assert (shifted_patch[0, :2] == 1).all()
