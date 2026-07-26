"""CLI entrypoint: train the wildfire segmentation model on the exported dataset.

Example:
    python -m model.scripts.train_model --limit 200 --epochs 2
"""

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader

from model.training.config import default_config
from model.training.dataset import FirePatchDataset, list_split_files
from model.training.losses import BCEDiceLoss
from model.training.model import TinyUNet
from model.training.normalization import compute_band_stats, save_band_stats
from model.training.train import evaluate, fit, plot_training_curves

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("model/dataset"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("model/checkpoints"))
    parser.add_argument("--epochs", type=int, default=None, help="Overrides the default epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Overrides the default batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Overrides the default learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="Overrides the default seed.")
    parser.add_argument("--limit", type=int, default=None, help="Cap samples per split, for smoke tests.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    overrides = {
        key: value
        for key, value in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "seed": args.seed,
        }.items()
        if value is not None
    }
    cfg = replace(default_config(checkpoint_dir=args.checkpoint_dir), **overrides)

    manifest_path = args.dataset_dir / "manifest.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    train_files = list_split_files(manifest_path, args.dataset_dir, "train", limit=args.limit, seed=cfg.seed)
    band_stats = compute_band_stats(train_files)
    save_band_stats(band_stats, cfg.checkpoint_dir / "norm_stats.json")

    train_ds = FirePatchDataset(
        manifest_path, args.dataset_dir, "train", band_stats, cfg.crop_size, augment=True, limit=args.limit, seed=cfg.seed
    )
    val_ds = FirePatchDataset(
        manifest_path, args.dataset_dir, "val", band_stats, cfg.crop_size, augment=False, limit=args.limit, seed=cfg.seed
    )
    test_ds = FirePatchDataset(
        manifest_path, args.dataset_dir, "test", band_stats, cfg.crop_size, augment=False, limit=args.limit, seed=cfg.seed
    )
    logger.info("Samples - train: %d, val: %d, test: %d", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = TinyUNet(in_channels=16, base_channels=cfg.base_channels, depth=cfg.depth).to(device)
    loss_fn = BCEDiceLoss(dice_weight=cfg.dice_weight)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_model_path = fit(model, train_loader, val_loader, cfg, optimizer, loss_fn, device)

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_metrics = evaluate(model, test_loader, loss_fn, device, cfg.seg_threshold)
    logger.info("Test metrics: %s", test_metrics)

    plot_training_curves(cfg.checkpoint_dir / "metrics.csv", cfg.checkpoint_dir / "training_curves.png")


if __name__ == "__main__":
    main()
