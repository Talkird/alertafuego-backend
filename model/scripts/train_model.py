"""CLI entrypoint: train the wildfire segmentation model on the exported dataset.

Example:
    python -m model.scripts.train_model --limit 200 --epochs 2
    python -m model.scripts.train_model --batch-size 256 --num-workers 6   # on a GPU
    python -m model.scripts.train_model --base-channels 32 --checkpoint-dir model/checkpoints/c32

After training, the highest threshold whose val fire recall reaches the recall target
(default 0.95; the lowest-threshold run otherwise) is saved with the checkpoint
(model_config.json, read by inference) and test is scored once at it. results.json
sums up the run.
"""

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from model.training.config import THRESHOLD_CANDIDATES, ModelConfig, default_config, pick_threshold, save_model_config
from model.training.dataset import FirePatchDataset, list_split_files
from model.training.losses import BCETverskyLoss
from model.training.model import TinyUNet
from model.training.normalization import compute_band_stats, save_band_stats
from model.training.strata import hour_balance_weights, sample_strata
from model.training.train import evaluate, evaluate_by_stratum, evaluate_thresholds, fit, plot_training_curves

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("model/dataset"))
    parser.add_argument(
        "--manifest", default="manifest.csv", help="Manifest file inside --dataset-dir (see merge_datasets)."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("model/checkpoints"))
    parser.add_argument("--epochs", type=int, default=None, help="Overrides the default epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Overrides the default batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Overrides the default learning rate.")
    parser.add_argument("--seed", type=int, default=None, help="Overrides the default seed.")
    parser.add_argument(
        "--num-workers", type=int, default=None, help="Processes loading samples in parallel (default 0: main process)."
    )
    parser.add_argument("--base-channels", type=int, default=None, help="Model width (default 16; 32 or 64 on a GPU).")
    parser.add_argument("--depth", type=int, default=None, help="Downsampling stages (default 3).")
    parser.add_argument(
        "--max-shift", type=int, default=None, help="Max pixels patches are shifted off-center (default 8; 0: none)."
    )
    parser.add_argument("--crop-size", type=int, default=None, help="Patch side the model trains on (default 32).")
    parser.add_argument(
        "--random-crop",
        action="store_true",
        help="Crop at a random position in the stored patch instead of shifting (use with --max-shift 0).",
    )
    parser.add_argument(
        "--balance-hours", action="store_true", help="Sample fire and fire-free train patches evenly within each UTC hour."
    )
    parser.add_argument(
        "--fn-weight", type=float, default=None, help="Tversky weight of a missed fire pixel (default 0.7; 0.5: Dice)."
    )
    parser.add_argument(
        "--recall-target", type=float, default=None, help="Val fire recall the threshold must reach (default 0.95)."
    )
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
            "num_workers": args.num_workers,
            "base_channels": args.base_channels,
            "depth": args.depth,
            "max_shift": args.max_shift,
            "crop_size": args.crop_size,
            "random_crop": args.random_crop or None,
            "balance_hours": args.balance_hours or None,
            "fn_weight": args.fn_weight,
            "recall_target": args.recall_target,
        }.items()
        if value is not None
    }
    cfg = replace(default_config(checkpoint_dir=args.checkpoint_dir), **overrides)

    manifest_path = args.dataset_dir / args.manifest
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    logger.info("Training on device: %s", device)

    train_files = list_split_files(manifest_path, args.dataset_dir, "train", limit=args.limit, seed=cfg.seed)
    band_stats = compute_band_stats(train_files)
    save_band_stats(band_stats, cfg.checkpoint_dir / "norm_stats.json")

    def dataset(split: str, augment: bool, max_shift: int, random_crop: bool = cfg.random_crop) -> FirePatchDataset:
        return FirePatchDataset(
            manifest_path,
            args.dataset_dir,
            split,
            band_stats,
            cfg.crop_size,
            augment=augment,
            limit=args.limit,
            seed=cfg.seed,
            max_shift=max_shift,
            random_crop=random_crop,
        )

    # Val and test are shifted (or randomly cropped) too, fixed per sample, so scores
    # reflect fires anywhere in a tile, as at inference. Centered test: comparable
    # with runs trained before shifting existed.
    train_ds = dataset("train", augment=True, max_shift=cfg.max_shift)
    val_ds = dataset("val", augment=False, max_shift=cfg.max_shift)
    test_ds = dataset("test", augment=False, max_shift=cfg.max_shift)
    test_centered_ds = dataset("test", augment=False, max_shift=0, random_crop=False)
    logger.info("Samples - train: %d, val: %d, test: %d", len(train_ds), len(val_ds), len(test_ds))

    # Workers are kept between epochs: on Windows each one starts a new Python process.
    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "persistent_workers": cfg.num_workers > 0,
        "pin_memory": device.type == "cuda",
    }
    if cfg.balance_hours:
        weights = hour_balance_weights(manifest_path, args.dataset_dir, train_ds.files)
        sampler = WeightedRandomSampler(
            weights, len(weights), replacement=True, generator=torch.Generator().manual_seed(cfg.seed)
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    test_centered_loader = DataLoader(test_centered_ds, shuffle=False, **loader_kwargs)

    model = TinyUNet(in_channels=16, base_channels=cfg.base_channels, depth=cfg.depth).to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))
    loss_fn = BCETverskyLoss(tversky_weight=cfg.tversky_weight, fn_weight=cfg.fn_weight)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_model_path = fit(model, train_loader, val_loader, cfg, optimizer, loss_fn, device)

    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # The threshold is picked on val; test is scored once, at it - choosing it on
    # test would overstate the result.
    val_sweep = evaluate_thresholds(model, val_loader, loss_fn, device, THRESHOLD_CANDIDATES)
    best = pick_threshold(val_sweep, cfg.recall_target)
    threshold = best["threshold"]
    logger.info(
        "Val threshold %.3f: fire_recall=%.4f (target %.2f) precision_tol=%.4f f1_tol=%.4f",
        threshold, best["fire_recall"], cfg.recall_target, best["precision_tol"], best["f1_tol"],
    )
    save_model_config(ModelConfig(cfg.base_channels, cfg.depth, threshold), cfg.checkpoint_dir)

    test_metrics = evaluate(model, test_loader, loss_fn, device, threshold)
    test_centered_metrics = evaluate(model, test_centered_loader, loss_fn, device, threshold)
    logger.info("Test metrics: %s", test_metrics)
    logger.info("Test metrics, centered patches: %s", test_centered_metrics)
    test_strata = sample_strata(manifest_path, args.dataset_dir, test_ds.files)
    test_by_stratum = {
        label: sweep[0]
        for label, sweep in evaluate_by_stratum(
            model, test_ds, test_strata, loss_fn, device, [threshold], loader_kwargs
        ).items()
    }

    results = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(cfg).items()},
        "parameters": sum(p.numel() for p in model.parameters()),
        "threshold": threshold,
        "val_sweep": val_sweep,
        "test": test_metrics,
        "test_centered": test_centered_metrics,
        "test_by_stratum": test_by_stratum,
    }
    (cfg.checkpoint_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    plot_training_curves(cfg.checkpoint_dir / "metrics.csv", cfg.checkpoint_dir / "training_curves.png")


if __name__ == "__main__":
    main()
