"""CLI entrypoint: score a trained checkpoint at several segmentation thresholds.

Pick the threshold on the val split, then report it once on test - choosing it on
test would overstate the result.

Example:
    python -m model.scripts.evaluate_model --split val
    python -m model.scripts.evaluate_model --split test --thresholds 0.4
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.training.config import default_config
from model.training.dataset import FirePatchDataset
from model.training.losses import BCEDiceLoss
from model.training.model import TinyUNet
from model.training.normalization import load_band_stats
from model.training.train import evaluate

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("model/dataset"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("model/checkpoints"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    cfg = default_config(checkpoint_dir=args.checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Each checkpoint is normalized with the stats of its own training run.
    band_stats = load_band_stats(cfg.checkpoint_dir / "norm_stats.json")
    dataset = FirePatchDataset(
        args.dataset_dir / "manifest.csv", args.dataset_dir, args.split, band_stats, cfg.crop_size, augment=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
    )

    model = TinyUNet(in_channels=16, base_channels=cfg.base_channels, depth=cfg.depth).to(device)
    model.load_state_dict(torch.load(cfg.checkpoint_dir / "model_best.pt", map_location=device))
    loss_fn = BCEDiceLoss(dice_weight=cfg.dice_weight)
    logger.info("Scoring %s on %d %s samples (%s)", cfg.checkpoint_dir, len(dataset), args.split, device)

    print(f"\n{'threshold':>9}  {'precision':>9}  {'recall':>6}  {'f1':>6}  {'iou':>6}")
    for threshold in args.thresholds:
        m = evaluate(model, loader, loss_fn, device, threshold)
        print(f"{threshold:>9.2f}  {m['precision']:>9.4f}  {m['recall']:>6.4f}  {m['f1']:>6.4f}  {m['iou']:>6.4f}")


if __name__ == "__main__":
    main()
