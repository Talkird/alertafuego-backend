"""CLI entrypoint: score a trained checkpoint at several segmentation thresholds.

Pick the threshold on the val split, then report it once on test - choosing it on
test would overstate the result.

Example:
    python -m model.scripts.evaluate_model --split val
    python -m model.scripts.evaluate_model --split test --thresholds 0.4
    python -m model.scripts.evaluate_model --checkpoint-dir model/checkpoints/c32 --max-shift 0
    python -m model.scripts.evaluate_model --split test --by-stratum   # per UTC hour and day/night

f1_tol allows a 1-pixel (2 km) offset between prediction and fire (see
metrics.tolerant_counts). fire_rec is the share of fires detected and false_al the
share of fire-free patches with any prediction (metrics.fire_counts). --by-stratum
scores each UTC hour and day/night separately, at the checkpoint's threshold unless
--thresholds is given; fires/no_fire are the sample counts behind each row.
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model.training.config import THRESHOLD_CANDIDATES, default_config, load_model_config
from model.training.dataset import FirePatchDataset
from model.training.losses import BCETverskyLoss
from model.training.model import TinyUNet
from model.training.normalization import load_band_stats
from model.training.strata import sample_strata
from model.training.train import evaluate_by_stratum, evaluate_thresholds

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("model/dataset"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("model/checkpoints"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--manifest", default="manifest.csv", help="Manifest file inside --dataset-dir.")
    parser.add_argument("--thresholds", type=float, nargs="+", default=None, help="Default: all candidates.")
    parser.add_argument("--by-stratum", action="store_true", help="Score each UTC hour and day/night separately.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-shift",
        type=int,
        default=default_config().max_shift,
        help="Fixed per-sample offset of up to this many pixels, as in training (0: centered patches).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    cfg = default_config(checkpoint_dir=args.checkpoint_dir)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Each checkpoint is normalized with the stats of its own training run.
    band_stats = load_band_stats(cfg.checkpoint_dir / "norm_stats.json")
    dataset = FirePatchDataset(
        args.dataset_dir / args.manifest,
        args.dataset_dir,
        args.split,
        band_stats,
        cfg.crop_size,
        augment=False,
        seed=cfg.seed,
        max_shift=args.max_shift,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
        "pin_memory": device.type == "cuda",
    }

    model_cfg = load_model_config(cfg.checkpoint_dir)
    model = TinyUNet(in_channels=16, base_channels=model_cfg.base_channels, depth=model_cfg.depth).to(device)
    model.load_state_dict(torch.load(cfg.checkpoint_dir / "model_best.pt", map_location=device))
    loss_fn = BCETverskyLoss(tversky_weight=cfg.tversky_weight, fn_weight=cfg.fn_weight)
    logger.info("Scoring %s on %d %s samples (%s)", cfg.checkpoint_dir, len(dataset), args.split, device)

    if args.by_stratum:
        thresholds = args.thresholds or [model_cfg.threshold]
        strata = sample_strata(args.dataset_dir / args.manifest, args.dataset_dir, dataset.files)
        results = evaluate_by_stratum(model, dataset, strata, loss_fn, device, thresholds, loader_kwargs)
    else:
        loader = DataLoader(dataset, **loader_kwargs)
        results = {"all": evaluate_thresholds(model, loader, loss_fn, device, args.thresholds or THRESHOLD_CANDIDATES)}

    print(
        f"\n{'stratum':>7}  {'threshold':>9}  {'precision':>9}  {'recall':>6}  {'f1':>6}  {'iou':>6}  {'f1_tol':>6}"
        f"  {'fire_rec':>8}  {'false_al':>8}  {'f2':>6}  {'fires':>6}  {'no_fire':>7}"
    )
    for stratum, sweep in results.items():
        for m in sweep:
            print(
                f"{stratum:>7}  {m['threshold']:>9.3f}  {m['precision']:>9.4f}  {m['recall']:>6.4f}  {m['f1']:>6.4f}"
                f"  {m['iou']:>6.4f}  {m['f1_tol']:>6.4f}  {m['fire_recall']:>8.4f}  {m['false_alarm_rate']:>8.4f}"
                f"  {m['f2']:>6.4f}  {m['fires']:>6d}  {m['no_fire_samples']:>7d}"
            )


if __name__ == "__main__":
    main()
