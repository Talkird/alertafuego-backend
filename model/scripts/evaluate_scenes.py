"""CLI entrypoint: score a checkpoint on full Argentina scenes, as it runs in production.

Patch metrics (evaluate_model) only see 32x32 patches around VIIRS fires or random
fire-free land. Here the model runs on the whole country, through the inference
tiling, at GOES-19 images matched to VIIRS overpasses on val/test days.

Two steps:
    fetch  picks the scenes (fixed, seeded: one day and one night overpass per month
           where available, then random extras up to --scenes) and caches each one
           once from Earth Engine in --cache-dir: the 16-band raster, the FDCF fire
           Mask, and every VIIRS detection within 12 h. Re-running skips cached scenes.
    score  runs offline on the cache and writes one row per scene and threshold.

Example:
    python -m model.scripts.evaluate_scenes fetch
    python -m model.scripts.evaluate_scenes score --checkpoint-dir model/checkpoints/b-shift
    python -m model.scripts.evaluate_scenes score --checkpoint-dir model/checkpoints/b-shift --thresholds 0.08 0.65

Scoring, inside Argentina's polygon and the tiled area:
- Fires: VIIRS detections (confidence >= nominal, as in training) within 15 minutes
  of the scene. fire_recall: fire pixels with a prediction within 1 pixel.
  cluster_recall: groups of adjacent fire pixels (one fire front) with any of them
  detected, closer to what an alert needs.
- False detections: predicted pixels with no VIIRS detection of any confidence
  within 1 pixel and 12 h. false_clusters groups adjacent ones: the false alarms a
  user would see. VIIRS misses fires under clouds or between overpasses, so some of
  these may be real; the count is an upper bound.
"""

import argparse
import csv
import json
import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ee
import numpy as np
import shapely
import torch
from shapely.geometry import shape

from model.data_pipeline.config import GOES_COLLECTION_ID, default_config as default_pipeline_config
from model.data_pipeline.ee_client import init_earth_engine
from model.data_pipeline.matching import fetch_viirs_detections
from model.data_pipeline.negative_sampling import load_argentina_land_geometry
from model.data_pipeline.patches import target_projection
from model.inference.config import default_config as default_inference_config
from model.inference.predictor import load_model, predict_tiles
from model.inference.raster_fetch import compute_chunk_grid, fetch_calibrated_chunks
from model.inference.tiling import assemble_raster, tile_raster
from model.training.config import load_model_config
from model.training.normalization import load_band_stats

logger = logging.getLogger(__name__)

FDCF_COLLECTION_ID = "NOAA/GOES/19/FDCF"
#: Same window training uses to pair a VIIRS detection with a GOES image.
OVERPASS_WINDOW = timedelta(minutes=15)
#: A prediction with any VIIRS detection this close in time isn't counted as false.
CONFIRM_WINDOW = timedelta(hours=12)
SCENES_FILENAME = "scenes.csv"
POLYGON_FILENAME = "argentina.geojson"
RESULTS_FIELDNAMES = [
    "scene", "goes_time", "daynight", "detector", "threshold", "fire_pixels", "fire_pixels_hit", "fire_recall",
    "fire_clusters", "fire_clusters_hit", "cluster_recall", "predicted_pixels", "precision_tol",
    "false_pixels", "false_clusters",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fetch", "score"])
    parser.add_argument("--cache-dir", type=Path, default=Path("model/scene_eval"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("model/dataset"))
    parser.add_argument("--scenes", type=int, default=30, help="fetch: how many scenes to pick.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("model/checkpoints"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=None, help="Default: the checkpoint's.")
    parser.add_argument("--output", type=Path, default=None, help="Default: <checkpoint-dir>/scene_results.csv.")
    return parser.parse_args()


def _scene_id(goes_time: datetime) -> str:
    return goes_time.strftime("%Y%m%dT%H%M%S")


def select_scenes(manifest_path: Path, n_scenes: int, seed: int) -> list[dict]:
    """GOES times of fire samples on val/test days: one day (16-20 UTC) and one night
    (03-07 UTC) overpass per month when there is one, then random others up to n."""
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in ("val", "test") and r["has_fire"] == "True"]
    candidates = {}
    for row in rows:
        goes_time = datetime.fromisoformat(row["goes_time"])
        candidates[row["goes_time"]] = {
            "scene": _scene_id(goes_time),
            "goes_time": row["goes_time"],
            "month": row["filename"].split("/")[0],
            "split": row["split"],
            "daynight": "day" if goes_time.hour >= 12 else "night",
        }

    rng = random.Random(seed)
    pool = sorted(candidates.values(), key=lambda s: s["goes_time"])
    rng.shuffle(pool)
    chosen, seen = [], set()
    for scene in pool:
        if (scene["month"], scene["daynight"]) not in seen:
            seen.add((scene["month"], scene["daynight"]))
            chosen.append(scene)
    chosen += [s for s in pool if s not in chosen][: max(0, n_scenes - len(chosen))]
    return sorted(chosen[:n_scenes], key=lambda s: s["goes_time"])


def _image_at(collection_id: str, when: datetime, window: timedelta) -> ee.Image:
    images = ee.ImageCollection(collection_id).filterDate((when - window).isoformat(), (when + window).isoformat())
    if images.size().getInfo() == 0:
        raise LookupError(f"No {collection_id} image within {window} of {when}")
    target = int(when.timestamp() * 1000)
    with_delta = images.map(
        lambda img: img.set("delta", ee.Number(img.get("system:time_start")).subtract(target).abs())
    )
    return ee.Image(with_delta.sort("delta").first())


def _fetch_fdcf_mask(when: datetime, bbox, chunk_size_px: int) -> np.ndarray:
    """FDCF's Mask band on the same grid as the inference raster, north-up (as in
    raster_fetch.fetch_chunk_raster), -1 where there is no data."""
    image = _image_at(FDCF_COLLECTION_ID, when, timedelta(minutes=5)).select("Mask").reproject(target_projection())
    chunks = []
    for chunk_bbox in compute_chunk_grid(bbox, chunk_size_px):
        region = ee.Geometry.BBox(chunk_bbox.west, chunk_bbox.south, chunk_bbox.east, chunk_bbox.north)
        properties = image.sampleRectangle(region=region, defaultValue=-1).getInfo()["properties"]
        chunks.append((chunk_bbox, np.array(properties["Mask"], dtype=np.float32)[None, ::-1, :]))
    return assemble_raster(chunks, bbox)[0].astype(np.int16)


def fetch(args: argparse.Namespace) -> None:
    init_earth_engine()
    ee.data.setDeadline(2 * 60 * 1000)  # see export_dataset: a hung request otherwise waits forever
    cfg = default_inference_config()
    bbox = cfg.argentina_bbox
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    scenes_path = args.cache_dir / SCENES_FILENAME
    if not scenes_path.exists():
        scenes = select_scenes(args.dataset_dir / "manifest.csv", args.scenes, args.seed)
        with scenes_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(scenes[0]))
            writer.writeheader()
            writer.writerows(scenes)
    with scenes_path.open(newline="", encoding="utf-8") as f:
        scenes = list(csv.DictReader(f))

    polygon_path = args.cache_dir / POLYGON_FILENAME
    if not polygon_path.exists():
        polygon_path.write_text(json.dumps(load_argentina_land_geometry(bbox).getInfo()))

    for i, scene in enumerate(scenes, 1):
        raster_path = args.cache_dir / f"{scene['scene']}.npz"
        if raster_path.exists():
            continue
        when = datetime.fromisoformat(scene["goes_time"])
        logger.info("Scene %d/%d: %s", i, len(scenes), scene["scene"])

        goes_image = _image_at(GOES_COLLECTION_ID, when, timedelta(minutes=1))
        raster = assemble_raster(fetch_calibrated_chunks(goes_image, bbox, cfg.chunk_size_px), bbox)
        fdcf = _fetch_fdcf_mask(when, bbox, cfg.chunk_size_px)

        # VIIRS collections are daily images: query whole days, keep +-12 h.
        start = (when - CONFIRM_WINDOW).replace(hour=0, minute=0, second=0, microsecond=0)
        detections = fetch_viirs_detections(bbox, start, when + CONFIRM_WINDOW + timedelta(days=1), min_confidence=0)
        with (args.cache_dir / f"{scene['scene']}_viirs.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["lat", "lon", "timestamp", "confidence", "frp"])
            for d in detections:
                if abs(d.timestamp - when) <= CONFIRM_WINDOW:
                    writer.writerow([d.lat, d.lon, d.timestamp.isoformat(), d.confidence, d.frp])

        # Written last, so a scene interrupted mid-fetch is fetched again.
        np.savez_compressed(raster_path, raster=raster, fdcf=fdcf)


def _dilate(mask: np.ndarray) -> np.ndarray:
    """Pixels within 1 pixel (8 neighbors) of a True pixel."""
    padded = np.pad(mask, 1)
    height, width = mask.shape
    return np.max([padded[dy : dy + height, dx : dx + width] for dy in range(3) for dx in range(3)], axis=0)


def clusters(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Groups of 8-connected True pixels. The masks here are sparse, so a search over
    the True pixels is enough."""
    remaining = set(zip(*np.nonzero(mask)))
    groups = []
    while remaining:
        stack = [remaining.pop()]
        group = []
        while stack:
            row, col = stack.pop()
            group.append((row, col))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (row + dy, col + dx)
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        groups.append(group)
    return groups


def _viirs_mask(rows: list[dict], shape_hw: tuple[int, int], bbox) -> np.ndarray:
    """Rasterize detections to the scene grid (inverse of tiling.pixel_to_latlon)."""
    height, width = shape_hw
    mask = np.zeros(shape_hw, dtype=bool)
    for row in rows:
        r = int((bbox.north - float(row["lat"])) / (bbox.north - bbox.south) * height)
        c = int((float(row["lon"]) - bbox.west) / (bbox.east - bbox.west) * width)
        if 0 <= r < height and 0 <= c < width:
            mask[r, c] = True
    return mask


def score_detection(predicted: np.ndarray, fires: np.ndarray, confirmed: np.ndarray) -> dict:
    """predicted, fires (overpass VIIRS) and confirmed (any VIIRS within 12 h) are
    boolean masks already restricted to the scored area."""
    near_predicted = _dilate(predicted)
    false_pixels = predicted & ~_dilate(confirmed)
    fire_groups = clusters(fires)
    hit_groups = sum(any(near_predicted[r, c] for r, c in group) for group in fire_groups)
    n_fire, n_pred = int(fires.sum()), int(predicted.sum())
    n_hit = int((fires & near_predicted).sum())
    return {
        "fire_pixels": n_fire,
        "fire_pixels_hit": n_hit,
        "fire_recall": n_hit / n_fire if n_fire else "",
        "fire_clusters": len(fire_groups),
        "fire_clusters_hit": hit_groups,
        "cluster_recall": hit_groups / len(fire_groups) if fire_groups else "",
        "predicted_pixels": n_pred,
        "precision_tol": (n_pred - int(false_pixels.sum())) / n_pred if n_pred else "",
        "false_pixels": int(false_pixels.sum()),
        "false_clusters": len(clusters(false_pixels)),
    }


def score(args: argparse.Namespace) -> None:
    cfg = default_inference_config(checkpoint_dir=args.checkpoint_dir)
    bbox = cfg.argentina_bbox
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = load_model(args.checkpoint_dir / "model_best.pt", device)
    band_stats = load_band_stats(args.checkpoint_dir / "norm_stats.json")
    thresholds = args.thresholds or [load_model_config(args.checkpoint_dir).threshold]
    polygon = shape(json.loads((args.cache_dir / POLYGON_FILENAME).read_text()))
    with (args.cache_dir / SCENES_FILENAME).open(newline="", encoding="utf-8") as f:
        scenes = list(csv.DictReader(f))

    output = args.output or args.checkpoint_dir / "scene_results.csv"
    results, scored_area = [], None
    for scene in scenes:
        raster_path = args.cache_dir / f"{scene['scene']}.npz"
        if not raster_path.exists():
            logger.warning("Scene %s is not cached, skipped (run fetch)", scene["scene"])
            continue
        raster = np.load(raster_path)["raster"]
        _, height, width = raster.shape

        if scored_area is None:
            # Argentina's polygon, within the center crop tile_raster() keeps. Every
            # scene shares the grid, so this is computed once.
            lat = bbox.north - (np.arange(height) + 0.5) / height * (bbox.north - bbox.south)
            lon = bbox.west + (np.arange(width) + 0.5) / width * (bbox.east - bbox.west)
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            scored_area = shapely.contains_xy(polygon, lon_grid, lat_grid)
            size = cfg.patch_size_px
            crop_top, crop_left = (height - height // size * size) // 2, (width - width // size * size) // 2
            in_crop = np.zeros_like(scored_area)
            in_crop[crop_top : crop_top + height // size * size, crop_left : crop_left + width // size * size] = True
            scored_area &= in_crop

        tiles, offsets = tile_raster(raster, cfg.patch_size_px)
        probabilities = np.zeros((height, width), dtype=np.float32)
        for (row, col), tile_probs in zip(offsets, predict_tiles(model, band_stats, tiles, device)):
            probabilities[crop_top + row : crop_top + row + size, crop_left + col : crop_left + col + size] = tile_probs

        when = datetime.fromisoformat(scene["goes_time"])
        with (args.cache_dir / f"{scene['scene']}_viirs.csv").open(newline="", encoding="utf-8") as f:
            viirs = list(csv.DictReader(f))
        overpass = [
            v for v in viirs
            if int(v["confidence"]) >= 1 and abs(datetime.fromisoformat(v["timestamp"]) - when) <= OVERPASS_WINDOW
        ]
        fires = _viirs_mask(overpass, (height, width), bbox) & scored_area
        confirmed = _viirs_mask(viirs, (height, width), bbox)

        for threshold in thresholds:
            predicted = (probabilities >= threshold) & scored_area
            row = {"scene": scene["scene"], "goes_time": scene["goes_time"], "daynight": scene["daynight"],
                   "detector": "model", "threshold": threshold}
            results.append(row | score_detection(predicted, fires, confirmed))
        logger.info("Scored %s: %d fire pixels", scene["scene"], int(fires.sum()))

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Wrote %s", output)
    _log_summary(results)


def _log_summary(results: list[dict]) -> None:
    """Totals over scenes, per detector, threshold and day/night."""
    groups: dict[tuple, list[dict]] = {}
    for row in results:
        for daynight in ("all", row["daynight"]):
            groups.setdefault((row["detector"], row["threshold"], daynight), []).append(row)
    print(f"\n{'detector':>10}  {'threshold':>9}  {'scenes':>6}  {'fire_rec':>8}  {'clust_rec':>9}"
          f"  {'prec_tol':>8}  {'false_px/scene':>14}  {'false_clusters/scene':>20}")
    for (detector, threshold, daynight), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        total = {k: sum(r[k] for r in rows) for k in
                 ("fire_pixels", "fire_pixels_hit", "fire_clusters", "fire_clusters_hit", "predicted_pixels",
                  "false_pixels", "false_clusters")}
        n = len(rows)
        print(
            f"{detector + ' ' + daynight:>10}  {threshold:>9}  {n:>6}"
            f"  {total['fire_pixels_hit'] / max(1, total['fire_pixels']):>8.4f}"
            f"  {total['fire_clusters_hit'] / max(1, total['fire_clusters']):>9.4f}"
            f"  {(total['predicted_pixels'] - total['false_pixels']) / max(1, total['predicted_pixels']):>8.4f}"
            f"  {total['false_pixels'] / n:>14.1f}  {total['false_clusters'] / n:>20.1f}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    if args.command == "fetch":
        fetch(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
