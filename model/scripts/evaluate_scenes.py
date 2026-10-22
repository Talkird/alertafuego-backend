"""CLI entrypoint: score a checkpoint on full Argentina scenes, as it runs in production.

Patch metrics (evaluate_model) only see 32x32 patches around VIIRS fires or random
fire-free land. Here the model runs on the whole country, through the inference
code (predictor.predict_raster), at GOES-19 images matched to VIIRS overpasses on
val/test days.

Two steps:
    fetch  picks the scenes (fixed, seeded: one day and one night overpass per month
           where available, then random extras up to --scenes) and caches each one
           once from Earth Engine in --cache-dir: the 16-band raster, the FDCF fire
           Mask, and every VIIRS detection within 12 h. Re-running skips cached scenes.
    calibrate  offline: picks the day and the night threshold with the best F2 on the
           val scenes and saves them in the checkpoint's model_config.json.
    score  runs offline on the cache and writes one row per scene and threshold, for
           the model, NOAA's FDCF fire product (no threshold), and model OR FDCF.
           Without --thresholds, the checkpoint's own (day/night) thresholds.

Example:
    python -m model.scripts.evaluate_scenes fetch
    python -m model.scripts.evaluate_scenes calibrate --checkpoint-dir model/checkpoints/b-shift
    python -m model.scripts.evaluate_scenes score --checkpoint-dir model/checkpoints/b-shift --split test
    python -m model.scripts.evaluate_scenes score --checkpoint-dir model/checkpoints/b-shift --thresholds 0.08 0.65

--split val or test scores only the scenes from that split's days: choose settings
on val scenes, report test scenes.

Scoring, inside Argentina's polygon:
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
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ee
import numpy as np
import shapely
import torch
from shapely.geometry import shape

from model.data_pipeline.config import GOES_COLLECTION_ID, VIIRS_COLLECTION_IDS, BBox
from model.data_pipeline.ee_client import init_earth_engine
from model.data_pipeline.matching import MAX_VIIRS_FEATURES_PER_QUERY, _fire_points_for_image
from model.data_pipeline.negative_sampling import load_argentina_land_geometry
from model.data_pipeline.patches import target_projection
from model.inference.config import default_config as default_inference_config
from model.inference.predictor import load_model, predict_raster
from model.inference.raster_fetch import compute_chunk_grid, fetch_calibrated_chunks
from model.inference.tiling import assemble_raster
from model.training.config import THRESHOLD_CANDIDATES, load_model_config, save_model_config
from model.training.metrics import f_beta
from model.training.normalization import load_band_stats

logger = logging.getLogger(__name__)

FDCF_COLLECTION_ID = "NOAA/GOES/19/FDCF"
#: FDCF Mask codes for a fire pixel: 10-15 (processed, saturated, cloud contaminated,
#: high, medium and low probability) and 30-35 (the same, temporally filtered).
FDCF_FIRE_CODES = (10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35)
#: Same window training uses to pair a VIIRS detection with a GOES image.
OVERPASS_WINDOW = timedelta(minutes=15)
#: A prediction with any VIIRS detection this close in time isn't counted as false.
CONFIRM_WINDOW = timedelta(hours=12)
#: VIIRS is queried in slices this long, filtered by acquisition time on the server,
#: so each query holds about one overpass per satellite and stays under the feature cap.
VIIRS_SLICE = timedelta(hours=3)
#: On whole scenes the best thresholds sit far below the patch ones (b-shift: ~0.001),
#: so calibration also tries these.
SCENE_THRESHOLD_CANDIDATES = [0.00005, 0.0001, 0.0002, 0.0005] + THRESHOLD_CANDIDATES
SCENES_FILENAME = "scenes.csv"
POLYGON_FILENAME = "argentina.geojson"
RESULTS_FIELDNAMES = [
    "scene", "goes_time", "daynight", "detector", "threshold", "fire_pixels", "fire_pixels_hit", "fire_recall",
    "fire_clusters", "fire_clusters_hit", "cluster_recall", "predicted_pixels", "precision_tol",
    "false_pixels", "false_clusters",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["fetch", "calibrate", "score"])
    parser.add_argument("--cache-dir", type=Path, default=Path("model/scene_eval"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("model/dataset"))
    parser.add_argument("--scenes", type=int, default=30, help="fetch: how many scenes to pick.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("model/checkpoints"))
    parser.add_argument("--thresholds", type=float, nargs="+", default=None, help="Default: the checkpoint's.")
    parser.add_argument("--output", type=Path, default=None, help="Default: <checkpoint-dir>/scene_results.csv.")
    parser.add_argument("--split", choices=["val", "test"], default=None, help="score: only this split's scenes.")
    parser.add_argument("--tta", action="store_true", help="score: average over the 8 flips and rotations.")
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


def _viirs_features(collection_id: str, bbox: BBox, start: datetime, end: datetime, depth: int = 0) -> list[dict]:
    """VIIRS fire points of one collection in bbox acquired in [start, end), filtered
    by acquisition time on the server. A query that hits the feature cap is split
    into four quarters of bbox, down to 3 levels (1/64 of the area)."""
    region = ee.Geometry.BBox(bbox.west, bbox.south, bbox.east, bbox.north)
    # Daily images: a detection sits in its own UTC day's image, give or take one.
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    images = ee.ImageCollection(collection_id).filterDate(
        (day - timedelta(days=1)).isoformat(), (end + timedelta(days=1)).isoformat()
    ).filterBounds(region)
    points = ee.FeatureCollection(
        images.toList(images.size()).map(lambda img: _fire_points_for_image(ee.Image(img), region, 0))
    ).flatten()
    points = points.filter(ee.Filter.rangeContains("acq_epoch", start.timestamp(), end.timestamp() - 1e-3))
    features = points.limit(MAX_VIIRS_FEATURES_PER_QUERY).getInfo()["features"]
    if len(features) < MAX_VIIRS_FEATURES_PER_QUERY:
        return features
    if depth == 3:
        logger.warning("%s hit the feature cap for %s, %s..%s; some detections are missing", collection_id, bbox, start, end)
        return features
    mid_lon, mid_lat = (bbox.west + bbox.east) / 2, (bbox.south + bbox.north) / 2
    # Half-open quarters would need exclusive edges; points exactly on a split line
    # may appear twice, which only duplicates a mask pixel.
    quarters = [
        BBox(west=w, south=s, east=e, north=n)
        for w, e in ((bbox.west, mid_lon), (mid_lon, bbox.east))
        for s, n in ((bbox.south, mid_lat), (mid_lat, bbox.north))
    ]
    return [f for q in quarters for f in _viirs_features(collection_id, q, start, end, depth + 1)]


def _fetch_viirs(bbox: BBox, start: datetime, end: datetime) -> list[dict]:
    """Every VIIRS detection (any confidence) acquired in [start, end), queried in
    VIIRS_SLICE steps. matching.fetch_viirs_detections caps whole days instead, which
    on busy days drops detections in collection order."""
    rows = []
    slice_start = start
    while slice_start < end:
        slice_end = min(end, slice_start + VIIRS_SLICE)
        for collection_id in VIIRS_COLLECTION_IDS:
            for feature in _viirs_features(collection_id, bbox, slice_start, slice_end):
                lon, lat = feature["geometry"]["coordinates"]
                props = feature["properties"]
                rows.append({
                    "lat": lat,
                    "lon": lon,
                    "timestamp": datetime.fromtimestamp(props["acq_epoch"], tz=timezone.utc).isoformat(),
                    "confidence": int(props["confidence"]),
                    "frp": float(props.get("frp", 0.0)),
                })
        slice_start = slice_end
    return rows


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
        when = datetime.fromisoformat(scene["goes_time"])
        # Each file is written whole once fetched, so an interrupted scene resumes
        # with whatever it is missing.
        raster_path = args.cache_dir / f"{scene['scene']}.npz"
        if not raster_path.exists():
            logger.info("Scene %d/%d: %s, raster and FDCF", i, len(scenes), scene["scene"])
            goes_image = _image_at(GOES_COLLECTION_ID, when, timedelta(minutes=1))
            raster = assemble_raster(fetch_calibrated_chunks(goes_image, bbox, cfg.chunk_size_px), bbox)
            fdcf = _fetch_fdcf_mask(when, bbox, cfg.chunk_size_px)
            np.savez_compressed(raster_path, raster=raster, fdcf=fdcf)

        viirs_path = args.cache_dir / f"{scene['scene']}_viirs.csv"
        if not viirs_path.exists():
            logger.info("Scene %d/%d: %s, VIIRS", i, len(scenes), scene["scene"])
            rows = _fetch_viirs(bbox, when - CONFIRM_WINDOW, when + CONFIRM_WINDOW)
            with viirs_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["lat", "lon", "timestamp", "confidence", "frp"])
                writer.writeheader()
                writer.writerows(rows)


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


def _load_scenes(args: argparse.Namespace, split: str | None) -> list[dict]:
    with (args.cache_dir / SCENES_FILENAME).open(newline="", encoding="utf-8") as f:
        return [s for s in csv.DictReader(f) if split in (None, s["split"])]


def _score_scenes(args: argparse.Namespace, scenes: list[dict], thresholds: list[float] | None) -> list[dict]:
    """Rows of score_detection results per scene for FDCF, and for the model and model
    OR FDCF at each threshold. thresholds None: the checkpoint's own (threshold, or
    night_threshold on night scenes), labeled "config"."""
    cfg = default_inference_config(checkpoint_dir=args.checkpoint_dir)
    bbox = cfg.argentina_bbox
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = load_model(args.checkpoint_dir / "model_best.pt", device)
    band_stats = load_band_stats(args.checkpoint_dir / "norm_stats.json")
    model_cfg = load_model_config(args.checkpoint_dir)
    polygon = shape(json.loads((args.cache_dir / POLYGON_FILENAME).read_text()))

    results, scored_area = [], None
    for scene in scenes:
        raster_path = args.cache_dir / f"{scene['scene']}.npz"
        if not raster_path.exists():
            logger.warning("Scene %s is not cached, skipped (run fetch)", scene["scene"])
            continue
        cached = np.load(raster_path)
        raster = cached["raster"]
        _, height, width = raster.shape

        if scored_area is None:
            # Argentina's polygon. Every scene shares the grid, so this is computed once.
            lat = bbox.north - (np.arange(height) + 0.5) / height * (bbox.north - bbox.south)
            lon = bbox.west + (np.arange(width) + 0.5) / width * (bbox.east - bbox.west)
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            scored_area = shapely.contains_xy(polygon, lon_grid, lat_grid)

        probabilities = predict_raster(model, band_stats, raster, device, tta=args.tta)

        when = datetime.fromisoformat(scene["goes_time"])
        with (args.cache_dir / f"{scene['scene']}_viirs.csv").open(newline="", encoding="utf-8") as f:
            viirs = list(csv.DictReader(f))
        overpass = [
            v for v in viirs
            if int(v["confidence"]) >= 1 and abs(datetime.fromisoformat(v["timestamp"]) - when) <= OVERPASS_WINDOW
        ]
        fires = _viirs_mask(overpass, (height, width), bbox) & scored_area
        confirmed = _viirs_mask(viirs, (height, width), bbox)

        scene_row = {"scene": scene["scene"], "goes_time": scene["goes_time"], "daynight": scene["daynight"]}
        fdcf = np.isin(cached["fdcf"], FDCF_FIRE_CODES) & scored_area
        results.append(scene_row | {"detector": "fdcf", "threshold": ""} | score_detection(fdcf, fires, confirmed))
        if thresholds is None:
            night = scene["daynight"] == "night" and model_cfg.night_threshold is not None
            scene_thresholds = [("config", model_cfg.night_threshold if night else model_cfg.threshold)]
        else:
            scene_thresholds = [(threshold, threshold) for threshold in thresholds]
        for label, threshold in scene_thresholds:
            predicted = (probabilities >= threshold) & scored_area
            results.append(
                scene_row | {"detector": "model", "threshold": label} | score_detection(predicted, fires, confirmed)
            )
            results.append(
                scene_row
                | {"detector": "model|fdcf", "threshold": label}
                | score_detection(predicted | fdcf, fires, confirmed)
            )
        logger.info("Scored %s: %d fire pixels", scene["scene"], int(fires.sum()))
    return results


def score(args: argparse.Namespace) -> None:
    results = _score_scenes(args, _load_scenes(args, args.split), args.thresholds)
    output = args.output or args.checkpoint_dir / "scene_results.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Wrote %s", output)
    _log_summary(results)


def _totals(rows: list[dict]) -> dict:
    total = {k: sum(r[k] for r in rows) for k in
             ("fire_pixels", "fire_pixels_hit", "fire_clusters", "fire_clusters_hit", "predicted_pixels",
              "false_pixels", "false_clusters")}
    total["fire_recall"] = total["fire_pixels_hit"] / max(1, total["fire_pixels"])
    total["precision_tol"] = (total["predicted_pixels"] - total["false_pixels"]) / max(1, total["predicted_pixels"])
    total["f2"] = f_beta(total["precision_tol"], total["fire_recall"], 2)
    return total


def calibrate(args: argparse.Namespace) -> None:
    """Pick the day and the night threshold with the best F2 (fire pixel recall
    weighted twice as much as precision) on the val scenes, and save them in the
    checkpoint's model_config.json. Report test scenes afterwards with score."""
    results = [
        r for r in _score_scenes(args, _load_scenes(args, "val"), SCENE_THRESHOLD_CANDIDATES) if r["detector"] == "model"
    ]
    chosen = {}
    print(f"\n{'scenes':>6}  {'threshold':>9}  {'fire_rec':>8}  {'prec_tol':>8}  {'f2':>6}")
    for daynight in ("day", "night"):
        sweep = []
        for threshold in SCENE_THRESHOLD_CANDIDATES:
            total = _totals([r for r in results if r["daynight"] == daynight and r["threshold"] == threshold])
            sweep.append((total["f2"], threshold))
            print(f"{daynight:>6}  {threshold:>9}  {total['fire_recall']:>8.4f}  {total['precision_tol']:>8.4f}  {total['f2']:>6.4f}")
        chosen[daynight] = max(sweep)[1]
        if chosen[daynight] == SCENE_THRESHOLD_CANDIDATES[0]:
            logger.warning("%s: best F2 at the lowest candidate %s; a lower one may be better", daynight, chosen[daynight])
    model_cfg = load_model_config(args.checkpoint_dir)
    save_model_config(
        replace(model_cfg, threshold=chosen["day"], night_threshold=chosen["night"]), args.checkpoint_dir
    )
    logger.info("Saved thresholds day %s, night %s in %s", chosen["day"], chosen["night"], args.checkpoint_dir)


def _log_summary(results: list[dict]) -> None:
    """Totals over scenes, per detector, threshold and day/night."""
    groups: dict[tuple, list[dict]] = {}
    for row in results:
        for daynight in ("all", row["daynight"]):
            groups.setdefault((row["detector"], row["threshold"], daynight), []).append(row)
    print(f"\n{'detector':>16}  {'threshold':>9}  {'scenes':>6}  {'fire_rec':>8}  {'clust_rec':>9}"
          f"  {'prec_tol':>8}  {'false_px/scene':>14}  {'false_clusters/scene':>20}")
    for (detector, threshold, daynight), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        total = _totals(rows)
        n = len(rows)
        print(
            f"{detector + ' ' + daynight:>16}  {threshold:>9}  {n:>6}"
            f"  {total['fire_pixels_hit'] / max(1, total['fire_pixels']):>8.4f}"
            f"  {total['fire_clusters_hit'] / max(1, total['fire_clusters']):>9.4f}"
            f"  {(total['predicted_pixels'] - total['false_pixels']) / max(1, total['predicted_pixels']):>8.4f}"
            f"  {total['false_pixels'] / n:>14.1f}  {total['false_clusters'] / n:>20.1f}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    {"fetch": fetch, "calibrate": calibrate, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
