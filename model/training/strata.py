"""Group samples by UTC hour and by day/night, to score the model per stratum.

Fire samples only exist at VIIRS overpass hours while fire-free samples span the
whole day, so a single score hides how the model behaves at each time of day.
"""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np


def solar_zenith_deg(lat, lon, when: datetime):
    """Solar zenith angle in degrees at (lat, lon) and UTC time `when`, from NOAA's
    general solar position equations (accurate to well under a degree). lat and lon
    can be numbers or numpy arrays."""
    day_of_year = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60 + when.second / 3600
    gamma = 2 * np.pi / 365 * (day_of_year - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    true_solar_minutes = hour * 60 + eqtime + 4 * lon
    hour_angle = np.radians(true_solar_minutes / 4 - 180)
    lat_rad = np.radians(lat)
    cos_zenith = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)
    return np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))


def is_day(lat, lon, when: datetime):
    """Sun above the horizon. In daylight the 3.9 um fire band also carries reflected
    sunlight, the main source of daytime false positives."""
    return solar_zenith_deg(lat, lon, when) < 90


def hour_balance_weights(manifest_path: Path, dataset_dir: Path, files: list[Path], max_weight: float = 10.0) -> list[float]:
    """Sampling weights, in the order of files, that give fire and fire-free samples
    the same total weight within each UTC hour, keeping each hour's total. At 17-18
    UTC ~92% of the samples are fire, so without this the time of day alone predicts
    fire. Hours with only one kind keep weight 1 (nothing to balance), and no sample
    weighs more than max_weight, so a handful of fires at an edge hour (07, 20 UTC)
    aren't repeated dozens of times."""
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = {dataset_dir / row["filename"]: row for row in csv.DictReader(f)}
    keys = [(datetime.fromisoformat(rows[p]["goes_time"]).hour, rows[p]["has_fire"] == "True") for p in files]
    counts: dict[tuple[int, bool], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    weights = []
    for hour, has_fire in keys:
        n_fire, n_free = counts.get((hour, True), 0), counts.get((hour, False), 0)
        if not n_fire or not n_free:
            weights.append(1.0)
        else:
            weights.append(min(max_weight, (n_fire + n_free) / (2 * (n_fire if has_fire else n_free))))
    return weights


def sample_strata(manifest_path: Path, dataset_dir: Path, files: list[Path]) -> list[tuple[str, str]]:
    """(UTC hour "HH", "day" or "night") for each file, in the order given."""
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = {dataset_dir / row["filename"]: row for row in csv.DictReader(f)}
    strata = []
    for path in files:
        row = rows[path]
        when = datetime.fromisoformat(row["goes_time"])
        lat, lon = float(row["lat"]), float(row["lon"])
        strata.append((f"{when.hour:02d}", "day" if is_day(lat, lon, when) else "night"))
    return strata
