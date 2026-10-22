"""Group samples by UTC hour and by day/night, to score the model per stratum.

Fire samples only exist at VIIRS overpass hours while fire-free samples span the
whole day, so a single score hides how the model behaves at each time of day.
"""

import csv
import math
from datetime import datetime
from pathlib import Path


def solar_zenith_deg(lat: float, lon: float, when: datetime) -> float:
    """Solar zenith angle in degrees at (lat, lon) and UTC time `when`, from NOAA's
    general solar position equations (accurate to well under a degree)."""
    day_of_year = when.timetuple().tm_yday
    hour = when.hour + when.minute / 60 + when.second / 3600
    gamma = 2 * math.pi / 365 * (day_of_year - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    true_solar_minutes = hour * 60 + eqtime + 4 * lon
    hour_angle = math.radians(true_solar_minutes / 4 - 180)
    lat_rad = math.radians(lat)
    cos_zenith = math.sin(lat_rad) * math.sin(decl) + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_zenith))))


def is_day(lat: float, lon: float, when: datetime) -> bool:
    """Sun above the horizon. In daylight the 3.9 um fire band also carries reflected
    sunlight, the main source of daytime false positives."""
    return solar_zenith_deg(lat, lon, when) < 90


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
