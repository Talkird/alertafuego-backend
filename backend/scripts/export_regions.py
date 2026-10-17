"""Export the `regions` table for the frontend, which ships it as static data instead
of asking the API: the list (code, name, province) and each region's outline,
simplified for drawing (~500 m).

    python -m backend.scripts.export_regions ../alertafuego-frontend

Writes app/data/regions.json and public/regions/<code>.json under that folder. Re-run
after backend.scripts.load_regions changes the regions.
"""

import json
import sys
from pathlib import Path

from sqlalchemy import func, select

from backend.app.db import SessionLocal
from backend.app.models import Region, RegionKind

SIMPLIFY_TOLERANCE_DEG = 0.005  # ~500 m: plenty for an outline on a map
DECIMALS = 4  # ~10 m


def main(frontend: Path) -> None:
    outline = func.ST_AsGeoJSON(func.ST_SimplifyPreserveTopology(Region.geometry, SIMPLIFY_TOLERANCE_DEG), DECIMALS)
    with SessionLocal() as session:
        rows = session.execute(
            select(Region.kind, Region.code, Region.name, Region.province_code, outline).order_by(Region.code)
        ).all()
    if not rows:
        sys.exit("The regions table is empty: run backend.scripts.load_regions first.")

    regions = [
        {"code": code, "name": name, "province_code": province if kind == RegionKind.DEPARTMENT else None}
        for kind, code, name, province, _ in rows
    ]
    data = frontend / "app" / "data" / "regions.json"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps(regions, ensure_ascii=False, separators=(",", ":")) + "\n")

    outlines = frontend / "public" / "regions"
    outlines.mkdir(parents=True, exist_ok=True)
    for _, code, _, _, geojson in rows:
        (outlines / f"{code}.json").write_text(geojson + "\n")
    print(f"{len(rows)} regions -> {data} and {outlines}/")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m backend.scripts.export_regions <frontend folder>")
    main(Path(sys.argv[1]))
