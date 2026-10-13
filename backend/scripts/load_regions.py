"""Load Argentina's provinces and departments from IGN into the `regions` table,
for per-region statistics (GET /stats/regions). Re-running replaces what's there.

    python -m backend.scripts.load_regions

Downloads ~250 MB of GeoJSON from IGN's WFS and simplifies it before storing:
GOES-19 pixels are ~2 km, so ~100 m of boundary detail is plenty.
"""

import json
import urllib.request

from geoalchemy2.shape import from_shape
from shapely.geometry import MultiPolygon, shape
from sqlalchemy import delete, insert

from backend.app.db import SessionLocal
from backend.app.models import Region, RegionKind

WFS_URL = (
    "https://wms.ign.gob.ar/geoserver/ign/ows?service=WFS&version=1.0.0&request=GetFeature"
    "&typeName=ign:{layer}&outputFormat=application/json"
)
LAYERS = {RegionKind.PROVINCE: "provincia", RegionKind.DEPARTMENT: "departamento"}
SIMPLIFY_TOLERANCE_DEG = 0.001  # ~100 m


def _fetch(layer: str) -> list[dict]:
    print(f"Downloading ign:{layer}...")
    with urllib.request.urlopen(WFS_URL.format(layer=layer), timeout=600) as response:
        return json.load(response)["features"]


def _row(kind: RegionKind, feature: dict) -> dict:
    props = feature["properties"]
    geometry = shape(feature["geometry"]).simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
    if not isinstance(geometry, MultiPolygon):
        geometry = MultiPolygon([geometry])
    code = props["in1"]
    return {
        "kind": kind,
        "code": code,
        "name": props["nam"],
        # A department's IGN code starts with its province's 2-digit code.
        "province_code": code[:2] if kind == RegionKind.DEPARTMENT else None,
        "geometry": from_shape(geometry, srid=4326),
    }


def main() -> None:
    rows = {kind: [_row(kind, f) for f in _fetch(layer)] for kind, layer in LAYERS.items()}

    with SessionLocal() as session:
        session.execute(delete(Region))
        for kind, kind_rows in rows.items():
            session.execute(insert(Region), kind_rows)
            print(f"{kind.value}: {len(kind_rows)} regions")
        session.commit()


if __name__ == "__main__":
    main()
