"""Database write operations."""

from geoalchemy2.elements import WKTElement
from sqlalchemy import insert
from sqlalchemy.orm import Session

from backend.app.models import Detection
from model.inference.service import DetectionResult


def save_detections(session: Session, result: DetectionResult) -> int:
    """Bulk-insert result.detections as rows. Returns the number of rows inserted."""
    if not result.detections:
        return 0

    rows = [
        {
            "location": WKTElement(f"POINT({lon} {lat})", srid=4326),
            "probability": probability,
            "image_time": result.image_time,
            "bbox_west": result.bbox.west,
            "bbox_south": result.bbox.south,
            "bbox_east": result.bbox.east,
            "bbox_north": result.bbox.north,
            "threshold": result.threshold,
        }
        for lat, lon, probability in result.detections
    ]
    session.execute(insert(Detection), rows)
    session.commit()
    return len(rows)
