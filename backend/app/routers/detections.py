"""Read endpoint for previously persisted detections - no Earth Engine calls, no
side effects, fast. This is the endpoint a frontend should poll/query."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from geoalchemy2.shape import to_shape
from sqlalchemy.orm import Session

from backend.app.crud import get_detection_image, list_detections
from backend.app.db import get_db
from backend.app.models import Detection
from backend.app.schemas import BBoxSchema, StoredDetection

router = APIRouter(prefix="/detections")

MAX_LIMIT = 5000


@router.get("", response_model=list[StoredDetection])
def get_detections(
    west: float | None = Query(None),
    south: float | None = Query(None),
    east: float | None = Query(None),
    north: float | None = Query(None),
    since: datetime | None = Query(None),
    until: datetime | None = Query(None),
    limit: int = Query(500, le=MAX_LIMIT),
    db: Session = Depends(get_db),
) -> list[StoredDetection]:
    bbox = None
    if None not in (west, south, east, north):
        bbox = (west, south, east, north)

    rows = list_detections(db, bbox=bbox, since=since, until=until, limit=limit)
    return [
        StoredDetection(
            id=row.id,
            lat=to_shape(row.location).y,
            lon=to_shape(row.location).x,
            probability=row.probability,
            image_time=row.image_time,
            detected_at=row.detected_at,
            bbox=BBoxSchema(west=row.bbox_west, south=row.bbox_south, east=row.bbox_east, north=row.bbox_north),
            threshold=row.threshold,
            report_count=report_count,
            has_image=has_image,
        )
        for row, report_count, has_image in rows
    ]


@router.get("/{detection_id}/image")
def get_detection_image_route(detection_id: int, db: Session = Depends(get_db)) -> Response:
    if db.get(Detection, detection_id) is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    image = get_detection_image(db, detection_id)
    if image is None:
        raise HTTPException(status_code=404, detail="This detection has no stored image")
    return Response(content=image, media_type="image/png")
