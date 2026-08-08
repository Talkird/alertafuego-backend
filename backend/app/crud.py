"""Database read/write operations."""

from datetime import datetime
from uuid import UUID

from geoalchemy2.elements import WKTElement
from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from backend.app.models import Detection, FalsePositiveReport, ReportCategory
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


def list_detections(
    session: Session,
    bbox: tuple[float, float, float, float] | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 500,
) -> list[tuple[Detection, int]]:
    """Query stored detections, optionally filtered by bbox (west, south, east, north)
    and/or image_time range. Ordered most-recent capture first. Each row is paired
    with its false-positive report count (correlated subquery, no join fan-out)."""
    report_count = (
        select(func.count(FalsePositiveReport.id))
        .where(FalsePositiveReport.detection_id == Detection.id)
        .correlate(Detection)
        .scalar_subquery()
    )
    query = select(Detection, report_count.label("report_count"))
    if bbox is not None:
        west, south, east, north = bbox
        envelope = func.ST_MakeEnvelope(west, south, east, north, 4326)
        query = query.where(func.ST_Intersects(Detection.location, envelope))
    if since is not None:
        query = query.where(Detection.image_time >= since)
    if until is not None:
        query = query.where(Detection.image_time <= until)
    query = query.order_by(Detection.image_time.desc()).limit(limit)
    return [(detection, count) for detection, count in session.execute(query)]


def list_reports(session: Session, detection_id: int) -> list[FalsePositiveReport]:
    """All false-positive reports for one detection, most recent first."""
    query = (
        select(FalsePositiveReport)
        .where(FalsePositiveReport.detection_id == detection_id)
        .order_by(FalsePositiveReport.created_at.desc())
    )
    return list(session.scalars(query))


def create_report(
    session: Session,
    detection_id: int,
    user_id: UUID,
    category: ReportCategory,
    comment: str | None,
) -> FalsePositiveReport:
    """Insert a false-positive report. Raises sqlalchemy.exc.IntegrityError if this
    user already reported this detection, or if detection_id doesn't exist."""
    report = FalsePositiveReport(detection_id=detection_id, user_id=user_id, category=category, comment=comment)
    session.add(report)
    session.commit()
    session.refresh(report)
    return report


def update_report(
    session: Session,
    detection_id: int,
    user_id: UUID,
    updates: dict,
) -> FalsePositiveReport | None:
    """Apply a partial update (category and/or comment) to this user's existing
    report for this detection. Returns None if they haven't reported it yet."""
    report = session.scalar(
        select(FalsePositiveReport).where(
            FalsePositiveReport.detection_id == detection_id,
            FalsePositiveReport.user_id == user_id,
        )
    )
    if report is None:
        return None
    for field, value in updates.items():
        setattr(report, field, value)
    session.commit()
    session.refresh(report)
    return report
