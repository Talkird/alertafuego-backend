"""Database read/write operations."""

from datetime import datetime
from uuid import UUID

from geoalchemy2.elements import WKTElement
from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from backend.app.models import Detection, DetectionReport, ReportCategory, ReportVerdict
from model.inference.service import DetectionResult


def save_detections(session: Session, result: DetectionResult) -> int:
    """Bulk-insert result.detections as rows. Returns the number of rows inserted."""
    if not result.detections:
        return 0

    rows = [
        {
            "location": WKTElement(f"POINT({lon} {lat})", srid=4326),
            "probability": probability,
            "image": image,
            "image_time": result.image_time,
            "bbox_west": result.bbox.west,
            "bbox_south": result.bbox.south,
            "bbox_east": result.bbox.east,
            "bbox_north": result.bbox.north,
            "threshold": result.threshold,
        }
        for lat, lon, probability, image in result.detections
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
) -> list[tuple[Detection, int, bool]]:
    """Query stored detections, optionally filtered by bbox (west, south, east, north)
    and/or image_time range. Ordered most-recent capture first. Each row is paired
    with its report count (correlated subquery, no join fan-out) and whether it has
    a stored image (Detection.image is deferred, so this avoids pulling every row's
    image bytes just to list them)."""
    report_count = (
        select(func.count(DetectionReport.id))
        .where(DetectionReport.detection_id == Detection.id)
        .correlate(Detection)
        .scalar_subquery()
    )
    query = select(Detection, report_count.label("report_count"), Detection.image.isnot(None).label("has_image"))
    if bbox is not None:
        west, south, east, north = bbox
        envelope = func.ST_MakeEnvelope(west, south, east, north, 4326)
        query = query.where(func.ST_Intersects(Detection.location, envelope))
    if since is not None:
        query = query.where(Detection.image_time >= since)
    if until is not None:
        query = query.where(Detection.image_time <= until)
    query = query.order_by(Detection.image_time.desc()).limit(limit)
    return [(detection, count, has_image) for detection, count, has_image in session.execute(query)]


def get_detection_image(session: Session, detection_id: int) -> bytes | None:
    """The stored thumbnail PNG for one detection, or None if it has none."""
    return session.scalar(select(Detection.image).where(Detection.id == detection_id))


def list_reports(session: Session, detection_id: int) -> list[DetectionReport]:
    """All reports (false-positive or confirmed-fire) for one detection, most recent first."""
    query = (
        select(DetectionReport)
        .where(DetectionReport.detection_id == detection_id)
        .order_by(DetectionReport.created_at.desc())
    )
    return list(session.scalars(query))


def create_report(
    session: Session,
    detection_id: int,
    user_id: UUID,
    verdict: ReportVerdict,
    category: ReportCategory | None,
    comment: str | None,
) -> DetectionReport:
    """Insert a report. Raises sqlalchemy.exc.IntegrityError if this user already
    reported this detection, or if detection_id doesn't exist."""
    report = DetectionReport(detection_id=detection_id, user_id=user_id, verdict=verdict, category=category, comment=comment)
    session.add(report)
    session.commit()
    session.refresh(report)
    return report


def get_own_report(session: Session, detection_id: int, user_id: UUID) -> DetectionReport | None:
    """This user's own report for this detection, or None if they haven't reported it."""
    return session.scalar(
        select(DetectionReport).where(
            DetectionReport.detection_id == detection_id,
            DetectionReport.user_id == user_id,
        )
    )


def update_report(
    session: Session,
    detection_id: int,
    user_id: UUID,
    updates: dict,
) -> DetectionReport | None:
    """Apply a partial update (verdict, category and/or comment) to this user's
    existing report for this detection. Returns None if they haven't reported it
    yet. Raises ValueError if the resulting verdict/category combination is invalid
    (confirmed_fire must not carry a category, false_positive must)."""
    report = get_own_report(session, detection_id, user_id)
    if report is None:
        return None
    for field, value in updates.items():
        setattr(report, field, value)
    if report.verdict == ReportVerdict.CONFIRMED_FIRE:
        report.category = None
    elif report.category is None:
        raise ValueError("category is required when verdict is false_positive")
    session.commit()
    session.refresh(report)
    return report
