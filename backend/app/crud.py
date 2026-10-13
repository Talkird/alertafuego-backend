"""Database read/write operations."""

from datetime import datetime, timedelta
from uuid import UUID

from geoalchemy2.elements import WKTElement
from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session, aliased

from backend.app.models import (
    AlertArea,
    AlertNotification,
    Detection,
    DetectionReport,
    Region,
    RegionKind,
    ReportCategory,
    ReportVerdict,
    UserSettings,
)
from model.inference.service import DetectionResult


def save_detections(session: Session, result: DetectionResult) -> list[int]:
    """Bulk-insert result.detections as rows. Returns the new rows' ids."""
    if not result.detections:
        return []

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
    ids = list(session.scalars(insert(Detection).returning(Detection.id), rows))
    session.commit()
    return ids


def _report_count():
    """Correlated subquery counting a Detection row's reports (no join fan-out)."""
    return (
        select(func.count(DetectionReport.id))
        .where(DetectionReport.detection_id == Detection.id)
        .correlate(Detection)
        .scalar_subquery()
    )


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
    query = select(Detection, _report_count().label("report_count"), Detection.image.isnot(None).label("has_image"))
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


def list_alert_areas(session: Session, user_id: UUID) -> list[AlertArea]:
    """This user's alert areas, oldest first."""
    query = select(AlertArea).where(AlertArea.user_id == user_id).order_by(AlertArea.created_at)
    return list(session.scalars(query))


def get_own_alert_area(session: Session, area_id: int, user_id: UUID) -> AlertArea | None:
    """One alert area, or None if it doesn't exist or belongs to another user."""
    return session.scalar(select(AlertArea).where(AlertArea.id == area_id, AlertArea.user_id == user_id))


def create_alert_area(
    session: Session,
    user_id: UUID,
    email: str,
    name: str,
    lat: float,
    lon: float,
    radius_km: float,
    min_probability: float,
) -> AlertArea:
    area = AlertArea(
        user_id=user_id,
        email=email,
        name=name,
        center=WKTElement(f"POINT({lon} {lat})", srid=4326),
        radius_km=radius_km,
        min_probability=min_probability,
    )
    session.add(area)
    session.commit()
    session.refresh(area)
    return area


def update_alert_area(session: Session, area: AlertArea, email: str, updates: dict) -> AlertArea:
    """Apply a partial update. `updates` may carry lat and lon (both or neither) in
    place of `center`. The email is refreshed from the caller's token every time."""
    if "lat" in updates or "lon" in updates:
        lat, lon = updates.pop("lat"), updates.pop("lon")
        area.center = WKTElement(f"POINT({lon} {lat})", srid=4326)
    for field, value in updates.items():
        setattr(area, field, value)
    area.email = email
    session.commit()
    session.refresh(area)
    return area


def delete_alert_area(session: Session, area: AlertArea) -> None:
    session.delete(area)
    session.commit()


def list_alerts(session: Session, user_id: UUID, limit: int) -> list[tuple[AlertNotification, str, Detection, int, bool]]:
    """This user's alerts, most recent capture first, each with its area's name and
    its detection (plus report count and has_image, as in list_detections)."""
    query = (
        select(
            AlertNotification,
            AlertArea.name,
            Detection,
            _report_count().label("report_count"),
            Detection.image.isnot(None).label("has_image"),
        )
        .join(AlertArea, AlertArea.id == AlertNotification.area_id)
        .join(Detection, Detection.id == AlertNotification.detection_id)
        .where(AlertArea.user_id == user_id)
        .order_by(Detection.image_time.desc(), AlertNotification.id.desc())
        .limit(limit)
    )
    return [tuple(row) for row in session.execute(query)]


def get_user_settings(session: Session, user_id: UUID) -> UserSettings | None:
    return session.get(UserSettings, user_id)


def _get_or_create_user_settings(session: Session, user_id: UUID) -> UserSettings:
    settings = session.get(UserSettings, user_id)
    if settings is None:
        settings = UserSettings(user_id=user_id)
        session.add(settings)
    return settings


def update_user_settings(session: Session, user_id: UUID, updates: dict) -> UserSettings:
    settings = _get_or_create_user_settings(session, user_id)
    for field, value in updates.items():
        setattr(settings, field, value)
    session.commit()
    session.refresh(settings)
    return settings


def set_pending_alert_email(
    session: Session, user_id: UUID, email: str, token_hash: str, expires_at: datetime
) -> UserSettings:
    """Store an address waiting for confirmation. Replaces any earlier pending one
    (its link stops working); the current alert address keeps working meanwhile."""
    settings = _get_or_create_user_settings(session, user_id)
    settings.pending_email = email
    settings.pending_token_hash = token_hash
    settings.pending_expires_at = expires_at
    session.commit()
    session.refresh(settings)
    return settings


def reset_alert_email(session: Session, user_id: UUID) -> UserSettings:
    """Back to the account's address: drop the alternate one and any pending one."""
    settings = _get_or_create_user_settings(session, user_id)
    settings.alert_email = None
    settings.pending_email = None
    settings.pending_token_hash = None
    settings.pending_expires_at = None
    session.commit()
    session.refresh(settings)
    return settings


def cancel_pending_alert_email(session: Session, user_id: UUID) -> UserSettings:
    """Drop the address waiting for confirmation; the current alert address stays."""
    settings = _get_or_create_user_settings(session, user_id)
    settings.pending_email = None
    settings.pending_token_hash = None
    settings.pending_expires_at = None
    session.commit()
    session.refresh(settings)
    return settings


def confirm_alert_email(session: Session, token_hash: str, now: datetime) -> str | None:
    """Make the pending address with this token the alert address. Returns it, or
    None if no unexpired pending address has this token."""
    settings = session.scalar(
        select(UserSettings).where(
            UserSettings.pending_token_hash == token_hash, UserSettings.pending_expires_at > now
        )
    )
    if settings is None:
        return None
    settings.alert_email = settings.pending_email
    settings.pending_email = None
    settings.pending_token_hash = None
    settings.pending_expires_at = None
    session.commit()
    return settings.alert_email


def alert_emails_by_user(session: Session, user_ids: set[UUID]) -> dict[UUID, str]:
    """The confirmed alternate alert address of each of these users who has one."""
    query = select(UserSettings.user_id, UserSettings.alert_email).where(
        UserSettings.user_id.in_(user_ids), UserSettings.alert_email.isnot(None)
    )
    return {user_id: email for user_id, email in session.execute(query)}


def _time_filters(query, since: datetime | None, until: datetime | None):
    if since is not None:
        query = query.where(Detection.image_time >= since)
    if until is not None:
        query = query.where(Detection.image_time <= until)
    return query


def detection_counts_by_region(
    session: Session, kind: RegionKind, since: datetime | None, until: datetime | None
) -> list[tuple[str, str, str | None, int, float]]:
    """(code, name, province name, detection count, max probability) for each region
    of this kind that has at least one detection in range, busiest first."""
    province = aliased(Region)
    query = (
        select(Region.code, Region.name, province.name, func.count(Detection.id), func.max(Detection.probability))
        .join(Detection, func.ST_Contains(Region.geometry, Detection.location))
        .outerjoin(province, (province.kind == RegionKind.PROVINCE) & (province.code == Region.province_code))
        .where(Region.kind == kind)
        .group_by(Region.code, Region.name, province.name)
        .order_by(func.count(Detection.id).desc(), Region.name)
    )
    return [tuple(row) for row in session.execute(_time_filters(query, since, until))]


def detection_counts_by_hour(session: Session, since: datetime | None, until: datetime | None) -> dict[int, int]:
    """Detection count per hour of the day (0-23), Argentina time (UTC-3, no DST).
    image_time is stored as naive UTC. Hours with no detections are omitted."""
    hour = func.extract("hour", Detection.image_time - timedelta(hours=3))
    query = select(hour, func.count(Detection.id)).group_by(hour)
    return {int(h): count for h, count in session.execute(_time_filters(query, since, until))}


def report_counts(
    session: Session, since: datetime | None, until: datetime | None
) -> list[tuple[ReportVerdict, ReportCategory | None, int]]:
    """Report count per (verdict, category), over reports on detections in range."""
    query = (
        select(DetectionReport.verdict, DetectionReport.category, func.count(DetectionReport.id))
        .join(Detection, Detection.id == DetectionReport.detection_id)
        .group_by(DetectionReport.verdict, DetectionReport.category)
    )
    return [tuple(row) for row in session.execute(_time_filters(query, since, until))]
