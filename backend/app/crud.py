"""Database read/write operations."""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from geoalchemy2.elements import WKTElement
from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session, aliased, defer

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

if TYPE_CHECKING:
    # Type only: importing the inference service would load torch with the API.
    from model.inference.service import DetectionResult


def save_detections(session: Session, result: "DetectionResult") -> list[int]:
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
    place=None,
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
    query = _place_filter(query, place)
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
    lat: float | None,
    lon: float | None,
    radius_km: float | None,
    polygon: list[tuple[float, float]] | None,
    region_code: str | None,
    min_probability: float,
) -> AlertArea:
    """A circle area (lat, lon, radius_km), a polygon one, or a region one
    (region_code). Exactly one is given (schemas.AlertAreaCreate checks it)."""
    area = AlertArea(
        user_id=user_id,
        email=email,
        name=name,
        center=WKTElement(f"POINT({lon} {lat})", srid=4326) if lat is not None else None,
        radius_km=radius_km,
        polygon=polygon_element(polygon) if polygon else None,
        region_code=region_code,
        min_probability=min_probability,
    )
    session.add(area)
    session.commit()
    session.refresh(area)
    return area


def update_alert_area(session: Session, area: AlertArea, email: str, updates: dict) -> AlertArea:
    """Apply a partial update. `updates` may carry lat and lon (both or neither) in
    place of `center`, and polygon as [lat, lon] corners. Setting one shape clears the
    others, so the area switches kind. The email is refreshed from the caller's token
    every time. Raises ValueError (and rolls back) if the result isn't exactly one whole
    shape (say, a region switched to a circle without a radius)."""
    if "lat" in updates or "lon" in updates:
        lat, lon = updates.pop("lat"), updates.pop("lon")
        area.center = WKTElement(f"POINT({lon} {lat})", srid=4326)
        area.polygon = area.region_code = None
    if "polygon" in updates:
        area.polygon = polygon_element(updates.pop("polygon"))
        area.center = area.radius_km = area.region_code = None
    if "region_code" in updates:
        area.center = area.radius_km = area.polygon = None
    for field, value in updates.items():
        setattr(area, field, value)
    has = [area.center is not None, area.polygon is not None, area.region_code is not None]
    if sum(has) != 1 or (area.center is not None) != (area.radius_km is not None):
        session.rollback()
        raise ValueError("an area is exactly one of: a circle (lat, lon and radius_km), a polygon, a region")
    area.email = email
    session.commit()
    session.refresh(area)
    return area


def polygon_element(points: list[tuple[float, float]]) -> WKTElement:
    """A closed WKT polygon from [lat, lon] corners."""
    ring = [*points, points[0]]
    return WKTElement("POLYGON((" + ",".join(f"{lon} {lat}" for lat, lon in ring) + "))", srid=4326)


#: As large as the largest circle (200 km radius), so neither is a way around the other.
MAX_POLYGON_AREA_KM2 = 3.1416 * 200**2


def polygon_problem(session: Session, points: list[tuple[float, float]]) -> str | None:
    """Why this polygon can't be an alert area (it crosses itself, or it's too big),
    or None if it can."""
    polygon = func.ST_GeomFromEWKT(f"SRID=4326;{polygon_element(points).data}")
    valid, area_m2 = session.execute(select(func.ST_IsValid(polygon), func.ST_Area(func.geography(polygon)))).one()
    if not valid:
        return "The polygon's sides cross each other"
    if area_m2 / 1e6 > MAX_POLYGON_AREA_KM2:
        return f"The polygon is larger than {MAX_POLYGON_AREA_KM2:,.0f} km²"
    return None


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


def _save(session: Session, settings: UserSettings) -> UserSettings:
    session.commit()
    session.refresh(settings)
    return settings


def update_user_settings(session: Session, user_id: UUID, updates: dict) -> UserSettings:
    settings = _get_or_create_user_settings(session, user_id)
    for field, value in updates.items():
        setattr(settings, field, value)
    return _save(session, settings)


def _clear_pending_email(settings: UserSettings) -> None:
    settings.pending_email = None
    settings.pending_token_hash = None
    settings.pending_expires_at = None


def set_pending_alert_email(
    session: Session, user_id: UUID, email: str, token_hash: str, expires_at: datetime
) -> UserSettings:
    """Store an address waiting for confirmation. Replaces any earlier pending one
    (its link stops working); the current alert address keeps working meanwhile."""
    settings = _get_or_create_user_settings(session, user_id)
    settings.pending_email = email
    settings.pending_token_hash = token_hash
    settings.pending_expires_at = expires_at
    return _save(session, settings)


def reset_alert_email(session: Session, user_id: UUID) -> UserSettings:
    """Back to the account's address: drop the alternate one and any pending one."""
    settings = _get_or_create_user_settings(session, user_id)
    settings.alert_email = None
    _clear_pending_email(settings)
    return _save(session, settings)


def cancel_pending_alert_email(session: Session, user_id: UUID) -> UserSettings:
    """Drop the address waiting for confirmation; the current alert address stays."""
    settings = _get_or_create_user_settings(session, user_id)
    _clear_pending_email(settings)
    return _save(session, settings)


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
    _clear_pending_email(settings)
    session.commit()
    return settings.alert_email


def set_telegram_link_token(session: Session, user_id: UUID, token_hash: str, expires_at: datetime) -> UserSettings:
    """Store the token of a new one-time bot link. Replaces any earlier one."""
    settings = _get_or_create_user_settings(session, user_id)
    settings.telegram_link_token_hash = token_hash
    settings.telegram_link_expires_at = expires_at
    return _save(session, settings)


def link_telegram_chat(session: Session, token_hash: str, chat_id: int, name: str, now: datetime) -> bool:
    """Connect a Telegram chat to the user whose unexpired link token this is.
    A chat belongs to one user: one linked elsewhere moves here. Returns whether
    the token was valid."""
    settings = session.scalar(
        select(UserSettings).where(
            UserSettings.telegram_link_token_hash == token_hash, UserSettings.telegram_link_expires_at > now
        )
    )
    if settings is None:
        return False
    previous = session.scalar(select(UserSettings).where(UserSettings.telegram_chat_id == chat_id))
    if previous is not None and previous is not settings:
        previous.telegram_chat_id = None
        previous.telegram_name = None
        session.flush()
    settings.telegram_chat_id = chat_id
    settings.telegram_name = name
    settings.telegram_link_token_hash = None
    settings.telegram_link_expires_at = None
    session.commit()
    return True


def unlink_telegram(session: Session, user_id: UUID) -> UserSettings:
    settings = _get_or_create_user_settings(session, user_id)
    settings.telegram_chat_id = None
    settings.telegram_name = None
    settings.telegram_link_token_hash = None
    settings.telegram_link_expires_at = None
    return _save(session, settings)


def unlink_telegram_chat(session: Session, chat_id: int) -> bool:
    """Disconnect this chat from whichever user has it. Returns whether one did."""
    settings = session.scalar(select(UserSettings).where(UserSettings.telegram_chat_id == chat_id))
    if settings is None:
        return False
    settings.telegram_chat_id = None
    settings.telegram_name = None
    session.commit()
    return True


def user_settings_by_user(session: Session, user_ids: set[UUID]) -> dict[UUID, UserSettings]:
    """The settings rows of these users (users without one have the defaults)."""
    return {s.user_id: s for s in session.scalars(select(UserSettings).where(UserSettings.user_id.in_(user_ids)))}


def region_kind_of(code: str) -> RegionKind:
    """IGN codes: 2 digits for a province, 5 for a department."""
    return RegionKind.PROVINCE if len(code) == 2 else RegionKind.DEPARTMENT


def get_region(session: Session, code: str) -> Region | None:
    query = select(Region).options(defer(Region.geometry)).where(Region.code == code, Region.kind == region_kind_of(code))
    return session.scalar(query)


def detection_place_filter(region_code: str | None = None, area: AlertArea | None = None):
    """A WHERE condition keeping detections inside a region or an alert area (circle,
    polygon or region), or None for no place filter."""
    if area is not None and area.region_code is not None:
        region_code = area.region_code
    if region_code is not None:
        geometry = (
            select(Region.geometry)
            .where(Region.code == region_code, Region.kind == region_kind_of(region_code))
            .scalar_subquery()
        )
        return func.ST_Contains(geometry, Detection.location)
    if area is not None and area.polygon is not None:
        polygon = select(AlertArea.polygon).where(AlertArea.id == area.id).scalar_subquery()
        return func.ST_Contains(polygon, Detection.location)
    if area is not None:
        center = select(AlertArea.center).where(AlertArea.id == area.id).scalar_subquery()
        return func.ST_DWithin(center, func.geography(Detection.location), area.radius_km * 1000)
    return None


def _place_filter(query, place):
    return query if place is None else query.where(place)


def _time_filters(query, since: datetime | None, until: datetime | None):
    if since is not None:
        query = query.where(Detection.image_time >= since)
    if until is not None:
        query = query.where(Detection.image_time <= until)
    return query


def detection_counts_by_region(
    session: Session, kind: RegionKind, since: datetime | None, until: datetime | None, place=None
) -> list[tuple[str, str, str | None, int, float]]:
    """(code, name, province name, detection count, max probability) for each region
    of this kind that has at least one detection in range (and in `place`, from
    detection_place_filter), busiest first."""
    province = aliased(Region)
    query = (
        select(Region.code, Region.name, province.name, func.count(Detection.id), func.max(Detection.probability))
        .join(Detection, func.ST_Contains(Region.geometry, Detection.location))
        .outerjoin(province, (province.kind == RegionKind.PROVINCE) & (province.code == Region.province_code))
        .where(Region.kind == kind)
        .group_by(Region.code, Region.name, province.name)
        .order_by(func.count(Detection.id).desc(), Region.name)
    )
    return [tuple(row) for row in session.execute(_place_filter(_time_filters(query, since, until), place))]


def detection_counts_by_hour(
    session: Session, since: datetime | None, until: datetime | None, place=None
) -> dict[int, int]:
    """Detection count per hour of the day (0-23), Argentina time (UTC-3, no DST).
    image_time is stored as naive UTC. Hours with no detections are omitted."""
    hour = func.extract("hour", Detection.image_time - timedelta(hours=3))
    query = select(hour, func.count(Detection.id)).group_by(hour)
    return {int(h): count for h, count in session.execute(_place_filter(_time_filters(query, since, until), place))}


#: Recurrence grid cell, in degrees: ~2 km, about a GOES-19 pixel over Argentina.
RECURRENCE_CELL_DEG = 0.02


def recurrence_cells(
    session: Session, since: datetime | None, until: datetime | None, place=None
) -> list[tuple[float, float, int, int, int, datetime]]:
    """Where fires come back: detections grouped into RECURRENCE_CELL_DEG cells, each
    as (center lat, center lon, distinct days with fire, distinct years, detection
    count, last image_time), most days first. Days and years are Argentina time
    (UTC-3), so a fire burning for hours counts once a day."""
    local_time = Detection.image_time - timedelta(hours=3)
    cell = RECURRENCE_CELL_DEG
    lat = (func.floor(func.ST_Y(Detection.location) / cell) + 0.5) * cell
    lon = (func.floor(func.ST_X(Detection.location) / cell) + 0.5) * cell
    days = func.count(func.distinct(func.date(local_time)))
    query = (
        select(
            lat.label("lat"),
            lon.label("lon"),
            days,
            func.count(func.distinct(func.extract("year", local_time))),
            func.count(Detection.id),
            func.max(Detection.image_time),
        )
        .group_by("lat", "lon")
        .order_by(days.desc())
    )
    return [tuple(row) for row in session.execute(_place_filter(_time_filters(query, since, until), place))]


def report_counts(
    session: Session, since: datetime | None, until: datetime | None, place=None
) -> list[tuple[ReportVerdict, ReportCategory | None, int]]:
    """Report count per (verdict, category), over reports on detections in range."""
    query = (
        select(DetectionReport.verdict, DetectionReport.category, func.count(DetectionReport.id))
        .join(Detection, Detection.id == DetectionReport.detection_id)
        .group_by(DetectionReport.verdict, DetectionReport.category)
    )
    return [tuple(row) for row in session.execute(_place_filter(_time_filters(query, since, until), place))]
