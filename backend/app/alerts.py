"""Region alerts: after a detection run is saved, find the new detections that fall
inside users' alert areas, record them as alerts and send them to each area's owner:
by email (to their confirmed alternate address, if they set one, unless they turned
email off) and by Telegram, if they connected it."""

import math
from datetime import timedelta

from geoalchemy2.shape import to_shape
from sqlalchemy import exists, func, insert, select
from sqlalchemy.orm import Session, aliased

from backend.app.crud import unlink_telegram_chat, user_settings_by_user
from backend.app.mailer import AlertedDetection, send_alert_email
from backend.app.models import AlertArea, AlertNotification, Detection
from backend.app.telegram import send_alert
from backend.app.tokens import utcnow

#: A fire stays visible for hours, across many runs. A detection doesn't alert an area
#: again if that area already alerted for one within COOLDOWN_RADIUS_KM and COOLDOWN
#: of it (by image time).
COOLDOWN = timedelta(hours=6)
COOLDOWN_RADIUS_KM = 5.0


def _distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Haversine distance between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def _matches(session: Session, detection_ids: list[int]) -> list[tuple[AlertArea, Detection]]:
    """(area, detection) pairs for active areas containing a new detection at or above
    the area's min_probability, minus those still in cooldown. Highest probability first
    within each area."""
    previous = aliased(Detection)
    in_cooldown = (
        exists()
        .where(
            AlertNotification.area_id == AlertArea.id,
            previous.id == AlertNotification.detection_id,
            previous.image_time > Detection.image_time - COOLDOWN,
            func.ST_DWithin(
                func.geography(previous.location), func.geography(Detection.location), COOLDOWN_RADIUS_KM * 1000
            ),
        )
        .correlate(AlertArea, Detection)
    )
    query = (
        select(AlertArea, Detection)
        .join(
            Detection,
            func.ST_DWithin(AlertArea.center, func.geography(Detection.location), AlertArea.radius_km * 1000),
        )
        .where(
            Detection.id.in_(detection_ids),
            AlertArea.active,
            Detection.probability >= AlertArea.min_probability,
            ~in_cooldown,
        )
        .order_by(AlertArea.id, Detection.probability.desc())
    )
    return [(area, detection) for area, detection in session.execute(query)]


def process_new_detections(session: Session, detection_ids: list[int]) -> int:
    """Raise alerts for these just-saved detections. Returns the number of alerts."""
    if not detection_ids:
        return 0

    # One fire usually covers several neighbouring pixels in the same run. The cooldown
    # query can't see the other new detections, so apply it within the batch here too:
    # keep only the strongest one, so it's one alert, not several.
    by_area: dict[int, tuple[AlertArea, list[tuple[Detection, tuple[float, float]]]]] = {}
    for area, detection in _matches(session, detection_ids):
        point = to_shape(detection.location)
        latlon = (point.y, point.x)
        kept = by_area.setdefault(area.id, (area, []))[1]
        if not any(
            _distance_km(latlon, other_latlon) <= COOLDOWN_RADIUS_KM
            and abs(detection.image_time - other.image_time) < COOLDOWN
            for other, other_latlon in kept
        ):
            kept.append((detection, latlon))

    settings_by_user = user_settings_by_user(session, {area.user_id for area, _ in by_area.values()})
    for area, kept in by_area.values():
        settings = settings_by_user.get(area.user_id)
        alerted = [
            AlertedDetection(lat=lat, lon=lon, probability=d.probability, image_time=d.image_time)
            for d, (lat, lon) in kept
        ]
        emailed_at = telegram_sent_at = None
        if settings is None or settings.email_alerts:
            to = (settings.alert_email if settings else None) or area.email
            emailed_at = utcnow() if send_alert_email(to, area.name, alerted) else None
        if settings is not None and settings.telegram_chat_id is not None:
            result = send_alert(settings.telegram_chat_id, area.name, alerted)
            telegram_sent_at = utcnow() if result == "sent" else None
            if result == "blocked":
                # They blocked the bot: stop trying, and show Telegram as disconnected.
                unlink_telegram_chat(session, settings.telegram_chat_id)
        session.execute(
            insert(AlertNotification),
            [
                {"area_id": area.id, "detection_id": d.id, "emailed_at": emailed_at, "telegram_sent_at": telegram_sent_at}
                for d, _ in kept
            ],
        )
    session.commit()
    return sum(len(kept) for _, kept in by_area.values())
