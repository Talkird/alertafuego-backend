"""Region alert endpoints: an authenticated user manages their alert areas (a point
plus a radius, a polygon, or a province or department) and reads the alerts raised
in them. Alerts are raised by
backend.app.alerts after each detection run, and emailed to the area's owner."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from geoalchemy2.shape import to_shape
from sqlalchemy.orm import Session

from backend.app.auth import UserWithEmail, get_current_user, get_current_user_with_email
from backend.app.crud import (
    create_alert_area,
    delete_alert_area,
    get_own_alert_area,
    get_region,
    polygon_problem,
    list_alert_areas,
    list_alerts,
    update_alert_area,
    user_settings_by_user,
)
from backend.app.db import get_db
from backend.app.models import AlertArea
from backend.app.routers.detections import to_stored_detection
from backend.app.schemas import AlertAreaCreate, AlertAreaResponse, AlertAreaUpdate, AlertResponse, RegionResponse

router = APIRouter()

MAX_ALERTS_LIMIT = 500


def _to_response(db: Session, area: AlertArea, alert_email: str | None) -> AlertAreaResponse:
    """`email` is where the area's alerts go: the owner's confirmed alternate
    address (`alert_email`) if they set one, else their account's."""
    point = to_shape(area.center) if area.center is not None else None
    region = get_region(db, area.region_code) if area.region_code else None
    return AlertAreaResponse(
        id=area.id,
        name=area.name,
        email=alert_email or area.email,
        lat=point.y if point else None,
        lon=point.x if point else None,
        radius_km=area.radius_km,
        # Stored closed (first corner repeated at the end); returned open, as sent.
        polygon=[(lat, lon) for lon, lat in to_shape(area.polygon).exterior.coords[:-1]]
        if area.polygon is not None
        else None,
        region=RegionResponse(code=region.code, name=region.name, province_code=region.province_code)
        if region
        else None,
        min_probability=area.min_probability,
        active=area.active,
        created_at=area.created_at,
    )


def _alert_email(db: Session, user_id: UUID) -> str | None:
    """The user's confirmed alternate alert address, if they set one."""
    settings = user_settings_by_user(db, {user_id}).get(user_id)
    return settings.alert_email if settings else None


def _check_shape(db: Session, region_code: str | None, polygon: list[tuple[float, float]] | None) -> None:
    if region_code is not None and get_region(db, region_code) is None:
        raise HTTPException(status_code=404, detail="Region not found")
    if polygon is not None and (problem := polygon_problem(db, polygon)):
        raise HTTPException(status_code=422, detail=problem)


def _own_area(db: Session, area_id: int, user_id: UUID) -> AlertArea:
    area = get_own_alert_area(db, area_id, user_id)
    if area is None:
        raise HTTPException(status_code=404, detail="Alert area not found")
    return area


@router.get("/alert-areas", response_model=list[AlertAreaResponse])
def get_alert_areas(user_id: UUID = Depends(get_current_user), db: Session = Depends(get_db)) -> list[AlertAreaResponse]:
    areas = list_alert_areas(db, user_id)
    alert_email = _alert_email(db, user_id)
    return [_to_response(db, area, alert_email) for area in areas]


@router.post("/alert-areas", response_model=AlertAreaResponse, status_code=201)
def post_alert_area(
    payload: AlertAreaCreate,
    user: UserWithEmail = Depends(get_current_user_with_email),
    db: Session = Depends(get_db),
) -> AlertAreaResponse:
    _check_shape(db, payload.region_code, payload.polygon)
    area = create_alert_area(db, user.id, user.email, **payload.model_dump())
    return _to_response(db, area, _alert_email(db, user.id))


@router.patch("/alert-areas/{area_id}", response_model=AlertAreaResponse)
def patch_alert_area(
    area_id: int,
    payload: AlertAreaUpdate,
    user: UserWithEmail = Depends(get_current_user_with_email),
    db: Session = Depends(get_db),
) -> AlertAreaResponse:
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    if any(updates.get(field) is None for field in updates):
        raise HTTPException(status_code=400, detail="Fields can't be set to null")
    _check_shape(db, updates.get("region_code"), updates.get("polygon"))
    try:
        area = update_alert_area(db, _own_area(db, area_id, user.id), user.email, updates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_response(db, area, _alert_email(db, user.id))


@router.delete("/alert-areas/{area_id}", status_code=204)
def remove_alert_area(area_id: int, user_id: UUID = Depends(get_current_user), db: Session = Depends(get_db)) -> None:
    delete_alert_area(db, _own_area(db, area_id, user_id))


@router.get("/alerts", response_model=list[AlertResponse])
def get_alerts(
    limit: int = Query(100, le=MAX_ALERTS_LIMIT),
    user_id: UUID = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[AlertResponse]:
    return [
        AlertResponse(
            id=notification.id,
            area_id=notification.area_id,
            area_name=area_name,
            created_at=notification.created_at,
            emailed=notification.emailed_at is not None,
            telegram_sent=notification.telegram_sent_at is not None,
            detection=to_stored_detection(detection, report_count, has_image),
        )
        for notification, area_name, detection, report_count, has_image in list_alerts(db, user_id, limit)
    ]
