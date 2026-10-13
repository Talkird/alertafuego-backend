"""Region alert endpoints: an authenticated user manages their alert areas (a point
plus a radius) and reads the alerts raised in them. Alerts are raised by
backend.app.alerts after each detection run, and emailed to the area's owner."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from geoalchemy2.shape import to_shape
from sqlalchemy.orm import Session

from backend.app.auth import UserWithEmail, get_current_user, get_current_user_with_email
from backend.app.crud import (
    alert_emails_by_user,
    create_alert_area,
    delete_alert_area,
    get_own_alert_area,
    list_alert_areas,
    list_alerts,
    update_alert_area,
)
from backend.app.db import get_db
from backend.app.models import AlertArea
from backend.app.routers.detections import to_stored_detection
from backend.app.schemas import AlertAreaCreate, AlertAreaResponse, AlertAreaUpdate, AlertResponse

router = APIRouter()

MAX_ALERTS_LIMIT = 500


def _to_response(area: AlertArea, alert_email: str | None) -> AlertAreaResponse:
    """`email` is where the area's alerts go: the owner's confirmed alternate
    address (`alert_email`) if they set one, else their account's."""
    point = to_shape(area.center)
    return AlertAreaResponse(
        id=area.id,
        name=area.name,
        email=alert_email or area.email,
        lat=point.y,
        lon=point.x,
        radius_km=area.radius_km,
        min_probability=area.min_probability,
        active=area.active,
        created_at=area.created_at,
    )


def _own_area(db: Session, area_id: int, user_id: UUID) -> AlertArea:
    area = get_own_alert_area(db, area_id, user_id)
    if area is None:
        raise HTTPException(status_code=404, detail="Alert area not found")
    return area


@router.get("/alert-areas", response_model=list[AlertAreaResponse])
def get_alert_areas(user_id: UUID = Depends(get_current_user), db: Session = Depends(get_db)) -> list[AlertAreaResponse]:
    areas = list_alert_areas(db, user_id)
    alert_email = alert_emails_by_user(db, {user_id}).get(user_id)
    return [_to_response(area, alert_email) for area in areas]


@router.post("/alert-areas", response_model=AlertAreaResponse, status_code=201)
def post_alert_area(
    payload: AlertAreaCreate,
    user: UserWithEmail = Depends(get_current_user_with_email),
    db: Session = Depends(get_db),
) -> AlertAreaResponse:
    area = create_alert_area(db, user.id, user.email, **payload.model_dump())
    return _to_response(area, alert_emails_by_user(db, {user.id}).get(user.id))


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
    area = update_alert_area(db, _own_area(db, area_id, user.id), user.email, updates)
    return _to_response(area, alert_emails_by_user(db, {user.id}).get(user.id))


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
            detection=to_stored_detection(detection, report_count, has_image),
        )
        for notification, area_name, detection, report_count, has_image in list_alerts(db, user_id, limit)
    ]
