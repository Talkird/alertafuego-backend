"""Detection report endpoints - authenticated users record their verdict on a
stored detection: either flagging it as not-fire, with a category (industrial
activity, controlled burn, etc), or confirming it's a real fire. Anyone can read
the reports for a detection (click-to-fetch from the map)."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.auth import get_current_user
from backend.app.crud import create_report, get_own_report, list_reports, update_report
from backend.app.db import get_db
from backend.app.models import Detection
from backend.app.schemas import ReportCreate, ReportPublic, ReportResponse, ReportUpdate

router = APIRouter(prefix="/detections")


@router.get("/{detection_id}/reports", response_model=list[ReportPublic])
def get_reports(detection_id: int, db: Session = Depends(get_db)) -> list[ReportPublic]:
    if db.get(Detection, detection_id) is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    return list_reports(db, detection_id)


@router.get("/{detection_id}/reports/mine", response_model=ReportPublic)
def get_my_report(
    detection_id: int,
    user_id: UUID = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ReportPublic:
    if db.get(Detection, detection_id) is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    report = get_own_report(db, detection_id, user_id)
    if report is None:
        raise HTTPException(status_code=404, detail="You have not reported this detection")
    return report


@router.post("/{detection_id}/reports", response_model=ReportResponse, status_code=201)
def submit_report(
    detection_id: int,
    payload: ReportCreate,
    user_id: UUID = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ReportResponse:
    if db.get(Detection, detection_id) is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    category = payload.category.value if payload.category is not None else None
    try:
        return create_report(db, detection_id, user_id, payload.verdict.value, category, payload.comment)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="You already reported this detection") from exc


@router.patch("/{detection_id}/reports", response_model=ReportResponse)
def edit_report(
    detection_id: int,
    payload: ReportUpdate,
    user_id: UUID = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ReportResponse:
    updates = payload.model_dump(exclude_unset=True)
    if "verdict" in updates:
        updates["verdict"] = payload.verdict.value
    if "category" in updates and updates["category"] is not None:
        updates["category"] = payload.category.value
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    try:
        report = update_report(db, detection_id, user_id, updates)
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if report is None:
        raise HTTPException(status_code=404, detail="You have not reported this detection")
    return report
