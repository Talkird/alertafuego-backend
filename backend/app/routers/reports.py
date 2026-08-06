"""False-positive report endpoints - authenticated users flag a stored detection
as not-fire, with a category (industrial activity, controlled burn, etc); anyone
can read the reports for a detection (click-to-fetch from the map)."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.auth import get_current_user
from backend.app.crud import create_report, list_reports
from backend.app.db import get_db
from backend.app.models import Detection
from backend.app.schemas import ReportCreate, ReportPublic, ReportResponse

router = APIRouter(prefix="/detections")


@router.get("/{detection_id}/reports", response_model=list[ReportPublic])
def get_reports(detection_id: int, db: Session = Depends(get_db)) -> list[ReportPublic]:
    if db.get(Detection, detection_id) is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    return list_reports(db, detection_id)


@router.post("/{detection_id}/reports", response_model=ReportResponse, status_code=201)
def report_false_positive(
    detection_id: int,
    payload: ReportCreate,
    user_id: UUID = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ReportResponse:
    if db.get(Detection, detection_id) is None:
        raise HTTPException(status_code=404, detail="Detection not found")
    try:
        return create_report(db, detection_id, user_id, payload.category.value, payload.comment)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="You already reported this detection") from exc
