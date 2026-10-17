"""The place filter that the detection and stats endpoints share: a province or
department (IGN code) or one of the signed-in user's alert areas. The frontend ships
the list of regions itself (backend.scripts.export_regions)."""

from uuid import UUID

from fastapi import Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.auth import get_optional_user
from backend.app.crud import detection_place_filter, get_own_alert_area, get_region
from backend.app.db import get_db
from backend.app.schemas import REGION_CODE_PATTERN


def detection_place(
    region_code: str | None = Query(None, pattern=REGION_CODE_PATTERN, description="Province or department code"),
    area_id: int | None = Query(None, description="One of your alert areas (needs a token)"),
    user_id: UUID | None = Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """Dependency: the WHERE condition for the region_code or area_id query parameter
    (at most one), or None to leave detections unfiltered by place."""
    if region_code is not None and area_id is not None:
        raise HTTPException(status_code=400, detail="Filter by region_code or area_id, not both")
    if region_code is not None:
        if get_region(db, region_code) is None:
            raise HTTPException(status_code=404, detail="Region not found")
        return detection_place_filter(region_code=region_code)
    if area_id is not None:
        if user_id is None:
            raise HTTPException(status_code=401, detail="Sign in to filter by your alert areas")
        area = get_own_alert_area(db, area_id, user_id)
        if area is None:
            raise HTTPException(status_code=404, detail="Alert area not found")
        return detection_place_filter(area=area)
    return None
