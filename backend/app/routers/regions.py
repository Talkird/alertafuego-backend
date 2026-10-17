"""Argentina's provinces and departments (IGN), for choosing a region alert area or
a statistics filter, and the place filter that the detection and stats endpoints
share."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from backend.app.auth import get_optional_user
from backend.app.crud import detection_place_filter, get_own_alert_area, get_region, list_regions, region_geojson
from backend.app.db import get_db
from backend.app.schemas import REGION_CODE_PATTERN, RegionResponse

router = APIRouter(prefix="/regions")


@router.get("", response_model=list[RegionResponse])
def get_regions(db: Session = Depends(get_db)) -> list[RegionResponse]:
    """Every province and then every department, by name. Empty until the regions are
    loaded (backend.scripts.load_regions)."""
    return [
        RegionResponse(code=r.code, name=r.name, province_code=r.province_code) for r in list_regions(db)
    ]


@router.get("/{code}/geometry")
def get_region_geometry(code: str, db: Session = Depends(get_db)) -> Response:
    """The region's outline (GeoJSON geometry), simplified for drawing on a map."""
    geojson = region_geojson(db, code)
    if geojson is None:
        raise HTTPException(status_code=404, detail="Region not found")
    return Response(content=geojson, media_type="application/geo+json")


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
