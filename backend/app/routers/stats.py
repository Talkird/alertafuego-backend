"""Aggregate statistics over stored detections, for the dashboard. Public and
read-only, like GET /detections. All take an optional image_time range."""

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from backend.app.crud import detection_counts_by_hour, detection_counts_by_region, report_counts
from backend.app.db import get_db
from backend.app.models import RegionKind as RegionKindModel
from backend.app.schemas import HourStat, RegionKind, RegionStat, ReportStat

router = APIRouter(prefix="/stats")


@router.get("/regions", response_model=list[RegionStat])
def get_region_stats(
    kind: RegionKind = Query(RegionKind.province),
    since: datetime | None = Query(None),
    until: datetime | None = Query(None),
    db: Session = Depends(get_db),
) -> list[RegionStat]:
    """Detections per province or department, busiest first. Regions with none are
    omitted. Empty until the regions are loaded (backend.scripts.load_regions)."""
    rows = detection_counts_by_region(db, RegionKindModel(kind.value), since, until)
    return [
        RegionStat(code=code, name=name, province_name=province, detection_count=count, max_probability=max_prob)
        for code, name, province, count, max_prob in rows
    ]


@router.get("/hourly", response_model=list[HourStat])
def get_hourly_stats(
    since: datetime | None = Query(None),
    until: datetime | None = Query(None),
    db: Session = Depends(get_db),
) -> list[HourStat]:
    """Detections per hour of the day (Argentina time), all 24 hours."""
    counts = detection_counts_by_hour(db, since, until)
    return [HourStat(hour=hour, detection_count=counts.get(hour, 0)) for hour in range(24)]


@router.get("/reports", response_model=list[ReportStat])
def get_report_stats(
    since: datetime | None = Query(None),
    until: datetime | None = Query(None),
    db: Session = Depends(get_db),
) -> list[ReportStat]:
    """User reports per verdict and category, on detections in range."""
    return [
        ReportStat(verdict=verdict, category=category, report_count=count)
        for verdict, category, count in report_counts(db, since, until)
    ]
