"""Pydantic response models for the detection endpoints."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class ReportVerdict(str, Enum):
    false_positive = "false_positive"
    confirmed_fire = "confirmed_fire"


class ReportCategory(str, Enum):
    """Only applicable when verdict is false_positive."""

    industrial_activity = "industrial_activity"
    controlled_burn = "controlled_burn"
    gas_flare = "gas_flare"
    sun_glint = "sun_glint"
    sensor_noise = "sensor_noise"
    other = "other"


class ReportCreate(BaseModel):
    verdict: ReportVerdict
    category: ReportCategory | None = None
    comment: str | None = None

    @model_validator(mode="after")
    def _category_matches_verdict(self) -> "ReportCreate":
        if self.verdict == ReportVerdict.false_positive and self.category is None:
            raise ValueError("category is required when verdict is false_positive")
        if self.verdict == ReportVerdict.confirmed_fire and self.category is not None:
            raise ValueError("category is not applicable when verdict is confirmed_fire")
        return self


class ReportUpdate(BaseModel):
    """Partial update for an existing report - fields left unset are unchanged."""

    verdict: ReportVerdict | None = None
    category: ReportCategory | None = None
    comment: str | None = None


class ReportResponse(BaseModel):
    id: int
    detection_id: int
    user_id: UUID
    verdict: ReportVerdict
    category: ReportCategory | None
    comment: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ReportPublic(BaseModel):
    """Report shape for public listing - omits user_id, reporter identity isn't exposed."""

    id: int
    verdict: ReportVerdict
    category: ReportCategory | None
    comment: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class Detection(BaseModel):
    lat: float
    lon: float
    probability: float


class BBoxSchema(BaseModel):
    west: float
    south: float
    east: float
    north: float


class DetectionResponse(BaseModel):
    image_time: datetime
    bbox: BBoxSchema
    threshold: float
    chunk_count: int
    detection_count: int
    detections: list[Detection]


class StoredDetection(BaseModel):
    """One row from the `detections` table - shape mirrors the DB schema directly."""

    id: int
    lat: float
    lon: float
    probability: float
    image_time: datetime
    detected_at: datetime
    bbox: BBoxSchema
    threshold: float
    report_count: int
    has_image: bool


class AlertAreaCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_km: float = Field(gt=0, le=200)
    #: Above the model's 0.5 detection threshold by default, to cut false alarms.
    min_probability: float = Field(0.65, ge=0, le=1)


class AlertAreaUpdate(BaseModel):
    """Partial update - fields left unset are unchanged. lat and lon go together."""

    name: str | None = Field(None, min_length=1, max_length=100)
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=180)
    radius_km: float | None = Field(None, gt=0, le=200)
    min_probability: float | None = Field(None, ge=0, le=1)
    active: bool | None = None

    @model_validator(mode="after")
    def _lat_lon_together(self) -> "AlertAreaUpdate":
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat and lon must be updated together")
        return self


class AlertAreaResponse(BaseModel):
    id: int
    name: str
    email: str
    lat: float
    lon: float
    radius_km: float
    min_probability: float
    active: bool
    created_at: datetime


class AlertResponse(BaseModel):
    """One alert: a detection inside one of the user's alert areas."""

    id: int
    area_id: int
    area_name: str
    created_at: datetime
    emailed: bool
    detection: StoredDetection


class RegionKind(str, Enum):
    province = "province"
    department = "department"


class RegionStat(BaseModel):
    code: str
    name: str
    #: The department's province. Null for provinces.
    province_name: str | None
    detection_count: int
    max_probability: float


class HourStat(BaseModel):
    #: Hour of the day, Argentina time (UTC-3).
    hour: int
    detection_count: int


class ReportStat(BaseModel):
    verdict: ReportVerdict
    category: ReportCategory | None
    report_count: int
