"""Pydantic response models for the detection endpoints."""

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, model_validator


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
