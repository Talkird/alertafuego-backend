"""Pydantic response models for the detection endpoints."""

from datetime import datetime

from pydantic import BaseModel


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
