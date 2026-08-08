"""SQLAlchemy ORM models."""

import enum
import uuid as uuid_module
from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import DateTime, Enum, Float, ForeignKey, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db import Base


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    location = mapped_column(Geometry("POINT", srid=4326, spatial_index=True), nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    image_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    bbox_west: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_south: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_east: Mapped[float] = mapped_column(Float, nullable=False)
    bbox_north: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)


class ReportCategory(str, enum.Enum):
    INDUSTRIAL_ACTIVITY = "industrial_activity"
    CONTROLLED_BURN = "controlled_burn"
    GAS_FLARE = "gas_flare"
    SUN_GLINT = "sun_glint"
    SENSOR_NOISE = "sensor_noise"
    OTHER = "other"


class FalsePositiveReport(Base):
    __tablename__ = "false_positive_reports"
    __table_args__ = (UniqueConstraint("detection_id", "user_id", name="uq_report_detection_user"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    detection_id: Mapped[int] = mapped_column(ForeignKey("detections.id"), nullable=False, index=True)
    user_id: Mapped[uuid_module.UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    category: Mapped[ReportCategory] = mapped_column(
        Enum(ReportCategory, name="report_category", values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
    )
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
