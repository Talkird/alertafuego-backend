"""SQLAlchemy ORM models."""

import enum
import uuid as uuid_module
from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import CheckConstraint, DateTime, Enum, Float, ForeignKey, LargeBinary, Text, UniqueConstraint, func
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
    #: PNG thumbnail (infrared band, cropped around the detected pixel). Deferred so
    #: bulk queries like list_detections() don't pull image bytes for every row.
    image: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True, deferred=True)


class ReportVerdict(str, enum.Enum):
    FALSE_POSITIVE = "false_positive"
    CONFIRMED_FIRE = "confirmed_fire"


class ReportCategory(str, enum.Enum):
    """Only meaningful when verdict is FALSE_POSITIVE - null otherwise."""

    INDUSTRIAL_ACTIVITY = "industrial_activity"
    CONTROLLED_BURN = "controlled_burn"
    GAS_FLARE = "gas_flare"
    SUN_GLINT = "sun_glint"
    SENSOR_NOISE = "sensor_noise"
    OTHER = "other"


class DetectionReport(Base):
    """A user's verdict on a detection - either a false-positive report (with a
    category explaining why) or a confirmation that it's a real fire."""

    __tablename__ = "detection_reports"
    __table_args__ = (
        UniqueConstraint("detection_id", "user_id", name="uq_report_detection_user"),
        CheckConstraint(
            "(verdict = 'false_positive' AND category IS NOT NULL) OR "
            "(verdict = 'confirmed_fire' AND category IS NULL)",
            name="ck_category_matches_verdict",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    detection_id: Mapped[int] = mapped_column(ForeignKey("detections.id"), nullable=False, index=True)
    user_id: Mapped[uuid_module.UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    verdict: Mapped[ReportVerdict] = mapped_column(
        Enum(ReportVerdict, name="report_verdict", values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
    )
    category: Mapped[ReportCategory | None] = mapped_column(
        Enum(ReportCategory, name="report_category", values_callable=lambda obj: [e.value for e in obj]),
        nullable=True,
    )
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
