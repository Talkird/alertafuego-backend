"""SQLAlchemy ORM models."""

import enum
import uuid as uuid_module
from datetime import datetime

from geoalchemy2 import Geography, Geometry
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    LargeBinary,
    Text,
    UniqueConstraint,
    func,
)
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


class AlertArea(Base):
    """A user's alert zone: a circle (center + radius). Every new detection inside it
    with probability >= min_probability raises an alert, emailed to `email`."""

    __tablename__ = "alert_areas"
    __table_args__ = (
        CheckConstraint("radius_km > 0 AND radius_km <= 200", name="ck_alert_area_radius"),
        CheckConstraint("min_probability >= 0 AND min_probability <= 1", name="ck_alert_area_min_probability"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[uuid_module.UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False, index=True)
    #: Copied from the user's JWT when the area is created or edited - the backend has
    #: no other way to look up a Supabase user's email.
    email: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    #: Geography (not Geometry) so ST_DWithin takes the radius in meters directly.
    center = mapped_column(Geography("POINT", srid=4326, spatial_index=True), nullable=False)
    radius_km: Mapped[float] = mapped_column(Float, nullable=False)
    min_probability: Mapped[float] = mapped_column(Float, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class AlertNotification(Base):
    """One alert: a detection that fell inside an alert area. Doubles as the user's
    in-app alert history and as the cooldown record that keeps one fire, seen again
    every run, from alerting over and over."""

    __tablename__ = "alert_notifications"
    __table_args__ = (UniqueConstraint("area_id", "detection_id", name="uq_alert_area_detection"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    area_id: Mapped[int] = mapped_column(ForeignKey("alert_areas.id", ondelete="CASCADE"), nullable=False, index=True)
    detection_id: Mapped[int] = mapped_column(ForeignKey("detections.id"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    #: Null when the email wasn't sent (no sender configured, or SES failed).
    emailed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class RegionKind(str, enum.Enum):
    PROVINCE = "province"
    DEPARTMENT = "department"


class Region(Base):
    """An administrative boundary from IGN (province or department), for per-region
    statistics. Loaded by `python -m backend.scripts.load_regions`."""

    __tablename__ = "regions"
    __table_args__ = (UniqueConstraint("kind", "code", name="uq_region_kind_code"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    kind: Mapped[RegionKind] = mapped_column(
        Enum(RegionKind, name="region_kind", values_callable=lambda obj: [e.value for e in obj]),
        nullable=False,
    )
    #: IGN `in1` code: 2 digits for a province, 5 for a department (the first 2 are
    #: its province's code).
    code: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    #: Null for provinces. For departments, the parent province's `code`.
    province_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    geometry = mapped_column(Geometry("MULTIPOLYGON", srid=4326, spatial_index=True), nullable=False)
