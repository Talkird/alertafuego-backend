"""SQLAlchemy ORM models."""

from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import DateTime, Float, func
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
