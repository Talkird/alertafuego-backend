"""Shared test setup.

The environment is fixed here, before any backend module is imported: the modules
call load_dotenv(), which never overrides a variable that is already set, so the
values below win over the developer's .env. That keeps the tests off the real
database, SES, Telegram and Supabase. Variables set to "" read as "not configured".

API tests need a PostGIS database, TEST_DATABASE_URL (default: a local one named
alertafuego_test). It is migrated with Alembic once per run and emptied after each
test.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

os.environ["DATABASE_URL"] = os.getenv(
    "TEST_DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/alertafuego_test"
)
os.environ["SUPABASE_URL"] = "http://supabase.test"
os.environ["DETECT_API_KEY"] = "test-detect-key"
os.environ["DETECTION_WORKER"] = "1"  # mounts the scheduled-detection route too
os.environ["FRONTEND_URL"] = "http://frontend.test"
for name in (
    "ALERT_EMAIL_FROM",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_BOT_USERNAME",
    "TELEGRAM_WEBHOOK_SECRET",
    "DEBUG_IMAGE_DIR",
    "EARTH_ENGINE_PROJECT_ID",
):
    os.environ[name] = ""

import jwt  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import ec  # noqa: E402
from geoalchemy2.elements import WKTElement  # noqa: E402
from sqlalchemy import text  # noqa: E402

from backend.app import auth  # noqa: E402
from backend.app.db import SessionLocal, engine  # noqa: E402
from backend.app.models import (  # noqa: E402
    Detection,
    DetectionReport,
    Region,
    RegionKind,
    ReportCategory,
    ReportVerdict,
)

DETECT_KEY = os.environ["DETECT_API_KEY"]

_signing_key = ec.generate_private_key(ec.SECP256R1())
_other_key = ec.generate_private_key(ec.SECP256R1())


def make_token(
    user_id: uuid.UUID,
    email: str | None = "user@example.com",
    *,
    expires_in: timedelta = timedelta(hours=1),
    audience: str = "authenticated",
    key=None,
) -> str:
    """A Supabase-like ES256 access token."""
    payload = {"sub": str(user_id), "aud": audience, "exp": datetime.now(timezone.utc) + expires_in}
    if email is not None:
        payload["email"] = email
    return jwt.encode(payload, key or _signing_key, algorithm="ES256")


def bearer(user_id: uuid.UUID, email: str | None = "user@example.com") -> dict:
    return {"Authorization": f"Bearer {make_token(user_id, email)}"}


@pytest.fixture(autouse=True)
def _fake_jwks(monkeypatch):
    """Tokens are verified against the test key pair instead of Supabase's JWKS.
    A token signed with any other key fails verification (401)."""
    client = SimpleNamespace(get_signing_key_from_jwt=lambda token: SimpleNamespace(key=_signing_key.public_key()))
    monkeypatch.setattr(auth, "_get_jwks_client", lambda: client)


@pytest.fixture(scope="session")
def other_key():
    return _other_key


@pytest.fixture(scope="session")
def migrated_db():
    """Apply the Alembic migrations once: the tests run against the real schema,
    constraints and PostGIS functions."""
    from alembic import command
    from alembic.config import Config

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"The PostGIS test database isn't reachable ({os.environ['DATABASE_URL']}): {exc}")
    # Not Config("alembic.ini"): env.py would then run its logging fileConfig, which
    # disables the loggers already created and breaks caplog in later tests.
    config = Config()
    config.set_main_option("script_location", "backend/alembic")
    command.upgrade(config, "head")
    yield


_TABLES = "alert_notifications, alert_areas, detection_reports, detection_runs, detections, regions, user_settings"


@pytest.fixture
def db(migrated_db):
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        with engine.begin() as connection:
            connection.execute(text(f"TRUNCATE {_TABLES} RESTART IDENTITY CASCADE"))


@pytest.fixture
def client(db):
    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def user_id():
    return uuid.uuid4()


@pytest.fixture
def add_detection(db):
    """Insert a detection and return it."""

    def _add(
        lat: float = -31.4,
        lon: float = -64.2,
        probability: float = 0.8,
        image_time: datetime = datetime(2026, 10, 1, 15, 0),
        image: bytes | None = None,
    ) -> Detection:
        detection = Detection(
            location=WKTElement(f"POINT({lon} {lat})", srid=4326),
            probability=probability,
            image_time=image_time,
            bbox_west=-73.6,
            bbox_south=-55.1,
            bbox_east=-53.6,
            bbox_north=-21.8,
            threshold=0.5,
            image=image,
        )
        db.add(detection)
        db.commit()
        db.refresh(detection)
        return detection

    return _add


@pytest.fixture
def add_report(db):
    def _add(detection_id: int, user: uuid.UUID, verdict: str = "confirmed_fire", category: str | None = None):
        report = DetectionReport(
            detection_id=detection_id,
            user_id=user,
            verdict=ReportVerdict(verdict),
            category=ReportCategory(category) if category else None,
        )
        db.add(report)
        db.commit()
        return report

    return _add


def square(west: float, south: float, east: float, north: float) -> str:
    return f"MULTIPOLYGON((({west} {south},{east} {south},{east} {north},{west} {north},{west} {south})))"


@pytest.fixture
def add_region(db):
    """Insert a province or department with a rectangular boundary."""

    def _add(code: str, name: str, bounds: tuple[float, float, float, float], province_code: str | None = None):
        region = Region(
            kind=RegionKind.PROVINCE if len(code) == 2 else RegionKind.DEPARTMENT,
            code=code,
            name=name,
            province_code=province_code,
            geometry=WKTElement(square(*bounds), srid=4326),
        )
        db.add(region)
        db.commit()
        return region

    return _add
