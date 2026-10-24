"""Functional tests: on-demand detection runs (POST /detect/*) and the scheduled
detection. The model and Earth Engine are replaced by a fake run_detection, so
these check the API around inference: auth, persistence, alerts and run records."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import ee
import pytest

from backend.app.models import Detection, DetectionRun
from backend.app.routers import detect, scheduled
from model.data_pipeline.config import BBox
from model.inference import raster_fetch, service
from model.inference.config import default_config
from model.inference.service import DetectionResult
from model.training.config import ModelConfig
from tests.conftest import DETECT_KEY

pytestmark = pytest.mark.db

KEY = {"X-API-Key": DETECT_KEY}
IMAGE_TIME = datetime(2026, 10, 1, 15, 0)


def _result(bbox: BBox, threshold: float) -> DetectionResult:
    return DetectionResult(
        image_time=IMAGE_TIME,
        bbox=bbox,
        threshold=threshold,
        chunk_count=1,
        detections=[(-30.0, -67.0, 0.91, b"png1"), (-29.5, -66.5, 0.55, b"png2")],
    )


@pytest.fixture
def fake_inference(client, monkeypatch):
    """No model, no Earth Engine: run_detection returns two fixed detections."""
    cfg = default_config(checkpoint_dir=Path("/nonexistent"))
    ctx = SimpleNamespace(cfg=cfg)
    calls = []

    def run_detection(ctx, bbox, threshold, **kwargs):
        calls.append((bbox, threshold, kwargs))
        return _result(bbox, threshold)

    client.app.dependency_overrides[detect.get_inference_ctx] = lambda: ctx
    monkeypatch.setattr(scheduled, "inference_ctx", lambda app: ctx)
    monkeypatch.setattr(service, "run_detection", run_detection)
    model_cfg = ModelConfig(base_channels=16, depth=2, threshold=0.42)
    monkeypatch.setattr(detect, "load_model_config", lambda path: model_cfg)
    monkeypatch.setattr(scheduled, "load_model_config", lambda path: model_cfg)
    yield SimpleNamespace(ctx=ctx, calls=calls)
    client.app.dependency_overrides.clear()


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_detect_requires_api_key(client, headers):
    assert client.post("/detect/demo", headers=headers).status_code == 401
    assert client.post("/detect/argentina", headers=headers).status_code == 401


def test_detect_unconfigured_key_is_503(client, monkeypatch):
    monkeypatch.setenv("DETECT_API_KEY", "")
    assert client.post("/detect/demo", headers=KEY).status_code == 503


def test_detect_demo_saves_detections(client, db, fake_inference):
    response = client.post("/detect/demo", headers=KEY)

    assert response.status_code == 200
    body = response.json()
    assert body["detection_count"] == 2 and body["threshold"] == 0.42
    assert body["bbox"] == {"west": -68.0, "south": -31.0, "east": -66.0, "north": -29.0}
    assert fake_inference.calls[0][0] == fake_inference.ctx.cfg.demo_bbox
    stored = client.get("/detections").json()
    assert len(stored) == 2 and all(d["has_image"] for d in stored)


def test_detect_argentina_uses_country_bbox(client, fake_inference):
    assert client.post("/detect/argentina", headers=KEY).status_code == 200
    assert fake_inference.calls[0][0] == fake_inference.ctx.cfg.argentina_bbox


@pytest.mark.parametrize("error", [ee.EEException("quota"), raster_fetch.NoImageAvailableError("none")])
def test_detect_earth_engine_unavailable_is_503(client, db, fake_inference, monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(service, "run_detection", fail)
    response = client.post("/detect/demo", headers=KEY)
    assert response.status_code == 503
    assert db.query(Detection).count() == 0


@pytest.fixture
def latest_image(monkeypatch):
    from model.data_pipeline import ee_client

    monkeypatch.setattr(ee_client, "init_earth_engine", lambda: None)
    monkeypatch.setattr(raster_fetch, "get_latest_goes_image", lambda: "goes-image")
    monkeypatch.setattr(raster_fetch, "image_capture_time", lambda image: IMAGE_TIME)


def test_scheduled_detection_processes_each_image_once(client, db, fake_inference, latest_image):
    first = client.post("/internal/scheduled-detection", json={}).json()
    assert first == {
        "status": "ok",
        "image_time": IMAGE_TIME.isoformat(),
        "threshold": 0.42,
        "detections": 2,
        "alerts": 0,
    }
    assert fake_inference.calls[0][2] == {"goes_image": "goes-image"}

    second = client.post("/internal/scheduled-detection", json={}).json()
    assert second["status"] == "skipped" and second["reason"] == "image already processed"

    run = db.query(DetectionRun).one()
    assert (run.status, run.detection_count, run.alert_count) == ("ok", 2, 0)


def test_scheduled_detection_skips_while_another_runs(client, db, fake_inference, latest_image):
    from backend.app.tokens import utcnow

    db.add(DetectionRun(image_time=datetime(2026, 9, 1), started_at=utcnow()))
    db.commit()
    body = client.post("/internal/scheduled-detection", json={}).json()
    assert body == {"status": "skipped", "reason": "another run in progress"}
    assert fake_inference.calls == []


def test_scheduled_detection_records_failures(client, db, fake_inference, latest_image, monkeypatch):
    from fastapi.testclient import TestClient

    def fail(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(service, "run_detection", fail)
    response = TestClient(client.app, raise_server_exceptions=False).post("/internal/scheduled-detection", json={})

    assert response.status_code == 500
    db.expire_all()
    run = db.query(DetectionRun).one()
    assert run.status == "error" and "model exploded" in run.error and run.finished_at is not None
