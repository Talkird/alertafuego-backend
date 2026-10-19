"""Detection endpoints: on-demand fetch-latest-image + run-model.

torch, Earth Engine and the inference code are imported only when a detection
runs, so importing this router (at every cold start) stays cheap."""

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from sqlalchemy.orm import Session

from backend.app.alerts import process_new_detections
from backend.app.auth import require_detect_key
from backend.app.crud import save_detections
from backend.app.db import get_db
from backend.app.schemas import BBoxSchema, Detection, DetectionResponse
from model.data_pipeline.config import BBox
from model.training.config import load_model_config

if TYPE_CHECKING:
    from model.inference.service import DetectionResult, InferenceContext

# Router-level dependencies run before the endpoint's own, so a request without
# the key is rejected before it can trigger loading the model.
router = APIRouter(prefix="/detect", dependencies=[Depends(require_detect_key)])

_load_lock = threading.Lock()


def inference_ctx(app: FastAPI) -> "InferenceContext":
    """The inference context (Earth Engine auth, model, normalization stats), loaded
    by the first detection run on this instance and kept for the next ones. The
    lock keeps two simultaneous first runs from loading it twice."""
    ctx = getattr(app.state, "inference_ctx", None)
    if ctx is None:
        with _load_lock:
            ctx = getattr(app.state, "inference_ctx", None)
            if ctx is None:
                from model.inference.service import load_context

                ctx = app.state.inference_ctx = load_context()
    return ctx


def get_inference_ctx(request: Request) -> "InferenceContext":
    return inference_ctx(request.app)


def _debug_image_dir() -> Path | None:
    """Opt-in local PNG dump of each detection run, for visually sanity-checking
    the model before it's presentable. Set DEBUG_IMAGE_DIR in .env to enable."""
    load_dotenv()
    raw = os.getenv("DEBUG_IMAGE_DIR")
    return Path(raw) if raw else None


def _to_response(result: "DetectionResult") -> DetectionResponse:
    return DetectionResponse(
        image_time=result.image_time,
        bbox=BBoxSchema(
            west=result.bbox.west, south=result.bbox.south, east=result.bbox.east, north=result.bbox.north
        ),
        threshold=result.threshold,
        chunk_count=result.chunk_count,
        detection_count=len(result.detections),
        detections=[Detection(lat=lat, lon=lon, probability=prob) for lat, lon, prob, _ in result.detections],
    )


def _run(ctx: "InferenceContext", bbox: BBox, db: Session) -> DetectionResponse:
    import ee

    from model.inference.raster_fetch import NoImageAvailableError
    from model.inference.service import run_detection

    # The threshold chosen on val when the model was trained.
    threshold = load_model_config(ctx.cfg.checkpoint_dir).threshold
    try:
        result = run_detection(ctx, bbox, threshold, debug_image_dir=_debug_image_dir())
    except (ee.EEException, NoImageAvailableError) as exc:
        raise HTTPException(status_code=503, detail=f"Earth Engine unavailable: {exc}") from exc
    detection_ids = save_detections(db, result)
    process_new_detections(db, detection_ids)
    return _to_response(result)


@router.post("/demo", response_model=DetectionResponse)
def detect_demo(
    ctx: "InferenceContext" = Depends(get_inference_ctx), db: Session = Depends(get_db)
) -> DetectionResponse:
    return _run(ctx, ctx.cfg.demo_bbox, db)


@router.post("/argentina", response_model=DetectionResponse)
def detect_argentina(
    ctx: "InferenceContext" = Depends(get_inference_ctx), db: Session = Depends(get_db)
) -> DetectionResponse:
    return _run(ctx, ctx.cfg.argentina_bbox, db)
