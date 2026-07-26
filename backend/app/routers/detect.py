"""Detection endpoints: on-demand fetch-latest-image + run-model."""

import ee
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from backend.app.crud import save_detections
from backend.app.db import get_db
from backend.app.schemas import BBoxSchema, Detection, DetectionResponse
from model.data_pipeline.config import BBox
from model.inference.raster_fetch import NoImageAvailableError
from model.inference.service import DetectionResult, InferenceContext, run_detection
from model.training.config import default_config as default_training_config

router = APIRouter(prefix="/detect")


def _to_response(result: DetectionResult) -> DetectionResponse:
    return DetectionResponse(
        image_time=result.image_time,
        bbox=BBoxSchema(
            west=result.bbox.west, south=result.bbox.south, east=result.bbox.east, north=result.bbox.north
        ),
        threshold=result.threshold,
        chunk_count=result.chunk_count,
        detection_count=len(result.detections),
        detections=[Detection(lat=lat, lon=lon, probability=prob) for lat, lon, prob in result.detections],
    )


def _run(ctx: InferenceContext, bbox: BBox, db: Session) -> DetectionResponse:
    threshold = default_training_config().seg_threshold
    try:
        result = run_detection(ctx, bbox, threshold)
    except (ee.EEException, NoImageAvailableError) as exc:
        raise HTTPException(status_code=503, detail=f"Earth Engine unavailable: {exc}") from exc
    save_detections(db, result)
    return _to_response(result)


@router.post("/demo", response_model=DetectionResponse)
def detect_demo(request: Request, db: Session = Depends(get_db)) -> DetectionResponse:
    ctx: InferenceContext = request.app.state.inference_ctx
    return _run(ctx, ctx.cfg.demo_bbox, db)


@router.post("/argentina", response_model=DetectionResponse)
def detect_argentina(request: Request, db: Session = Depends(get_db)) -> DetectionResponse:
    ctx: InferenceContext = request.app.state.inference_ctx
    return _run(ctx, ctx.cfg.argentina_bbox, db)
