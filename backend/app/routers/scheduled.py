"""Scheduled detection: EventBridge Scheduler invokes the detector Lambda every few
minutes, and the Lambda Web Adapter forwards that (non-HTTP) event here as a POST.

Only the detector function mounts this router (DETECTION_WORKER=1, see main.py): the
public API never runs detections for the schedule, so a ~50 s run never occupies the
instances serving the app, and the route doesn't exist on the public URL.

Each GOES-19 image is processed once: detection_runs.image_time is unique. The cheap
checks (another run in progress, image already processed) come before loading the
model, so a run that finds nothing new costs a few seconds.
"""

import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.alerts import process_new_detections
from backend.app.crud import save_detections
from backend.app.db import get_db
from backend.app.models import DetectionRun
from backend.app.routers.detect import inference_ctx
from backend.app.tokens import utcnow
from model.training.config import load_model_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal")

#: A run still "running" after this long died (e.g. the Lambda timed out): it no longer
#: blocks new runs. Longer than the detector's timeout.
STALE_RUN = timedelta(minutes=15)


@router.post("/scheduled-detection")
def scheduled_detection(request: Request, db: Session = Depends(get_db)) -> dict:
    """Detect fires in the latest GOES-19 image over Argentina, unless it was already
    processed or another run is still going. The event body is ignored."""
    from model.data_pipeline.ee_client import init_earth_engine
    from model.inference.raster_fetch import get_latest_goes_image, image_capture_time

    running = db.scalar(
        select(DetectionRun.id).where(
            DetectionRun.status == "running", DetectionRun.started_at > utcnow() - STALE_RUN
        )
    )
    if running is not None:
        logger.info("Skipped: run %d still in progress", running)
        return {"status": "skipped", "reason": "another run in progress"}

    init_earth_engine()
    goes_image = get_latest_goes_image()
    image_time = image_capture_time(goes_image)

    run = DetectionRun(image_time=image_time, started_at=utcnow())
    db.add(run)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        logger.info("Skipped: image %s already processed", image_time)
        return {"status": "skipped", "reason": "image already processed", "image_time": image_time.isoformat()}

    try:
        from model.inference.service import run_detection

        ctx = inference_ctx(request.app)
        threshold = load_model_config(ctx.cfg.checkpoint_dir).threshold
        result = run_detection(ctx, ctx.cfg.argentina_bbox, threshold, goes_image=goes_image)
        detection_ids = save_detections(db, result)
        alert_count = process_new_detections(db, detection_ids)
    except Exception as exc:
        db.rollback()
        run.status, run.error, run.finished_at = "error", repr(exc)[:2000], utcnow()
        db.commit()
        logger.exception("Scheduled detection failed for image %s", image_time)
        raise

    run.status, run.finished_at = "ok", utcnow()
    run.detection_count, run.alert_count = len(detection_ids), alert_count
    db.commit()
    logger.info(
        "Scheduled detection: image %s, %d detections, %d alerts", image_time, len(detection_ids), alert_count
    )
    return {
        "status": "ok",
        "image_time": image_time.isoformat(),
        "threshold": threshold,
        "detections": len(detection_ids),
        "alerts": alert_count,
    }
