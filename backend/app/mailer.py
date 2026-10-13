"""Alert emails, sent through Amazon SES. Sending is skipped (and logged) when
ALERT_EMAIL_FROM isn't set, so local runs don't need AWS credentials."""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta

import boto3
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

#: Argentina has no DST, so local time is always UTC-3.
_ART_OFFSET = timedelta(hours=-3)


@dataclass(frozen=True)
class AlertedDetection:
    lat: float
    lon: float
    probability: float
    image_time: datetime  # naive UTC, as stored


def _format_detection(d: AlertedDetection) -> str:
    local = (d.image_time + _ART_OFFSET).strftime("%d/%m/%Y %H:%M")
    return (
        f"- Probabilidad {d.probability:.0%}, imagen del {local} (hora argentina)\n"
        f"  {d.lat:.4f}, {d.lon:.4f} - https://www.google.com/maps?q={d.lat},{d.lon}"
    )


def send_alert_email(to: str, area_name: str, detections: list[AlertedDetection]) -> bool:
    """Email `to` about new detections in their area. Returns whether it was sent;
    failures are logged, never raised, so they can't fail the detection run."""
    load_dotenv()
    sender = os.getenv("ALERT_EMAIL_FROM")
    if not sender:
        logger.warning("ALERT_EMAIL_FROM is not set, skipping alert email to %s", to)
        return False
    frontend_url = os.getenv("FRONTEND_URL") or "http://localhost:3000"

    count = len(detections)
    subject = (
        f"AlertaFuego: posible foco en {area_name}"
        if count == 1
        else f"AlertaFuego: {count} posibles focos en {area_name}"
    )
    body = (
        f"Se detectó actividad de fuego dentro de tu zona de alerta \"{area_name}\":\n\n"
        + "\n".join(_format_detection(d) for d in detections)
        + f"\n\nVer y reportar estas detecciones: {frontend_url}/alertas\n\n"
        "Las detecciones son automáticas y pueden ser falsos positivos. Si sabés que no es "
        "un incendio, reportalo desde la app: nos ayuda a mejorar el modelo.\n"
    )
    try:
        boto3.client("sesv2").send_email(
            FromEmailAddress=sender,
            Destination={"ToAddresses": [to]},
            Content={"Simple": {"Subject": {"Data": subject}, "Body": {"Text": {"Data": body}}}},
        )
    except Exception:
        logger.exception("Failed to send alert email to %s", to)
        return False
    return True
