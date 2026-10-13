"""The Telegram bot's webhook: Telegram POSTs each message sent to the bot here.
Registered with `python -m backend.app.telegram set-webhook <backend URL>`."""

import hmac
import logging

from fastapi import APIRouter, Body, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.telegram import handle_update, webhook_secret

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/telegram/webhook", include_in_schema=False)
def post_telegram_webhook(
    update: dict = Body(...),
    secret: str | None = Header(None, alias="X-Telegram-Bot-Api-Secret-Token"),
    db: Session = Depends(get_db),
) -> dict:
    # Telegram sends back the secret given to setWebhook: anything without it
    # didn't come from Telegram.
    expected = webhook_secret()
    if expected is None or secret is None or not hmac.compare_digest(secret, expected):
        raise HTTPException(status_code=403)
    try:
        handle_update(db, update)
    except Exception:
        # Still 200: Telegram re-sends a failed update for hours, and one that
        # fails every time would block the ones after it.
        logger.exception("Failed to handle Telegram update %s", update.get("update_id"))
    return {}
