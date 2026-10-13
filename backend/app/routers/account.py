"""Account endpoints: the user's alert address (their account's, or another one they
confirm through an emailed link) and the starting values for their new alert areas."""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.auth import UserWithEmail, get_current_user_with_email
from backend.app.crud import (
    cancel_pending_alert_email,
    confirm_alert_email,
    get_user_settings,
    reset_alert_email,
    set_pending_alert_email,
    update_user_settings,
)
from backend.app.db import get_db
from backend.app.mailer import send_confirmation_email
from backend.app.models import UserSettings
from backend.app.schemas import (
    AccountSettings,
    AccountSettingsUpdate,
    AlertEmailChange,
    AlertEmailConfirm,
    AlertEmailConfirmed,
)

router = APIRouter()

CONFIRMATION_TTL = timedelta(hours=24)

#: Used until the user saves their own (the same values the area form starts from).
DEFAULT_RADIUS_KM = 25.0
DEFAULT_MIN_PROBABILITY = 0.65


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _to_response(settings: UserSettings | None, account_email: str) -> AccountSettings:
    alert_email = settings.alert_email if settings else None
    pending = bool(settings and settings.pending_email and settings.pending_expires_at > _now())
    return AccountSettings(
        alert_email=alert_email or account_email,
        uses_account_email=alert_email is None,
        pending_email=settings.pending_email if pending else None,
        default_radius_km=settings.default_radius_km if settings else DEFAULT_RADIUS_KM,
        default_min_probability=settings.default_min_probability if settings else DEFAULT_MIN_PROBABILITY,
    )


@router.get("/account", response_model=AccountSettings)
def get_account(user: UserWithEmail = Depends(get_current_user_with_email), db: Session = Depends(get_db)) -> AccountSettings:
    return _to_response(get_user_settings(db, user.id), user.email)


@router.patch("/account", response_model=AccountSettings)
def patch_account(
    payload: AccountSettingsUpdate,
    user: UserWithEmail = Depends(get_current_user_with_email),
    db: Session = Depends(get_db),
) -> AccountSettings:
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    if any(value is None for value in updates.values()):
        raise HTTPException(status_code=400, detail="Fields can't be set to null")
    return _to_response(update_user_settings(db, user.id, updates), user.email)


@router.post("/account/alert-email", response_model=AccountSettings)
def post_alert_email(
    payload: AlertEmailChange,
    user: UserWithEmail = Depends(get_current_user_with_email),
    db: Session = Depends(get_db),
) -> AccountSettings:
    """Ask to receive alerts at another address. It takes effect once the link
    emailed to it is opened. The account's own address needs no confirmation."""
    email = payload.email.strip().lower()
    if email == user.email.lower():
        return _to_response(reset_alert_email(db, user.id), user.email)

    token = secrets.token_urlsafe(32)
    if not send_confirmation_email(email, token, int(CONFIRMATION_TTL.total_seconds() // 3600)):
        raise HTTPException(status_code=502, detail="Couldn't send the confirmation email")
    settings = set_pending_alert_email(db, user.id, email, _hash_token(token), _now() + CONFIRMATION_TTL)
    return _to_response(settings, user.email)


@router.delete("/account/alert-email", response_model=AccountSettings)
def delete_alert_email(
    user: UserWithEmail = Depends(get_current_user_with_email), db: Session = Depends(get_db)
) -> AccountSettings:
    """Go back to the account's address, cancelling any pending change."""
    return _to_response(reset_alert_email(db, user.id), user.email)


@router.delete("/account/alert-email/pending", response_model=AccountSettings)
def delete_pending_alert_email(
    user: UserWithEmail = Depends(get_current_user_with_email), db: Session = Depends(get_db)
) -> AccountSettings:
    """Cancel a change still waiting for confirmation (its link stops working)."""
    return _to_response(cancel_pending_alert_email(db, user.id), user.email)


@router.post("/account/alert-email/confirm", response_model=AlertEmailConfirmed)
def post_confirm_alert_email(payload: AlertEmailConfirm, db: Session = Depends(get_db)) -> AlertEmailConfirmed:
    """Open from the emailed link. No login needed: the token proves access to the
    inbox, and the link may well be opened on another device."""
    email = confirm_alert_email(db, _hash_token(payload.token), _now())
    if email is None:
        raise HTTPException(status_code=400, detail="The link is invalid or has expired")
    return AlertEmailConfirmed(email=email)
