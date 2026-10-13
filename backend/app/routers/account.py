"""Account endpoints: the user's alert channels. Email goes to their account's
address or to another one they confirm through an emailed link; Telegram is
connected by opening a one-time link to the bot (see backend.app.telegram)."""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.auth import UserWithEmail, get_current_user_with_email
from backend.app.crud import (
    cancel_pending_alert_email,
    confirm_alert_email,
    get_user_settings,
    reset_alert_email,
    set_pending_alert_email,
    set_telegram_link_token,
    unlink_telegram,
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
    TelegramLink,
)
from backend.app.telegram import bot_username, send_message
from backend.app.tokens import hash_token, new_token, utcnow

router = APIRouter()

CONFIRMATION_TTL = timedelta(hours=24)
#: The bot link is opened right away, from the same screen.
TELEGRAM_LINK_TTL = timedelta(minutes=30)


def _to_response(settings: UserSettings | None, account_email: str) -> AccountSettings:
    alert_email = settings.alert_email if settings else None
    pending = bool(settings and settings.pending_email and settings.pending_expires_at > utcnow())
    return AccountSettings(
        email_alerts=settings.email_alerts if settings else True,
        alert_email=alert_email or account_email,
        uses_account_email=alert_email is None,
        pending_email=settings.pending_email if pending else None,
        telegram_available=bot_username() is not None,
        telegram_connected=bool(settings and settings.telegram_chat_id is not None),
        telegram_name=settings.telegram_name if settings else None,
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
    return _to_response(update_user_settings(db, user.id, payload.model_dump()), user.email)


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

    token = new_token()
    if not send_confirmation_email(email, token, int(CONFIRMATION_TTL.total_seconds() // 3600)):
        raise HTTPException(status_code=502, detail="Couldn't send the confirmation email")
    settings = set_pending_alert_email(db, user.id, email, hash_token(token), utcnow() + CONFIRMATION_TTL)
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
    email = confirm_alert_email(db, hash_token(payload.token), utcnow())
    if email is None:
        raise HTTPException(status_code=400, detail="The link is invalid or has expired")
    return AlertEmailConfirmed(email=email)


@router.post("/account/telegram/link", response_model=TelegramLink)
def post_telegram_link(
    user: UserWithEmail = Depends(get_current_user_with_email), db: Session = Depends(get_db)
) -> TelegramLink:
    """A one-time link to the bot. Pressing Start in Telegram connects that chat."""
    username = bot_username()
    if username is None:
        raise HTTPException(status_code=503, detail="Telegram is not configured")
    token = new_token()
    expires_at = utcnow() + TELEGRAM_LINK_TTL
    set_telegram_link_token(db, user.id, hash_token(token), expires_at)
    return TelegramLink(url=f"https://t.me/{username}?start={token}", expires_at=expires_at)


@router.delete("/account/telegram", response_model=AccountSettings)
def delete_telegram(
    user: UserWithEmail = Depends(get_current_user_with_email), db: Session = Depends(get_db)
) -> AccountSettings:
    settings = get_user_settings(db, user.id)
    if settings and settings.telegram_chat_id is not None:
        send_message(
            settings.telegram_chat_id, "Desconectaste Telegram desde AlertaFuego: no vas a recibir más alertas acá."
        )
    return _to_response(unlink_telegram(db, user.id), user.email)


@router.post("/account/telegram/test", status_code=204)
def post_telegram_test(user: UserWithEmail = Depends(get_current_user_with_email), db: Session = Depends(get_db)) -> None:
    """Send a test message, so the user can check that alerts will arrive."""
    settings = get_user_settings(db, user.id)
    if settings is None or settings.telegram_chat_id is None:
        raise HTTPException(status_code=409, detail="Telegram is not connected")
    result = send_message(
        settings.telegram_chat_id, "👋 Mensaje de prueba de AlertaFuego: las alertas de tus zonas van a llegar acá."
    )
    if result == "blocked":
        unlink_telegram(db, user.id)
        raise HTTPException(status_code=409, detail="The bot was blocked in Telegram")
    if result == "failed":
        raise HTTPException(status_code=502, detail="Couldn't send the Telegram message")
