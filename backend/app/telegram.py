"""Telegram alerts, sent by a bot through the Telegram Bot API.

A user connects by opening a one-time link to the bot (t.me/<bot>?start=<token>,
created from their account page) and pressing Start: Telegram then sends the bot
"/start <token>", and handle_update links that chat to the user. A bot can't
message anyone who hasn't started it, so this is the only way in.

Updates reach handle_update through POST /telegram/webhook in production. Telegram
can't reach localhost, so for local runs `python -m backend.app.telegram poll`
fetches them instead. Needs TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_USERNAME; without
them Telegram is shown as unavailable and nothing is sent.
"""

import html
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Literal

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from backend.app.crud import link_telegram_chat, unlink_telegram_chat
from backend.app.mailer import AlertedDetection, format_local_time
from backend.app.tokens import hash_token, utcnow

logger = logging.getLogger(__name__)

SendResult = Literal["sent", "blocked", "failed"]

REQUEST_TIMEOUT_SECONDS = 10


def _token() -> str | None:
    load_dotenv()
    return os.getenv("TELEGRAM_BOT_TOKEN") or None


def bot_username() -> str | None:
    """The bot's @username (without the @), or None if Telegram isn't configured."""
    load_dotenv()
    if not _token():
        return None
    return os.getenv("TELEGRAM_BOT_USERNAME") or None


def webhook_secret() -> str | None:
    load_dotenv()
    return os.getenv("TELEGRAM_WEBHOOK_SECRET") or None


def _frontend_url() -> str:
    return os.getenv("FRONTEND_URL") or "http://localhost:3000"


class TelegramError(Exception):
    def __init__(self, status: int, description: str):
        super().__init__(f"{status}: {description}")
        self.status = status


def _call(method: str, payload: dict, timeout: float = REQUEST_TIMEOUT_SECONDS) -> dict | list | bool:
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{_token()}/{method}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)["result"]
    except urllib.error.HTTPError as error:
        try:
            description = json.load(error).get("description", "")
        except ValueError:
            description = ""
        raise TelegramError(error.code, description) from None


def send_message(chat_id: int, text: str) -> SendResult:
    """Send an HTML-formatted message. "blocked" means the user blocked the bot or
    deleted the chat (Telegram answers 403), so the chat should be disconnected.
    Failures are logged, never raised, so they can't fail the detection run."""
    if not _token():
        return "failed"
    try:
        _call(
            "sendMessage",
            {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "link_preview_options": {"is_disabled": True}},
        )
    except TelegramError as error:
        if error.status == 403:
            logger.info("Telegram chat %s blocked the bot", chat_id)
            return "blocked"
        logger.exception("Failed to send Telegram message to chat %s", chat_id)
        return "failed"
    except Exception:
        logger.exception("Failed to send Telegram message to chat %s", chat_id)
        return "failed"
    return "sent"


def alert_message(area_name: str, detections: list[AlertedDetection]) -> str:
    count = len(detections)
    name = html.escape(area_name)
    title = f"🔥 <b>Posible foco en {name}</b>" if count == 1 else f"🔥 <b>{count} posibles focos en {name}</b>"
    lines = [
        f"• Probabilidad {d.probability:.0%}, imagen del {format_local_time(d.image_time)}\n"
        f'  <a href="https://www.google.com/maps?q={d.lat},{d.lon}">{d.lat:.4f}, {d.lon:.4f}</a>'
        for d in detections
    ]
    return (
        f"{title}\n\n"
        + "\n".join(lines)
        + f'\n\n<a href="{_frontend_url()}/alertas">Ver y reportar en AlertaFuego</a>\n'
        "<i>Detección automática: puede ser un falso positivo.</i>"
    )


def send_alert(chat_id: int, area_name: str, detections: list[AlertedDetection]) -> SendResult:
    return send_message(chat_id, alert_message(area_name, detections))


def _chat_name(chat: dict) -> str:
    username = chat.get("username")
    return f"@{username}" if username else (chat.get("first_name") or "Telegram")


def handle_update(session: Session, update: dict) -> None:
    """React to one update from Telegram: link a chat on "/start <token>", unlink it
    on "/stop", and answer anything else with instructions. Only private chats."""
    message = update.get("message") or {}
    chat = message.get("chat") or {}
    text = (message.get("text") or "").strip()
    if chat.get("type") != "private" or not text:
        return
    chat_id = chat["id"]
    account_url = f"{_frontend_url()}/cuenta"

    command, _, argument = text.partition(" ")
    command = command.split("@")[0]
    if command == "/start" and argument:
        if link_telegram_chat(session, hash_token(argument.strip()), chat_id, _chat_name(chat), utcnow()):
            reply = (
                "✅ <b>Listo, Telegram quedó conectado.</b>\n\n"
                "Vas a recibir acá las alertas de tus zonas de AlertaFuego. "
                "Para dejar de recibirlas, escribí /stop o desconectalo desde Mi cuenta."
            )
        else:
            reply = (
                "Este enlace venció o ya se usó. Generá uno nuevo desde "
                f'<a href="{account_url}">Mi cuenta</a> en AlertaFuego.'
            )
    elif command == "/stop":
        if unlink_telegram_chat(session, chat_id):
            reply = "Listo, no vas a recibir más alertas acá. Podés volver a conectarte desde Mi cuenta."
        else:
            reply = "Este chat no está conectado a ninguna cuenta de AlertaFuego."
    else:
        reply = (
            "Soy el bot de alertas de AlertaFuego. Para recibir acá los avisos de tus zonas, "
            f'entrá a <a href="{account_url}">Mi cuenta</a> y tocá «Conectar Telegram».\n\n'
            "/stop deja de enviarte alertas."
        )
    send_message(chat_id, reply)


def _set_webhook(base_url: str) -> None:
    secret = webhook_secret()
    if not secret:
        sys.exit("Set TELEGRAM_WEBHOOK_SECRET first (any random string, also set on the backend).")
    url = base_url.rstrip("/") + "/telegram/webhook"
    _call("setWebhook", {"url": url, "secret_token": secret, "allowed_updates": ["message"]})
    _call(
        "setMyCommands",
        {"commands": [{"command": "stop", "description": "Dejar de recibir alertas"}]},
    )
    print(f"Webhook set to {url}")


def _poll() -> None:
    """Local runs: fetch updates instead of receiving them. Removes the webhook,
    since Telegram doesn't allow both; run set-webhook again when done."""
    from backend.app.db import SessionLocal

    _call("deleteWebhook", {})
    print("Polling Telegram for updates (webhook removed). Ctrl+C to stop.")
    offset = 0
    while True:
        try:
            updates = _call("getUpdates", {"offset": offset, "timeout": 30}, timeout=40)
        except (TelegramError, OSError) as error:
            print(f"getUpdates failed: {error}", file=sys.stderr)
            time.sleep(5)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            with SessionLocal() as session:
                handle_update(session, update)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not _token():
        sys.exit("Set TELEGRAM_BOT_TOKEN in .env first.")
    match sys.argv[1:]:
        case ["set-webhook", base_url]:
            _set_webhook(base_url)
        case ["poll"]:
            _poll()
        case _:
            sys.exit("Usage: python -m backend.app.telegram set-webhook <backend URL> | poll")
