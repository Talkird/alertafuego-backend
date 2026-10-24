"""Functional tests: account alert channels (email address change with confirmation
link, Telegram connection) and the Telegram bot webhook."""

from datetime import timedelta
from unittest.mock import MagicMock

import pytest

from backend.app import telegram
from backend.app.routers import account
from backend.app.tokens import utcnow
from tests.conftest import bearer

pytestmark = pytest.mark.db


@pytest.fixture
def bot(monkeypatch):
    """A configured Telegram bot whose API calls are recorded, not sent."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")
    monkeypatch.setenv("TELEGRAM_BOT_USERNAME", "AlertaFuegoTestBot")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "s3cret")
    call = MagicMock(return_value=True)
    monkeypatch.setattr(telegram, "_call", call)
    return call


@pytest.fixture
def fixed_token(monkeypatch):
    monkeypatch.setattr(account, "new_token", lambda: "fixed-token")
    return "fixed-token"


def test_account_defaults(client, user_id):
    body = client.get("/account", headers=bearer(user_id, "yo@example.com")).json()
    assert body == {
        "email_alerts": True,
        "alert_email": "yo@example.com",
        "uses_account_email": True,
        "pending_email": None,
        "telegram_available": False,
        "telegram_connected": False,
        "telegram_name": None,
    }


def test_account_requires_sign_in(client):
    assert client.get("/account").status_code == 401
    assert client.patch("/account", json={"email_alerts": False}).status_code == 401


def test_turn_email_alerts_off(client, user_id):
    response = client.patch("/account", json={"email_alerts": False}, headers=bearer(user_id))
    assert response.json()["email_alerts"] is False
    assert client.patch("/account", json={}, headers=bearer(user_id)).status_code == 422


def test_change_alert_email_with_confirmation(client, user_id, fixed_token):
    headers = bearer(user_id, "yo@example.com")
    pending = client.post("/account/alert-email", json={"email": "Otro@Example.com"}, headers=headers).json()
    assert pending["pending_email"] == "otro@example.com"
    assert pending["alert_email"] == "yo@example.com"  # unchanged until confirmed

    confirmed = client.post("/account/alert-email/confirm", json={"token": fixed_token})
    assert confirmed.json() == {"email": "otro@example.com"}

    body = client.get("/account", headers=headers).json()
    assert body["alert_email"] == "otro@example.com" and body["uses_account_email"] is False
    # The link works once.
    assert client.post("/account/alert-email/confirm", json={"token": fixed_token}).status_code == 400


def test_confirm_with_unknown_or_expired_token(client, db, user_id, fixed_token):
    assert client.post("/account/alert-email/confirm", json={"token": "nope"}).status_code == 400
    assert client.post("/account/alert-email/confirm", json={"token": ""}).status_code == 422

    client.post("/account/alert-email", json={"email": "otro@example.com"}, headers=bearer(user_id))
    from backend.app.crud import get_user_settings

    settings = get_user_settings(db, user_id)
    settings.pending_expires_at = utcnow() - timedelta(minutes=1)
    db.commit()
    assert client.post("/account/alert-email/confirm", json={"token": fixed_token}).status_code == 400


def test_alert_email_back_to_account_and_cancel_pending(client, user_id, fixed_token):
    headers = bearer(user_id, "yo@example.com")
    client.post("/account/alert-email", json={"email": "otro@example.com"}, headers=headers)
    client.post("/account/alert-email/confirm", json={"token": fixed_token})

    client.post("/account/alert-email", json={"email": "tercero@example.com"}, headers=headers)
    cancelled = client.delete("/account/alert-email/pending", headers=headers).json()
    assert cancelled["pending_email"] is None and cancelled["alert_email"] == "otro@example.com"

    reset = client.delete("/account/alert-email", headers=headers).json()
    assert reset["alert_email"] == "yo@example.com" and reset["uses_account_email"] is True


def test_alert_email_same_as_account_needs_no_confirmation(client, user_id):
    body = client.post("/account/alert-email", json={"email": "YO@example.com"}, headers=bearer(user_id, "yo@example.com")).json()
    assert body["uses_account_email"] is True and body["pending_email"] is None


def test_alert_email_send_failure_is_502(client, user_id, monkeypatch):
    monkeypatch.setattr(account, "send_confirmation_email", lambda *args: False)
    response = client.post("/account/alert-email", json={"email": "otro@example.com"}, headers=bearer(user_id))
    assert response.status_code == 502


def test_alert_email_validation(client, user_id):
    assert client.post("/account/alert-email", json={"email": "nope"}, headers=bearer(user_id)).status_code == 422
    # Surrounding spaces fail the schema's pattern before the endpoint's strip() runs
    # (defect D-01 in docs/PRUEBAS.md: the frontend trims before sending).
    padded = client.post("/account/alert-email", json={"email": " otro@example.com "}, headers=bearer(user_id))
    assert padded.status_code == 422


def test_telegram_link_unavailable_without_bot(client, user_id):
    assert client.post("/account/telegram/link", headers=bearer(user_id)).status_code == 503


def test_connect_telegram_end_to_end(client, user_id, bot, monkeypatch):
    monkeypatch.setattr(account, "new_token", lambda: "link-token")
    link = client.post("/account/telegram/link", headers=bearer(user_id)).json()
    assert link["url"] == "https://t.me/AlertaFuegoTestBot?start=link-token"

    update = {"message": {"chat": {"id": 777, "type": "private", "username": "vecino"}, "text": "/start link-token"}}
    response = client.post("/telegram/webhook", json=update, headers={"X-Telegram-Bot-Api-Secret-Token": "s3cret"})
    assert response.status_code == 200
    assert "conectado" in bot.call_args.args[1]["text"]

    body = client.get("/account", headers=bearer(user_id)).json()
    assert body["telegram_available"] is True and body["telegram_connected"] is True
    assert body["telegram_name"] == "@vecino"

    # The link is single-use.
    client.post("/telegram/webhook", json=update, headers={"X-Telegram-Bot-Api-Secret-Token": "s3cret"})
    assert "venció" in bot.call_args.args[1]["text"]


def _connect(client, user_id, bot, monkeypatch, chat_id=777):
    monkeypatch.setattr(account, "new_token", lambda: "link-token")
    client.post("/account/telegram/link", headers=bearer(user_id))
    update = {"message": {"chat": {"id": chat_id, "type": "private"}, "text": "/start link-token"}}
    client.post("/telegram/webhook", json=update, headers={"X-Telegram-Bot-Api-Secret-Token": "s3cret"})


def test_telegram_test_message(client, user_id, bot, monkeypatch):
    assert client.post("/account/telegram/test", headers=bearer(user_id)).status_code == 409
    _connect(client, user_id, bot, monkeypatch)
    assert client.post("/account/telegram/test", headers=bearer(user_id)).status_code == 204
    assert bot.call_args.args[1]["chat_id"] == 777


@pytest.mark.parametrize("result,status", [("blocked", 409), ("failed", 502)])
def test_telegram_test_message_failures(client, user_id, bot, monkeypatch, result, status):
    _connect(client, user_id, bot, monkeypatch)
    monkeypatch.setattr(account, "send_message", lambda chat_id, text: result)
    assert client.post("/account/telegram/test", headers=bearer(user_id)).status_code == status
    connected = client.get("/account", headers=bearer(user_id)).json()["telegram_connected"]
    assert connected is (result != "blocked")


def test_disconnect_telegram(client, user_id, bot, monkeypatch):
    _connect(client, user_id, bot, monkeypatch)
    body = client.delete("/account/telegram", headers=bearer(user_id)).json()
    assert body["telegram_connected"] is False
    assert "Desconectaste" in bot.call_args.args[1]["text"]


def test_webhook_rejects_missing_or_wrong_secret(client, bot):
    update = {"message": {"chat": {"id": 1, "type": "private"}, "text": "hola"}}
    assert client.post("/telegram/webhook", json=update).status_code == 403
    assert client.post("/telegram/webhook", json=update, headers={"X-Telegram-Bot-Api-Secret-Token": "x"}).status_code == 403
    bot.assert_not_called()


def test_webhook_rejects_when_unconfigured(client):
    response = client.post("/telegram/webhook", json={}, headers={"X-Telegram-Bot-Api-Secret-Token": ""})
    assert response.status_code == 403


def test_webhook_stop_and_help(client, user_id, bot, monkeypatch):
    headers = {"X-Telegram-Bot-Api-Secret-Token": "s3cret"}
    _connect(client, user_id, bot, monkeypatch, chat_id=555)

    client.post("/telegram/webhook", json={"message": {"chat": {"id": 555, "type": "private"}, "text": "/stop"}}, headers=headers)
    assert "no vas a recibir más alertas" in bot.call_args.args[1]["text"]
    assert client.get("/account", headers=bearer(user_id)).json()["telegram_connected"] is False

    client.post("/telegram/webhook", json={"message": {"chat": {"id": 555, "type": "private"}, "text": "hola"}}, headers=headers)
    assert "Conectar Telegram" in bot.call_args.args[1]["text"]


def test_webhook_ignores_group_chats(client, bot):
    update = {"message": {"chat": {"id": -100, "type": "group"}, "text": "/start x"}}
    response = client.post("/telegram/webhook", json=update, headers={"X-Telegram-Bot-Api-Secret-Token": "s3cret"})
    assert response.status_code == 200
    bot.assert_not_called()


def test_webhook_answers_200_when_handling_fails(client, bot, monkeypatch):
    from backend.app.routers import telegram as telegram_router

    def boom(db, update):
        raise RuntimeError("bug")

    monkeypatch.setattr(telegram_router, "handle_update", boom)
    response = client.post("/telegram/webhook", json={"update_id": 1}, headers={"X-Telegram-Bot-Api-Secret-Token": "s3cret"})
    assert response.status_code == 200
