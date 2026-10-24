"""Unit tests for backend/app logic that needs no database: one-time tokens, request
schema validation, alert message formatting, email sending and the API-key check."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend.app import alerts, auth, mailer, telegram
from backend.app.mailer import AlertedDetection
from backend.app.schemas import AlertAreaCreate, AlertAreaUpdate, AlertEmailChange, ReportCreate
from backend.app.tokens import hash_token, new_token, utcnow


def test_new_token_is_unique_and_fits_telegram_start_parameter():
    tokens = {new_token() for _ in range(100)}
    assert len(tokens) == 100
    assert all(len(token) == 32 and len(token) <= 64 for token in tokens)


def test_hash_token_is_sha256_and_deterministic():
    assert hash_token("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert hash_token("abc") != hash_token("abd")


def test_utcnow_is_naive():
    assert utcnow().tzinfo is None


def test_report_false_positive_requires_category():
    with pytest.raises(ValidationError, match="category is required"):
        ReportCreate(verdict="false_positive")
    assert ReportCreate(verdict="false_positive", category="gas_flare").category.value == "gas_flare"


def test_report_confirmed_fire_rejects_category():
    with pytest.raises(ValidationError, match="not applicable"):
        ReportCreate(verdict="confirmed_fire", category="gas_flare")


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "x"},  # no shape
        {"name": "x", "lat": -31, "lon": -64},  # circle without radius
        {"name": "x", "lat": -31, "lon": -64, "radius_km": 10, "region_code": "14"},  # two shapes
        {"name": "x", "region_code": "123"},  # not 2 or 5 digits
        {"name": "x", "lat": -31, "lon": -64, "radius_km": 500},  # radius over 200 km
        {"name": "x", "polygon": [[-31, -64], [-31, -63]]},  # fewer than 3 corners
        {"name": "x", "polygon": [[-91, -64], [-31, -63], [-30, -63]]},  # latitude out of range
        {"name": "", "region_code": "14"},  # empty name
        {"name": "x", "region_code": "14", "min_probability": 1.5},
    ],
)
def test_alert_area_create_rejects_invalid(payload):
    with pytest.raises(ValidationError):
        AlertAreaCreate(**payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "Casa", "lat": -31, "lon": -64, "radius_km": 10},
        {"name": "Campo", "polygon": [[-31, -64], [-31, -63], [-30, -63]]},
        {"name": "Córdoba", "region_code": "14"},
        {"name": "Capital", "region_code": "14014"},
    ],
)
def test_alert_area_create_accepts_each_shape(payload):
    assert AlertAreaCreate(**payload).min_probability == 0.65


def test_alert_area_update_lat_lon_together_and_one_shape():
    with pytest.raises(ValidationError, match="together"):
        AlertAreaUpdate(lat=-31)
    with pytest.raises(ValidationError, match="at most one"):
        AlertAreaUpdate(region_code="14", radius_km=5)
    assert AlertAreaUpdate(name="nuevo").model_dump(exclude_unset=True) == {"name": "nuevo"}


@pytest.mark.parametrize("email", ["no-at-sign", "a@b", "a b@c.com", "x" * 250 + "@a.com"])
def test_alert_email_change_rejects_malformed(email):
    with pytest.raises(ValidationError):
        AlertEmailChange(email=email)


def test_distance_km_haversine():
    assert alerts._distance_km((-31.0, -64.0), (-31.0, -64.0)) == 0
    # 0.045 degrees of latitude is about 5 km.
    assert alerts._distance_km((-31.0, -64.0), (-31.045, -64.0)) == pytest.approx(5.0, abs=0.01)


DETECTIONS = [AlertedDetection(lat=-31.4, lon=-64.2, probability=0.87, image_time=datetime(2026, 10, 13, 17, 30))]


def test_format_local_time_is_argentina_time():
    assert mailer.format_local_time(datetime(2026, 10, 13, 2, 0)) == "12/10/2026 23:00"


def test_telegram_alert_message_escapes_area_name_and_uses_local_time():
    message = telegram.alert_message("<Casa & campo>", DETECTIONS)
    assert "Posible foco en &lt;Casa &amp; campo&gt;" in message
    assert "Probabilidad 87%" in message and "13/10/2026 14:30" in message
    assert "http://frontend.test/alertas" in message


def test_telegram_alert_message_plural():
    assert "2 posibles focos" in telegram.alert_message("Casa", DETECTIONS * 2)


def test_telegram_send_message_unconfigured_fails_without_calling_api(monkeypatch):
    call = MagicMock()
    monkeypatch.setattr(telegram, "_call", call)
    assert telegram.send_message(1, "hola") == "failed"
    call.assert_not_called()


@pytest.mark.parametrize("status,result", [(403, "blocked"), (400, "failed")])
def test_telegram_send_message_maps_api_errors(monkeypatch, status, result):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:abc")

    def fail(method, payload):
        raise telegram.TelegramError(status, "error")

    monkeypatch.setattr(telegram, "_call", fail)
    assert telegram.send_message(1, "hola") == result


def test_send_alert_email_skipped_without_sender(monkeypatch):
    boto3 = MagicMock()
    monkeypatch.setattr(mailer, "boto3", boto3)
    assert mailer.send_alert_email("a@b.com", "Casa", DETECTIONS) is False
    boto3.client.assert_not_called()


def test_send_alert_email_through_ses(monkeypatch):
    boto3 = MagicMock()
    monkeypatch.setattr(mailer, "boto3", boto3)
    monkeypatch.setenv("ALERT_EMAIL_FROM", "alertas@alertafuego.test")

    assert mailer.send_alert_email("a@b.com", "Casa", DETECTIONS * 2) is True

    kwargs = boto3.client.return_value.send_email.call_args.kwargs
    assert kwargs["Destination"] == {"ToAddresses": ["a@b.com"]}
    assert kwargs["Content"]["Simple"]["Subject"]["Data"] == "AlertaFuego: 2 posibles focos en Casa"


def test_send_email_failure_is_logged_not_raised(monkeypatch):
    boto3 = MagicMock()
    boto3.client.return_value.send_email.side_effect = RuntimeError("SES down")
    monkeypatch.setattr(mailer, "boto3", boto3)
    monkeypatch.setenv("ALERT_EMAIL_FROM", "alertas@alertafuego.test")
    assert mailer.send_alert_email("a@b.com", "Casa", DETECTIONS) is False


def test_confirmation_email_without_sender_counts_as_sent(monkeypatch):
    boto3 = MagicMock()
    monkeypatch.setattr(mailer, "boto3", boto3)
    assert mailer.send_confirmation_email("a@b.com", "tok", 24) is True
    boto3.client.assert_not_called()


def test_require_detect_key_fails_closed_when_unconfigured(monkeypatch):
    monkeypatch.setenv("DETECT_API_KEY", "")
    with pytest.raises(HTTPException) as exc:
        auth.require_detect_key("anything")
    assert exc.value.status_code == 503


def test_require_detect_key_checks_value():
    with pytest.raises(HTTPException) as exc:
        auth.require_detect_key("wrong")
    assert exc.value.status_code == 401
    auth.require_detect_key("test-detect-key")
