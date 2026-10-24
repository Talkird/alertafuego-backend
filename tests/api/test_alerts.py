"""Functional tests: alert areas (circle, polygon, region), the user's alert list and
the alert-raising logic run after each detection (cooldown, thresholds, channels)."""

import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from backend.app import alerts
from backend.app.crud import set_telegram_link_token, link_telegram_chat, update_user_settings
from tests.conftest import bearer

pytestmark = pytest.mark.db

CIRCLE = {"name": "Casa", "lat": -31.0, "lon": -64.0, "radius_km": 10}
POLYGON = {"name": "Campo", "polygon": [[-31.0, -64.0], [-31.0, -63.0], [-30.0, -63.0], [-30.0, -64.0]]}


@pytest.fixture
def sent(monkeypatch):
    """Capture alert emails and Telegram messages instead of sending them."""
    email = MagicMock(return_value=True)
    telegram = MagicMock(return_value="sent")
    monkeypatch.setattr(alerts, "send_alert_email", email)
    monkeypatch.setattr(alerts, "send_alert", telegram)
    return MagicMock(email=email, telegram=telegram)


def test_create_circle_area(client, user_id):
    response = client.post("/alert-areas", json=CIRCLE, headers=bearer(user_id, "yo@example.com"))
    assert response.status_code == 201
    body = response.json()
    assert (body["lat"], body["lon"], body["radius_km"]) == (-31.0, -64.0, 10)
    assert body["polygon"] is None and body["region"] is None
    assert body["email"] == "yo@example.com" and body["min_probability"] == 0.65 and body["active"] is True


def test_create_polygon_area_returns_open_ring(client, user_id):
    body = client.post("/alert-areas", json=POLYGON, headers=bearer(user_id)).json()
    assert body["polygon"] == POLYGON["polygon"]


def test_create_region_area(client, user_id, add_region):
    add_region("14", "Córdoba", (-66, -35, -62, -29))
    body = client.post("/alert-areas", json={"name": "Prov", "region_code": "14"}, headers=bearer(user_id)).json()
    assert body["region"] == {"code": "14", "name": "Córdoba", "province_code": None}


def test_create_area_unknown_region(client, user_id):
    response = client.post("/alert-areas", json={"name": "X", "region_code": "99"}, headers=bearer(user_id))
    assert response.status_code == 404


def test_create_area_self_crossing_polygon(client, user_id):
    bowtie = {"name": "X", "polygon": [[-31.0, -64.0], [-30.0, -63.0], [-31.0, -63.0], [-30.0, -64.0]]}
    response = client.post("/alert-areas", json=bowtie, headers=bearer(user_id))
    assert response.status_code == 422
    assert "cross" in response.json()["detail"]


def test_create_area_polygon_too_large(client, user_id):
    huge = {"name": "X", "polygon": [[-40.0, -70.0], [-40.0, -60.0], [-30.0, -60.0], [-30.0, -70.0]]}
    response = client.post("/alert-areas", json=huge, headers=bearer(user_id))
    assert response.status_code == 422
    assert "larger" in response.json()["detail"]


def test_create_area_validation_and_auth(client, user_id):
    assert client.post("/alert-areas", json={"name": "X"}, headers=bearer(user_id)).status_code == 422
    assert client.post("/alert-areas", json=CIRCLE).status_code == 401
    # The alert email comes from the token: a token without one can't create areas.
    assert client.post("/alert-areas", json=CIRCLE, headers=bearer(user_id, email=None)).status_code == 400


def test_list_areas_only_own(client, user_id):
    client.post("/alert-areas", json=CIRCLE, headers=bearer(user_id))
    client.post("/alert-areas", json=POLYGON, headers=bearer(uuid.uuid4()))
    body = client.get("/alert-areas", headers=bearer(user_id)).json()
    assert [a["name"] for a in body] == ["Casa"]


def test_update_area_switches_shape(client, user_id, add_region):
    add_region("14", "Córdoba", (-66, -35, -62, -29))
    area = client.post("/alert-areas", json=CIRCLE, headers=bearer(user_id)).json()
    url = f"/alert-areas/{area['id']}"

    body = client.patch(url, json={"region_code": "14", "name": "Prov"}, headers=bearer(user_id)).json()
    assert body["region"]["code"] == "14" and body["lat"] is None and body["radius_km"] is None
    assert body["name"] == "Prov"

    body = client.patch(url, json={"polygon": POLYGON["polygon"]}, headers=bearer(user_id)).json()
    assert body["polygon"] == POLYGON["polygon"] and body["region"] is None


def test_update_area_errors(client, user_id):
    area = client.post("/alert-areas", json={"name": "P", **{k: v for k, v in POLYGON.items() if k != "name"}}, headers=bearer(user_id)).json()
    url = f"/alert-areas/{area['id']}"
    assert client.patch(url, json={}, headers=bearer(user_id)).status_code == 400
    assert client.patch(url, json={"name": None}, headers=bearer(user_id)).status_code == 400
    # A polygon switched to a circle without a radius isn't a whole shape.
    assert client.patch(url, json={"lat": -31, "lon": -64}, headers=bearer(user_id)).status_code == 400
    assert client.patch(url, json={"name": "x"}, headers=bearer(uuid.uuid4())).status_code == 404
    assert client.patch(url, json={"min_probability": 2}, headers=bearer(user_id)).status_code == 422


def test_delete_area(client, user_id):
    area = client.post("/alert-areas", json=CIRCLE, headers=bearer(user_id)).json()
    url = f"/alert-areas/{area['id']}"
    assert client.delete(url, headers=bearer(uuid.uuid4())).status_code == 404
    assert client.delete(url, headers=bearer(user_id)).status_code == 204
    assert client.delete(url, headers=bearer(user_id)).status_code == 404


def test_alerts_raised_for_detections_inside_area(client, db, user_id, add_detection, sent):
    client.post("/alert-areas", json=CIRCLE, headers=bearer(user_id))
    inside = add_detection(lat=-31.02, lon=-64.0, probability=0.9)
    outside = add_detection(lat=-35.0, lon=-64.0, probability=0.9)
    weak = add_detection(lat=-31.0, lon=-64.1, probability=0.6)  # under min_probability 0.65

    assert alerts.process_new_detections(db, [inside.id, outside.id, weak.id]) == 1

    sent.email.assert_called_once()
    assert sent.email.call_args.args[0] == "user@example.com"
    body = client.get("/alerts", headers=bearer(user_id)).json()
    assert len(body) == 1
    assert body[0]["detection"]["id"] == inside.id and body[0]["area_name"] == "Casa"
    assert body[0]["emailed"] is True and body[0]["telegram_sent"] is False


def test_alerts_one_per_fire_within_a_run_and_cooldown(client, db, user_id, add_detection, sent):
    client.post("/alert-areas", json=CIRCLE, headers=bearer(user_id))
    t0 = datetime(2026, 10, 1, 15, 0)
    # Two neighbouring pixels of one fire, same run: one alert, the strongest.
    strong = add_detection(lat=-31.0, lon=-64.0, probability=0.95, image_time=t0)
    neighbour = add_detection(lat=-31.01, lon=-64.0, probability=0.8, image_time=t0)
    assert alerts.process_new_detections(db, [strong.id, neighbour.id]) == 1

    # The same fire one hour later: in cooldown.
    later = add_detection(lat=-31.0, lon=-64.0, probability=0.9, image_time=t0 + timedelta(hours=1))
    assert alerts.process_new_detections(db, [later.id]) == 0

    # After the cooldown it alerts again.
    much_later = add_detection(lat=-31.0, lon=-64.0, probability=0.9, image_time=t0 + timedelta(hours=7))
    assert alerts.process_new_detections(db, [much_later.id]) == 1

    ids = [a["detection"]["id"] for a in client.get("/alerts", headers=bearer(user_id)).json()]
    assert ids == [much_later.id, strong.id]


def test_alerts_respect_inactive_area_and_email_off(client, db, user_id, add_detection, sent):
    area = client.post("/alert-areas", json=CIRCLE, headers=bearer(user_id)).json()
    client.patch(f"/alert-areas/{area['id']}", json={"active": False}, headers=bearer(user_id))
    assert alerts.process_new_detections(db, [add_detection().id]) == 0

    client.patch(f"/alert-areas/{area['id']}", json={"active": True}, headers=bearer(user_id))
    update_user_settings(db, user_id, {"email_alerts": False})
    detection = add_detection(lat=-31.0, lon=-64.0)
    assert alerts.process_new_detections(db, [detection.id]) == 1
    sent.email.assert_not_called()


def test_alerts_by_telegram_and_blocked_bot_unlinks(client, db, user_id, add_detection, sent):
    client.post("/alert-areas", json=CIRCLE, headers=bearer(user_id))
    set_telegram_link_token(db, user_id, "hash", datetime(2100, 1, 1))
    assert link_telegram_chat(db, "hash", 4242, "@vecino", datetime(2026, 1, 1))
    sent.telegram.return_value = "blocked"

    assert alerts.process_new_detections(db, [add_detection(lat=-31.0, lon=-64.0).id]) == 1

    assert sent.telegram.call_args.args[0] == 4242
    account = client.get("/account", headers=bearer(user_id)).json()
    assert account["telegram_connected"] is False


def test_alerts_region_and_polygon_areas(client, db, user_id, add_detection, add_region, sent):
    add_region("14", "Córdoba", (-66, -35, -62, -29))
    client.post("/alert-areas", json={"name": "Prov", "region_code": "14"}, headers=bearer(user_id))
    client.post("/alert-areas", json=POLYGON, headers=bearer(user_id))
    detection = add_detection(lat=-30.5, lon=-63.5)  # inside both
    assert alerts.process_new_detections(db, [detection.id]) == 2


def test_alerts_list_auth_and_limit(client, user_id):
    assert client.get("/alerts").status_code == 401
    assert client.get("/alerts", params={"limit": 501}, headers=bearer(user_id)).status_code == 422
    assert client.get("/alerts", headers=bearer(user_id)).json() == []


def test_process_no_detections(db):
    assert alerts.process_new_detections(db, []) == 0
