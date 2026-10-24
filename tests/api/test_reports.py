"""Functional tests: community reports on a detection (confirm fire / false positive),
including authentication."""

import uuid
from datetime import timedelta

import pytest

from tests.conftest import bearer, make_token

pytestmark = pytest.mark.db

FALSE_POSITIVE = {"verdict": "false_positive", "category": "industrial_activity", "comment": "Es una fábrica"}


def test_submit_report(client, add_detection, user_id):
    detection = add_detection()
    response = client.post(f"/detections/{detection.id}/reports", json=FALSE_POSITIVE, headers=bearer(user_id))
    assert response.status_code == 201
    body = response.json()
    assert body["detection_id"] == detection.id
    assert body["user_id"] == str(user_id)
    assert (body["verdict"], body["category"]) == ("false_positive", "industrial_activity")


def test_public_reports_hide_the_reporter(client, add_detection, user_id):
    detection = add_detection()
    client.post(f"/detections/{detection.id}/reports", json=FALSE_POSITIVE, headers=bearer(user_id))
    body = client.get(f"/detections/{detection.id}/reports").json()
    assert len(body) == 1
    assert "user_id" not in body[0]


def test_reports_of_unknown_detection(client, user_id):
    assert client.get("/detections/999/reports").status_code == 404
    assert client.post("/detections/999/reports", json=FALSE_POSITIVE, headers=bearer(user_id)).status_code == 404


def test_submit_report_requires_sign_in(client, add_detection):
    detection = add_detection()
    assert client.post(f"/detections/{detection.id}/reports", json=FALSE_POSITIVE).status_code == 401


@pytest.mark.parametrize(
    "token",
    [
        "not-a-jwt",
        make_token(uuid.uuid4(), expires_in=timedelta(seconds=-10)),
        make_token(uuid.uuid4(), audience="anon"),
    ],
    ids=["malformed", "expired", "wrong-audience"],
)
def test_invalid_token_is_401(client, add_detection, token):
    detection = add_detection()
    response = client.post(
        f"/detections/{detection.id}/reports", json=FALSE_POSITIVE, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401


def test_token_signed_with_another_key_is_401(client, add_detection, other_key):
    detection = add_detection()
    token = make_token(uuid.uuid4(), key=other_key)
    response = client.post(
        f"/detections/{detection.id}/reports", json=FALSE_POSITIVE, headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401


@pytest.mark.parametrize(
    "payload",
    [
        {"verdict": "false_positive"},
        {"verdict": "confirmed_fire", "category": "gas_flare"},
        {"verdict": "maybe"},
        {"verdict": "false_positive", "category": "aliens"},
    ],
)
def test_submit_report_validation(client, add_detection, user_id, payload):
    detection = add_detection()
    assert client.post(f"/detections/{detection.id}/reports", json=payload, headers=bearer(user_id)).status_code == 422


def test_one_report_per_user_and_detection(client, add_detection, user_id):
    detection = add_detection()
    url = f"/detections/{detection.id}/reports"
    assert client.post(url, json=FALSE_POSITIVE, headers=bearer(user_id)).status_code == 201
    assert client.post(url, json={"verdict": "confirmed_fire"}, headers=bearer(user_id)).status_code == 409
    # Another user can still report it.
    assert client.post(url, json={"verdict": "confirmed_fire"}, headers=bearer(uuid.uuid4())).status_code == 201


def test_my_report(client, add_detection, user_id):
    detection = add_detection()
    url = f"/detections/{detection.id}/reports"
    assert client.get(f"{url}/mine", headers=bearer(user_id)).status_code == 404
    client.post(url, json={"verdict": "confirmed_fire"}, headers=bearer(user_id))
    assert client.get(f"{url}/mine", headers=bearer(user_id)).json()["verdict"] == "confirmed_fire"
    assert client.get(f"{url}/mine").status_code == 401


def test_edit_report_to_confirmed_fire_clears_category(client, add_detection, user_id):
    detection = add_detection()
    url = f"/detections/{detection.id}/reports"
    client.post(url, json=FALSE_POSITIVE, headers=bearer(user_id))
    response = client.patch(url, json={"verdict": "confirmed_fire"}, headers=bearer(user_id))
    assert response.status_code == 200
    assert (response.json()["verdict"], response.json()["category"]) == ("confirmed_fire", None)


def test_edit_report_to_false_positive_needs_category(client, add_detection, user_id):
    detection = add_detection()
    url = f"/detections/{detection.id}/reports"
    client.post(url, json={"verdict": "confirmed_fire"}, headers=bearer(user_id))
    assert client.patch(url, json={"verdict": "false_positive"}, headers=bearer(user_id)).status_code == 400
    ok = client.patch(url, json={"verdict": "false_positive", "category": "sun_glint"}, headers=bearer(user_id))
    assert ok.json()["category"] == "sun_glint"


def test_edit_report_errors(client, add_detection, user_id):
    detection = add_detection()
    url = f"/detections/{detection.id}/reports"
    assert client.patch(url, json={"comment": "x"}, headers=bearer(user_id)).status_code == 404
    client.post(url, json={"verdict": "confirmed_fire"}, headers=bearer(user_id))
    assert client.patch(url, json={}, headers=bearer(user_id)).status_code == 400
