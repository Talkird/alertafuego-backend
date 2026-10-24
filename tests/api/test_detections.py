"""Functional tests: GET /detections (map data) and the detection thumbnail."""

from datetime import datetime

import pytest

from tests.conftest import bearer

pytestmark = pytest.mark.db


def test_root(client):
    assert client.get("/").status_code == 200


def test_list_detections_empty(client):
    response = client.get("/detections")
    assert response.status_code == 200
    assert response.json() == []


def test_list_detections_shape_and_order(client, add_detection, add_report, user_id):
    older = add_detection(image_time=datetime(2026, 10, 1, 12, 0))
    newer = add_detection(lat=-32.0, lon=-65.0, image_time=datetime(2026, 10, 2, 12, 0), image=b"\x89PNG")
    add_report(older.id, user_id)

    body = client.get("/detections").json()

    assert [d["id"] for d in body] == [newer.id, older.id]
    assert body[0]["lat"] == pytest.approx(-32.0) and body[0]["lon"] == pytest.approx(-65.0)
    assert body[0]["has_image"] is True and body[1]["has_image"] is False
    assert body[1]["report_count"] == 1
    assert set(body[0]["bbox"]) == {"west", "south", "east", "north"}


def test_list_detections_bbox_filter(client, add_detection):
    inside = add_detection(lat=-31.0, lon=-64.0)
    add_detection(lat=-45.0, lon=-70.0)
    body = client.get("/detections", params={"west": -65, "south": -32, "east": -63, "north": -30}).json()
    assert [d["id"] for d in body] == [inside.id]


def test_list_detections_time_range_and_limit(client, add_detection):
    add_detection(image_time=datetime(2026, 9, 1))
    in_range = add_detection(image_time=datetime(2026, 10, 1))
    add_detection(image_time=datetime(2026, 11, 1))

    body = client.get("/detections", params={"since": "2026-09-15T00:00:00", "until": "2026-10-15T00:00:00"}).json()
    assert [d["id"] for d in body] == [in_range.id]
    assert len(client.get("/detections", params={"limit": 2}).json()) == 2


@pytest.mark.parametrize(
    "params",
    [{"limit": 5001}, {"since": "not-a-date"}, {"region_code": "1"}, {"west": "west"}],
)
def test_list_detections_rejects_invalid_query(client, params):
    assert client.get("/detections", params=params).status_code == 422


def test_list_detections_region_filter(client, add_detection, add_region):
    add_region("14", "Córdoba", (-66, -35, -62, -29))
    inside = add_detection(lat=-31.0, lon=-64.0)
    add_detection(lat=-45.0, lon=-70.0)
    body = client.get("/detections", params={"region_code": "14"}).json()
    assert [d["id"] for d in body] == [inside.id]


def test_list_detections_unknown_region(client):
    assert client.get("/detections", params={"region_code": "99"}).status_code == 404


def test_list_detections_region_and_area_together(client):
    assert client.get("/detections", params={"region_code": "14", "area_id": 1}).status_code == 400


def test_list_detections_area_filter_needs_sign_in(client):
    assert client.get("/detections", params={"area_id": 1}).status_code == 401


def test_list_detections_area_filter(client, add_detection, user_id):
    headers = bearer(user_id)
    area = client.post(
        "/alert-areas", json={"name": "Casa", "lat": -31.0, "lon": -64.0, "radius_km": 10}, headers=headers
    ).json()
    near = add_detection(lat=-31.05, lon=-64.0)
    add_detection(lat=-32.0, lon=-64.0)  # ~110 km away

    body = client.get("/detections", params={"area_id": area["id"]}, headers=headers).json()
    assert [d["id"] for d in body] == [near.id]


def test_list_detections_cannot_use_another_users_area(client, user_id):
    import uuid

    area = client.post(
        "/alert-areas", json={"name": "Casa", "lat": -31.0, "lon": -64.0, "radius_km": 10}, headers=bearer(user_id)
    ).json()
    response = client.get("/detections", params={"area_id": area["id"]}, headers=bearer(uuid.uuid4()))
    assert response.status_code == 404


def test_detection_image(client, add_detection):
    detection = add_detection(image=b"\x89PNGdata")
    response = client.get(f"/detections/{detection.id}/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"\x89PNGdata"


def test_detection_image_missing(client, add_detection):
    without_image = add_detection()
    assert client.get(f"/detections/{without_image.id}/image").status_code == 404
    assert client.get("/detections/999999/image").status_code == 404
