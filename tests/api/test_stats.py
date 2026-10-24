"""Functional tests: dashboard statistics (per region, per hour, reports, recurrence)."""

from datetime import datetime

import pytest

pytestmark = pytest.mark.db


@pytest.fixture
def regions(add_region):
    add_region("14", "Córdoba", (-66, -35, -62, -29))
    add_region("14014", "Capital", (-64.5, -31.6, -64.0, -31.2), province_code="14")
    add_region("62", "Río Negro", (-72, -42, -62, -37))


def test_region_stats_by_province_busiest_first(client, regions, add_detection):
    add_detection(lat=-31.4, lon=-64.2, probability=0.7)
    add_detection(lat=-30.0, lon=-65.0, probability=0.9)
    add_detection(lat=-40.0, lon=-68.0, probability=0.6)

    body = client.get("/stats/regions").json()

    assert body == [
        {"code": "14", "name": "Córdoba", "province_name": None, "detection_count": 2, "max_probability": 0.9},
        {"code": "62", "name": "Río Negro", "province_name": None, "detection_count": 1, "max_probability": 0.6},
    ]


def test_region_stats_by_department(client, regions, add_detection):
    add_detection(lat=-31.4, lon=-64.2)
    body = client.get("/stats/regions", params={"kind": "department"}).json()
    assert [(r["code"], r["province_name"]) for r in body] == [("14014", "Córdoba")]


def test_region_stats_invalid_kind(client):
    assert client.get("/stats/regions", params={"kind": "country"}).status_code == 422


def test_hourly_stats_in_argentina_time(client, add_detection):
    add_detection(image_time=datetime(2026, 10, 1, 2, 0))  # 23:00 ART the day before
    add_detection(image_time=datetime(2026, 10, 1, 18, 30))  # 15:30 ART
    add_detection(image_time=datetime(2026, 10, 2, 18, 0))

    body = client.get("/stats/hourly").json()

    assert [h["hour"] for h in body] == list(range(24))
    counts = {h["hour"]: h["detection_count"] for h in body}
    assert counts[23] == 1 and counts[15] == 2 and sum(counts.values()) == 3


def test_hourly_stats_place_filter(client, regions, add_detection):
    add_detection(lat=-31.4, lon=-64.2)
    add_detection(lat=-40.0, lon=-68.0)
    body = client.get("/stats/hourly", params={"region_code": "62"}).json()
    assert sum(h["detection_count"] for h in body) == 1


def test_report_stats(client, add_detection, add_report, user_id):
    import uuid

    detection = add_detection()
    add_report(detection.id, user_id, "false_positive", "gas_flare")
    add_report(detection.id, uuid.uuid4(), "false_positive", "gas_flare")
    add_report(detection.id, uuid.uuid4(), "confirmed_fire")

    body = client.get("/stats/reports").json()

    assert sorted((r["verdict"], r["category"], r["report_count"]) for r in body) == [
        ("confirmed_fire", None, 1),
        ("false_positive", "gas_flare", 2),
    ]


def test_recurrence_counts_distinct_local_days(client, add_detection):
    # Same ~2 km cell: two detections on 1 Oct (ART), one on 3 Oct.
    add_detection(lat=-31.401, lon=-64.201, image_time=datetime(2026, 10, 1, 15, 0))
    add_detection(lat=-31.402, lon=-64.202, image_time=datetime(2026, 10, 1, 20, 0))
    add_detection(lat=-31.401, lon=-64.201, image_time=datetime(2026, 10, 3, 15, 0))
    add_detection(lat=-40.0, lon=-68.0, image_time=datetime(2026, 10, 1, 15, 0))

    body = client.get("/stats/recurrence").json()

    assert body["cell_size_deg"] == 0.02
    top = body["cells"][0]
    assert (top["fire_days"], top["years"], top["detection_count"]) == (2, 1, 3)
    assert top["lat"] == pytest.approx(-31.41) and top["lon"] == pytest.approx(-64.21)
    assert len(body["cells"]) == 2


def test_stats_time_range(client, add_detection):
    add_detection(image_time=datetime(2025, 1, 1))
    add_detection(image_time=datetime(2026, 10, 1))
    body = client.get("/stats/recurrence", params={"since": "2026-01-01T00:00:00"}).json()
    assert sum(c["detection_count"] for c in body["cells"]) == 1
    assert client.get("/stats/hourly", params={"until": "yesterday"}).status_code == 422
