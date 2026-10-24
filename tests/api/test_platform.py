"""Functional tests: cross-cutting API behaviour, interactive docs (RF-36) and the
browser origins allowed to call the API (RF-37)."""

import pytest

pytestmark = pytest.mark.db


def test_interactive_docs_and_openapi_spec(client):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert {"/detections", "/detections/{detection_id}/reports", "/alert-areas", "/account"} <= set(spec["paths"])
    # The Telegram webhook is internal: left out of the public contract.
    assert "/telegram/webhook" not in spec["paths"]


def _preflight(client, origin: str):
    return client.options(
        "/detections",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )


def test_cors_allows_the_configured_frontend(client):
    response = _preflight(client, "http://localhost:3000")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_rejects_other_origins(client):
    response = _preflight(client, "https://evil.example.com")
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
