import pytest


@pytest.mark.asyncio
async def test_list_models(client):
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["default"] == "stub-en"
    ids = [m["identifier"] for m in body["models"]]
    assert "stub-en" in ids
    stub = next(m for m in body["models"] if m["identifier"] == "stub-en")
    assert stub["state"] == "READY"
    assert stub["languages"] == ["en"]


@pytest.mark.asyncio
async def test_healthz(client):
    resp = await client.get("/api/v1/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
