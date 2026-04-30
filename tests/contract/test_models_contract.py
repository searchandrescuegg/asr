import pytest

from asr.api.v1.schemas import ModelListResponse


@pytest.mark.asyncio
async def test_models_list_validates_against_schema(client):
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    parsed = ModelListResponse.model_validate(body)
    assert parsed.default
    assert any(m.identifier == "stub-en" for m in parsed.models)
