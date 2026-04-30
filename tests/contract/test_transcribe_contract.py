"""T043 — Per-error-code contract tests.

Each documented ErrorCode is exercised end-to-end and the response body is
asserted against the ErrorEnvelope schema.
"""

from __future__ import annotations

import pytest

from asr.api.errors import ErrorCode, ErrorEnvelope
from tests.conftest import make_silence_wav_bytes, make_tone_wav_bytes


def assert_envelope(body: dict, expected_code: ErrorCode) -> None:
    envelope = ErrorEnvelope.model_validate(body)
    assert envelope.code == expected_code
    assert envelope.message
    assert envelope.correlation_id


@pytest.mark.asyncio
async def test_invalid_format(client):
    files = {"file": ("not.bin", b"definitely not audio", "application/octet-stream")}
    resp = await client.post("/api/v1/transcribe", files=files)
    assert resp.status_code == 400
    assert_envelope(resp.json(), ErrorCode.INVALID_FORMAT)


@pytest.mark.asyncio
async def test_audio_too_long(client, monkeypatch):
    monkeypatch.setenv("ASR_MAX_AUDIO_SECONDS", "0.01")
    from asr.config import reset_settings_for_tests

    reset_settings_for_tests()

    audio = make_tone_wav_bytes(seconds=0.5)
    resp = await client.post(
        "/api/v1/transcribe", files={"file": ("clip.wav", audio, "audio/wav")}
    )
    assert resp.status_code == 400
    assert_envelope(resp.json(), ErrorCode.AUDIO_TOO_LONG)


@pytest.mark.asyncio
async def test_file_too_large(client, monkeypatch):
    monkeypatch.setenv("ASR_MAX_FILE_BYTES", "10")
    from asr.config import reset_settings_for_tests

    reset_settings_for_tests()

    audio = make_tone_wav_bytes(seconds=0.5)
    resp = await client.post(
        "/api/v1/transcribe", files={"file": ("clip.wav", audio, "audio/wav")}
    )
    assert resp.status_code == 413
    body = resp.json()
    assert_envelope(body, ErrorCode.FILE_TOO_LARGE)
    assert body["details"]["limit_bytes"] == 10


@pytest.mark.asyncio
async def test_model_not_found(client):
    audio = make_tone_wav_bytes(seconds=0.2)
    resp = await client.post(
        "/api/v1/transcribe",
        files={"file": ("clip.wav", audio, "audio/wav")},
        data={"model": "nonexistent-99"},
    )
    assert resp.status_code == 404
    body = resp.json()
    assert_envelope(body, ErrorCode.MODEL_NOT_FOUND)
    assert "available" in body["details"]


@pytest.mark.asyncio
async def test_no_default_model_available(stub_model, monkeypatch):
    """Default not registered → NO_DEFAULT_MODEL on a request without explicit model."""
    monkeypatch.setenv("ASR_DEFAULT_MODEL", "parakeet-tdt-0.6b-v2")
    monkeypatch.setenv("ASR_ENABLED_MODELS", "stub-en")
    from asr.config import reset_settings_for_tests

    reset_settings_for_tests()

    from contextlib import AsyncExitStack

    from httpx import ASGITransport, AsyncClient

    from asr.app import create_app

    app = create_app(models=[stub_model])

    async with AsyncExitStack() as stack:
        await stack.enter_async_context(app.router.lifespan_context(app))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            audio = make_tone_wav_bytes(seconds=0.2)
            resp = await c.post(
                "/api/v1/transcribe",
                files={"file": ("clip.wav", audio, "audio/wav")},
            )
            assert resp.status_code == 503
            assert_envelope(resp.json(), ErrorCode.NO_DEFAULT_MODEL)


@pytest.mark.asyncio
async def test_success_envelope_uses_x_correlation_id(client):
    audio = make_silence_wav_bytes(seconds=0.2)
    resp = await client.post(
        "/api/v1/transcribe", files={"file": ("clip.wav", audio, "audio/wav")}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["correlation_id"] == resp.headers["x-correlation-id"]
