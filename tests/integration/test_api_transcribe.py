from __future__ import annotations

import pytest

from tests.conftest import make_silence_wav_bytes, make_tone_wav_bytes


@pytest.mark.asyncio
async def test_default_model_success(client):
    audio = make_tone_wav_bytes(seconds=0.5)
    files = {"file": ("clip.wav", audio, "audio/wav")}
    resp = await client.post("/api/v1/transcribe", files=files)
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert body["text"] == "hello world"
    assert body["model"] == "stub-en"
    assert body["audio_duration_s"] > 0
    assert body["inference_duration_s"] >= 0
    assert body["downmix_applied"] is False
    assert body["no_speech_detected"] is False
    assert body["correlation_id"]
    assert resp.headers.get("x-correlation-id") == body["correlation_id"]


@pytest.mark.asyncio
async def test_invalid_format_error(client):
    files = {"file": ("not-audio.bin", b"not audio at all", "application/octet-stream")}
    resp = await client.post("/api/v1/transcribe", files=files)
    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "INVALID_FORMAT"
    assert "correlation_id" in body
    assert resp.headers.get("x-correlation-id") == body["correlation_id"]


@pytest.mark.asyncio
async def test_no_speech_flag(client):
    audio = make_silence_wav_bytes(seconds=0.5)
    files = {"file": ("silence.wav", audio, "audio/wav")}
    resp = await client.post("/api/v1/transcribe", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["text"] == ""
    assert body["no_speech_detected"] is True


@pytest.mark.asyncio
async def test_file_too_large_error(client, monkeypatch):
    from asr.config import get_settings

    monkeypatch.setenv("ASR_MAX_FILE_BYTES", "100")
    from asr.config import reset_settings_for_tests

    reset_settings_for_tests()
    assert get_settings().max_file_bytes == 100

    audio = make_tone_wav_bytes(seconds=0.5)
    assert len(audio) > 100
    files = {"file": ("big.wav", audio, "audio/wav")}
    resp = await client.post("/api/v1/transcribe", files=files)
    assert resp.status_code == 413
    body = resp.json()
    assert body["code"] == "FILE_TOO_LARGE"
    assert body["details"]["limit_bytes"] == 100


@pytest.mark.asyncio
async def test_explicit_unknown_model_returns_404(client):
    audio = make_tone_wav_bytes(seconds=0.5)
    files = {"file": ("clip.wav", audio, "audio/wav")}
    resp = await client.post(
        "/api/v1/transcribe", files=files, data={"model": "does-not-exist"}
    )
    assert resp.status_code == 404
    body = resp.json()
    assert body["code"] == "MODEL_NOT_FOUND"
