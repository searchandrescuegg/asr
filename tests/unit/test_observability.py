"""T022 — observability log content test.

Verifies that the orchestrator's log lines contain the required fields
and DO NOT contain the transcription text or audio bytes (FR-013, the
2026-04-29 clarification on log content).
"""

from __future__ import annotations

import json

import pytest

from asr.config import get_settings
from asr.models.registry import ModelRegistry
from asr.observability.logging import configure_logging
from asr.pipeline.queue import QueueRouter
from asr.pipeline.transcribe import Pipeline
from tests.conftest import StubModel, make_tone_wav_bytes


@pytest.mark.asyncio
async def test_orchestrator_log_fields_present_and_no_secret_content(capsys):
    secret_text = "this transcription text MUST NOT appear in logs"

    stub = StubModel(text=secret_text)
    stub.load()
    registry = ModelRegistry(default_identifier=stub.identifier)
    registry.register(stub)
    queue_router = QueueRouter(depth=get_settings().queue_depth)
    pipeline = Pipeline(registry, queue_router)

    configure_logging()
    audio = make_tone_wav_bytes(seconds=0.2)
    result = await pipeline.transcribe(audio, None)

    assert result.text == secret_text  # the API does carry the text — only logs do not
    captured = capsys.readouterr()
    log_text = captured.out

    # Pull every parseable JSON line out
    seen_events: list[dict] = []
    for line in log_text.splitlines():
        try:
            seen_events.append(json.loads(line))
        except Exception:
            continue

    completion = next(
        (e for e in seen_events if e.get("event") == "transcription_completed"), None
    )
    assert completion is not None, f"missing transcription_completed log: {log_text!r}"

    for required in (
        "model",
        "audio_duration_s",
        "inference_duration_s",
        "audio_sha256",
        "status",
    ):
        assert required in completion, f"log missing {required}: {completion}"

    assert secret_text not in log_text, (
        "transcription text leaked into logs (FR-013 violation)"
    )
    assert "audio_bytes_b64" not in log_text
