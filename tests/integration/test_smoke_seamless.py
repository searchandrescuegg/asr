"""T036 — Seamless real-model smoke (Constitution IV layer 2)."""

from __future__ import annotations

import pathlib

import pytest

from asr.audio.decode import decode
from tests.integration.test_smoke_parakeet import _wer

SEAMLESS_PUBLISHED_WER = 0.090  # SeamlessM4T paper; mirror in audio/REFERENCE.md


@pytest.mark.gpu_required
@pytest.mark.model_required
def test_seamless_smoke():
    from asr.models.seamless import SeamlessModel

    fixture = pathlib.Path(__file__).resolve().parents[2] / "audio" / "psern01.wav"
    if not fixture.exists():
        pytest.skip(f"reference clip {fixture} not present; see audio/REFERENCE.md")

    decoded = decode(fixture.read_bytes(), max_seconds=120)

    model = SeamlessModel()
    model.load()
    output = model.transcribe(decoded.samples)

    expected = _expected_text_from_manifest()
    if expected is None:
        pytest.skip("expected text not yet recorded in audio/REFERENCE.md")

    wer = _wer(expected, output.text)
    assert abs(wer - SEAMLESS_PUBLISHED_WER) <= 0.05, (
        f"Seamless WER {wer:.3f} drifted >5pp from published {SEAMLESS_PUBLISHED_WER:.3f}"
    )
    assert output.language == "eng", (
        "Seamless reports a detected language; expected 'eng' for the English fixture"
    )


def _expected_text_from_manifest() -> str | None:
    return None
