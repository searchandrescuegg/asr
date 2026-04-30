"""T051 — Parakeet real-model smoke (Constitution IV layer 2).

Loads the production Parakeet adapter, transcribes the SC-003 reference
clip (`audio/psern01.wav` per `audio/REFERENCE.md`), and asserts the
measured WER is within 5pp of the published number.

Skipped on CPU CI; runs on GPU runners.
"""

from __future__ import annotations

import pathlib

import pytest

from asr.audio.decode import decode

PARAKEET_PUBLISHED_WER = 0.064  # NVIDIA model card; mirror in audio/REFERENCE.md


def _wer(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    n = max(len(ref), 1)
    # Levenshtein-by-words via simple DP
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1] / n


@pytest.mark.gpu_required
@pytest.mark.model_required
def test_parakeet_smoke():
    from asr.models.parakeet import ParakeetModel

    fixture = pathlib.Path(__file__).resolve().parents[2] / "audio" / "psern01.wav"
    if not fixture.exists():
        pytest.skip(f"reference clip {fixture} not present; see audio/REFERENCE.md")

    decoded = decode(fixture.read_bytes(), max_seconds=120)

    model = ParakeetModel()
    model.load()
    output = model.transcribe(decoded.samples)

    # The reference manifest defines the expected text. Until that is
    # populated, this test will skip rather than spuriously fail.
    expected = _expected_text_from_manifest()
    if expected is None:
        pytest.skip("expected text not yet recorded in audio/REFERENCE.md")

    wer = _wer(expected, output.text)
    assert abs(wer - PARAKEET_PUBLISHED_WER) <= 0.05, (
        f"Parakeet WER {wer:.3f} drifted >5pp from published {PARAKEET_PUBLISHED_WER:.3f}"
    )
    assert output.language is None, "Parakeet does not report a detected language"


def _expected_text_from_manifest() -> str | None:
    # Placeholder: the manifest is markdown, not structured. Operators
    # populating REFERENCE.md should also add a line like
    # `EXPECTED_psern01="…"` to a sibling shell-style file or simply
    # return the literal string here.
    return None
