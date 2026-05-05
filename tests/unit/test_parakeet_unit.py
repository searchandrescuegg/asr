from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asr.models.parakeet import ParakeetModel


@dataclass
class _FakeHypothesis:
    text: str

    def __repr__(self) -> str:  # mimic NeMo's verbose repr
        return f"Hypothesis(score=0.0, text='{self.text}', length={len(self.text)})"


class _FakeNemoModel:
    def __init__(self, result):
        self._result = result

    def transcribe(self, _batch):
        return [self._result]


def _model_with(result) -> ParakeetModel:
    m = ParakeetModel()
    m._model = _FakeNemoModel(result)
    return m


def test_empty_hypothesis_returns_empty_text_not_repr():
    """Regression: empty hypothesis must collapse to '' so the pipeline
    sets no_speech_detected=True instead of forwarding the Hypothesis repr."""
    samples = np.zeros(16_000, dtype=np.float32)
    out = _model_with(_FakeHypothesis(text="")).transcribe(samples)
    assert out.text == ""
    assert "Hypothesis" not in out.text


def test_populated_hypothesis_returns_text():
    samples = np.zeros(16_000, dtype=np.float32)
    out = _model_with(_FakeHypothesis(text="hello world")).transcribe(samples)
    assert out.text == "hello world"


def test_string_result_is_passed_through():
    """Older NeMo API returns plain strings; preserve that path."""
    samples = np.zeros(16_000, dtype=np.float32)
    out = _model_with("legacy api result").transcribe(samples)
    assert out.text == "legacy api result"


def test_transcribe_before_load_raises():
    m = ParakeetModel()
    with __import__("pytest").raises(RuntimeError):
        m.transcribe(np.zeros(16_000, dtype=np.float32))
