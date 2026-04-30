from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class ModelState(StrEnum):
    LOADING = "LOADING"
    READY = "READY"
    FAILED = "FAILED"


@dataclass
class ModelOutput:
    text: str
    language: str | None = None


class ASRModel(ABC):
    identifier: str
    name: str
    vendor: str
    languages: list[str]
    expected_sr_hz: int = 16_000

    def __init__(self) -> None:
        self.state: ModelState = ModelState.LOADING
        self.last_error: str | None = None

    @abstractmethod
    def load(self) -> None:
        """Load weights into memory and transition state to READY (or FAILED)."""

    def warm_up(self) -> None:  # noqa: B027 — intentional default no-op hook
        """Optional warm-up after load(); default is a no-op."""

    @abstractmethod
    def transcribe(self, samples: np.ndarray) -> ModelOutput:
        """Run inference on a 1-D float32 mono 16 kHz array."""
