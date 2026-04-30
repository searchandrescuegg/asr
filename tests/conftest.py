from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator

import numpy as np
import pytest

os.environ.setdefault("ASR_ALLOW_CPU", "1")
os.environ.setdefault("ASR_DEFAULT_MODEL", "stub-en")
os.environ.setdefault("ASR_ENABLED_MODELS", "stub-en")
os.environ.setdefault("ASR_MAX_FILE_BYTES", str(10 * 1024 * 1024))
os.environ.setdefault("ASR_MAX_AUDIO_SECONDS", "60")
os.environ.setdefault("ASR_QUEUE_DEPTH", "2")

from asr.config import reset_settings_for_tests  # noqa: E402
from asr.models.base import ASRModel, ModelOutput, ModelState  # noqa: E402


class StubModel(ASRModel):
    identifier = "stub-en"
    name = "Stub English ASR"
    vendor = "test"
    languages = ["en"]
    expected_sr_hz = 16_000

    def __init__(self, text: str = "hello world", language: str | None = None) -> None:
        super().__init__()
        self._text = text
        self._language = language

    def load(self) -> None:
        self.state = ModelState.READY

    def transcribe(self, samples: np.ndarray) -> ModelOutput:
        if samples.size == 0 or float(np.abs(samples).max(initial=0.0)) < 1e-6:
            return ModelOutput(text="", language=self._language)
        return ModelOutput(text=self._text, language=self._language)


class SecondaryStubModel(StubModel):
    identifier = "stub-2"
    name = "Stub Secondary"


@pytest.fixture(autouse=True)
def _reset_settings() -> Iterator[None]:
    reset_settings_for_tests()
    yield
    reset_settings_for_tests()


@pytest.fixture
def stub_model() -> StubModel:
    return StubModel()


@pytest.fixture
def app(stub_model: StubModel):
    from asr.app import create_app

    application = create_app(models=[stub_model])
    return application


@pytest.fixture
def app_two_stubs():
    from asr.app import create_app

    primary = StubModel()
    secondary = SecondaryStubModel(text="goodbye world")
    return create_app(models=[primary, secondary])


@pytest.fixture
async def client(app):
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        async with _lifespan(app):
            yield c


from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def _lifespan(app):
    """Manually drive the FastAPI lifespan for the in-process test client."""
    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
        await stack.enter_async_context(app.router.lifespan_context(app))
        yield


def make_silence_wav_bytes(seconds: float = 1.0, sr: int = 16_000) -> bytes:
    import io

    import soundfile as sf

    samples = np.zeros(int(seconds * sr), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def make_tone_wav_bytes(
    seconds: float = 1.0, sr: int = 16_000, freq_hz: float = 440.0, channels: int = 1
) -> bytes:
    import io

    import soundfile as sf

    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False, dtype=np.float32)
    mono = (0.1 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    if channels == 1:
        samples = mono
    else:
        samples = np.stack([mono] * channels, axis=1)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()
