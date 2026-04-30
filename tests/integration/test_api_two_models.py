"""US2 integration: two stubs, explicit model selection, and MODEL_BUSY."""

from __future__ import annotations

import asyncio

import pytest

from tests.conftest import make_tone_wav_bytes


@pytest.fixture
async def two_stub_client(monkeypatch):
    monkeypatch.setenv("ASR_DEFAULT_MODEL", "stub-en")
    monkeypatch.setenv("ASR_ENABLED_MODELS", "stub-en,stub-2")
    monkeypatch.setenv("ASR_QUEUE_DEPTH", "1")
    from asr.config import reset_settings_for_tests

    reset_settings_for_tests()

    from contextlib import AsyncExitStack

    from httpx import ASGITransport, AsyncClient

    from asr.app import create_app
    from tests.conftest import SecondaryStubModel, StubModel

    primary = StubModel()
    secondary = SecondaryStubModel(text="goodbye world")
    app = create_app(models=[primary, secondary])

    async with AsyncExitStack() as stack:
        await stack.enter_async_context(app.router.lifespan_context(app))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c


@pytest.mark.asyncio
async def test_explicit_model_selection(two_stub_client):
    audio = make_tone_wav_bytes(seconds=0.3)
    files = {"file": ("clip.wav", audio, "audio/wav")}

    resp = await two_stub_client.post(
        "/api/v1/transcribe", files=files, data={"model": "stub-2"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "stub-2"
    assert body["text"] == "goodbye world"


@pytest.mark.asyncio
async def test_model_busy_when_queue_full(two_stub_client, monkeypatch):
    """With ASR_QUEUE_DEPTH=1 a 3rd concurrent request is rejected.

    Cap = 1 in-flight + 1 waiter = 2 in-system. The third arrival overflows.
    A monkeypatched transcribe blocks until release so the test can stack
    multiple concurrent requests deterministically.
    """

    import time

    release = asyncio.Event()

    from tests.conftest import StubModel

    original = StubModel.transcribe

    def slow(self, samples):
        deadline = time.time() + 5.0
        while not release.is_set() and time.time() < deadline:
            time.sleep(0.01)
        return original(self, samples)

    monkeypatch.setattr(StubModel, "transcribe", slow)

    audio = make_tone_wav_bytes(seconds=0.2)
    files_payload = {"file": ("clip.wav", audio, "audio/wav")}

    async def post():
        return await two_stub_client.post(
            "/api/v1/transcribe",
            files=files_payload,
            data={"model": "stub-en"},
        )

    # Stagger the three POSTs: t1 enters flight, t2 enters wait queue, t3 overflows.
    t1 = asyncio.create_task(post())
    await asyncio.sleep(0.05)
    t2 = asyncio.create_task(post())
    await asyncio.sleep(0.05)
    t3 = asyncio.create_task(post())

    # Wait for t3 to come back; with depth=1 it should reject quickly.
    third_resp = await asyncio.wait_for(t3, timeout=2.0)
    assert third_resp.status_code == 503, third_resp.text
    assert third_resp.json()["code"] == "MODEL_BUSY"

    release.set()
    first_resp = await asyncio.wait_for(t1, timeout=10.0)
    second_resp = await asyncio.wait_for(t2, timeout=10.0)
    assert first_resp.status_code == 200
    assert second_resp.status_code == 200
