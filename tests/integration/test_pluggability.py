"""T058 — FR-014 / SC-004 pluggability proof.

A new ASR model can be added by registering it from outside the
production package, without touching `asr.api.*`, `asr.ui.*`, or
`asr.pipeline.*`. This test plays the role of "we added a model" by
defining a brand-new adapter inline and registering it via the existing
`create_app(models=...)` extension point — the same one Parakeet and
Seamless use.

It also performs a static import-graph check: the transport, UI, and
pipeline packages must NOT import any concrete model adapter. They may
only depend on `asr.models.base` and `asr.models.registry`.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from asr.app import create_app
from asr.models.base import ASRModel, ModelOutput, ModelState
from tests.conftest import make_tone_wav_bytes

REPO = pathlib.Path(__file__).resolve().parents[2]


def _module_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            out.add(mod)
            for alias in node.names:
                out.add(f"{mod}.{alias.name}" if mod else alias.name)
    return out


_FORBIDDEN_FOR_TRANSPORT_UI_PIPELINE = {
    "asr.models.parakeet",
    "asr.models.seamless",
}


def test_transport_ui_pipeline_do_not_import_concrete_model_adapters():
    boundaries = ["src/asr/api", "src/asr/ui", "src/asr/pipeline"]
    offenders = []
    for sub in boundaries:
        for path in (REPO / sub).rglob("*.py"):
            imports = _module_imports(path)
            bad = imports & _FORBIDDEN_FOR_TRANSPORT_UI_PIPELINE
            if bad:
                offenders.append((str(path.relative_to(REPO)), bad))
    assert not offenders, (
        "transport/UI/pipeline code imports a concrete model adapter "
        "(violates Constitution I / FR-014): " + repr(offenders)
    )


class ThirdPartyStubModel(ASRModel):
    identifier = "third-party-stub"
    name = "Third-Party Stub (test)"
    vendor = "test"
    languages = ["en"]
    expected_sr_hz = 16_000

    def load(self) -> None:
        self.state = ModelState.READY

    def transcribe(self, samples: np.ndarray) -> ModelOutput:
        return ModelOutput(text="third party output", language=None)


@pytest.mark.asyncio
async def test_register_third_party_model_end_to_end(monkeypatch):
    monkeypatch.setenv("ASR_DEFAULT_MODEL", "third-party-stub")
    monkeypatch.setenv("ASR_ENABLED_MODELS", "third-party-stub")
    from asr.config import reset_settings_for_tests

    reset_settings_for_tests()

    from contextlib import AsyncExitStack

    from httpx import ASGITransport, AsyncClient

    app = create_app(models=[ThirdPartyStubModel()])

    async with AsyncExitStack() as stack:
        await stack.enter_async_context(app.router.lifespan_context(app))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            list_resp = await c.get("/api/v1/models")
            assert list_resp.status_code == 200
            ids = [m["identifier"] for m in list_resp.json()["models"]]
            assert "third-party-stub" in ids

            audio = make_tone_wav_bytes(seconds=0.2)
            tx = await c.post(
                "/api/v1/transcribe",
                files={"file": ("clip.wav", audio, "audio/wav")},
                data={"model": "third-party-stub"},
            )
            assert tx.status_code == 200
            body = tx.json()
            assert body["model"] == "third-party-stub"
            assert body["text"] == "third party output"
