# Implementation Plan: Multi-Model Speech Transcription Application

**Branch**: `001-multi-model-asr` | **Date**: 2026-04-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-multi-model-asr/spec.md`

## Summary

Restructure the existing single-file ASR server into a modular, multi-model
application that supports both NVIDIA Parakeet (NeMo) and Facebook Seamless-M4T-v2
(Hugging Face Transformers) behind a single `ASRModel` interface. Expose an
HTTP API and a Gradio UI as first-class surfaces with shared pipeline code,
per-model serialized inference, full observability (structured logs +
Prometheus metrics + OpenTelemetry traces), and the four-layer test stack
required by the constitution (contract / real-model smoke / containerized
E2E / GPU-presence). All "defensive programming" called out in the spec —
input validation, structured errors, queue overflow, model-failure handling,
content-hash logging without leaking spoken content — is implemented at
system boundaries (HTTP intake, model adapter boundary) and not duplicated
inside trusted internals.

## Technical Context

**Language/Version**: Python 3.11 (constitutional, `requires-python = "==3.11.*"`)
**Primary Dependencies**: FastAPI (transport), Gradio (UI), Uvicorn (ASGI),
NeMo-toolkit[asr] (Parakeet), Transformers + sentencepiece + soundfile
(Seamless-M4T-v2), librosa (audio decode/resample), pydantic + pydantic-settings
(config & schemas), structlog (logging), prometheus-client (metrics),
opentelemetry-api/sdk/exporter-otlp + opentelemetry-instrumentation-fastapi
(traces), torch (CUDA runtime; pinned by `pyproject.toml`).
**Storage**: None (transient in-memory only; constitutional — no persistence in v1).
**Testing**: pytest, pytest-asyncio, httpx (API contract), testcontainers
(E2E container), pytest markers `gpu_required` / `model_required` to gate
real-model and CUDA tests, plus a standing fixtures directory for golden audio.
**Target Platform**: Linux server with single CUDA-capable NVIDIA GPU
(≥ 8 GB VRAM for steady-state two-model load); CPU-only allowed in dev/test
behind `ASR_ALLOW_CPU=1`.
**Project Type**: web-service (single Python project; FastAPI app with mounted
Gradio UI under one process).
**Performance Goals** (per Constitution VI):
  - p95 RTF ≤ 0.5 over the standard fixture set per model
  - p95 API overhead (decode + serialize, exclusive of inference) ≤ 100 ms
  - Cold-start model load ≤ 60 s
  - Steady-state ≤ 8 GB GPU VRAM idle after warm-up
**Constraints**:
  - File size cap default 100 MB; duration cap default 10 min (both configurable).
  - Per-model in-flight = 1; bounded per-model wait queue (default depth 4)
    rejects with structured `MODEL_BUSY` error when full (FR-016/FR-017).
  - Audio normalized to 16 kHz mono before inference (constitution + FR-009/010).
  - Logs MUST omit transcription text, original filename, audio bytes; MUST
    include SHA-256 content-hash (FR-013, clarified 2026-04-29).
  - All public endpoints under `/api/v1/...`; OpenAPI schema checked into
    `specs/001-multi-model-asr/contracts/openapi.json` (constitution II).
**Scale/Scope**: Single-host, single-user/small-team operation (no auth,
no multi-tenant, no rate limiting per spec assumptions). Two registered
models in v1; new models pluggable via `ASRModel` adapter without API/UI
changes (FR-014 / SC-004).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Each gate maps to a principle in `.specify/memory/constitution.md`. Mark each
PASS / FAIL / N/A and justify any FAIL in the Complexity Tracking table.

- [x] **I. Multi-Model by Design** — **PASS.** Concrete adapters
      `asr.models.parakeet.ParakeetModel` and `asr.models.seamless.SeamlessModel`
      sit behind `asr.models.base.ASRModel` (abstract). Transport (`asr.api.*`),
      UI (`asr.ui.gradio_app`), and orchestration (`asr.pipeline.*`) import
      only `ASRModel` and the registry — no model-specific imports leak
      out. The current `main.py` direct-imports of `nemo.collections.asr`
      will be removed.
- [x] **II. API and UI as First-Class Surfaces** — **PASS.** Endpoints under
      `/api/v1/...` (`/api/v1/transcribe`, `/api/v1/models`, `/api/v1/healthz`,
      `/metrics`); OpenAPI schema dumped to
      `specs/001-multi-model-asr/contracts/openapi.json` and checked in;
      a contract test (`tests/contract/test_openapi_freshness.py`) fails CI
      if the live FastAPI schema diverges. Gradio UI calls the same
      `pipeline.transcribe.transcribe_audio` function the API uses — no
      bypass.
- [x] **III. Pragmatic Test Discipline** — **PASS.** Critical paths covered
      in the same PR: API contract (200 + 4xx envelopes), audio decode/downmix/
      resample, model registry + default-model resolution, per-model queue +
      overflow, error envelope shape, content-hash log fields. Unit tests for
      non-trivial helpers; integration tests for the four boundary classes.
- [x] **IV. Layered Integration Testing** — **PASS.** All four layers planned:
      (1) API contract via `httpx.AsyncClient` against the in-process app
      with a stub `ASRModel`; (2) real-model smoke per model behind
      `@pytest.mark.gpu_required @pytest.mark.model_required`; (3)
      containerized E2E via `testcontainers` building from the existing
      `Dockerfile`; (4) CUDA-presence test that asserts startup behavior on
      both a GPU host and a CPU-only host with `ASR_ALLOW_CPU=1`.
- [x] **V. Full-Stack Observability** — **PASS.** `structlog` JSON logs
      with per-request correlation ID (middleware-injected), required fields
      `model_name`, `audio_duration_s`, `inference_duration_s`, `status`,
      `audio_sha256`. Prometheus `/metrics` exposes request count, latency
      histogram (overall + per-model), GPU utilization (via `pynvml`), and
      error counters by class. OpenTelemetry spans for the four pipeline
      stages (`request_received → audio_decoded → model_inference →
      response_serialized`); spans carry the same correlation ID.
- [x] **VI. Performance with Enforced SLOs** — **PASS.** SLO regression
      harness (`tests/perf/test_slos.py`) runs the standard fixture set,
      computes RTF/overhead/cold-start/VRAM, and fails if any SLO regresses
      > 10% versus the prior tagged baseline. Initial baseline captured
      from a one-off run on the reference GPU and committed under
      `tests/perf/baseline.json`. Nightly job in CI runs this on the
      GPU runner.

**Result**: All gates PASS. No entries required in Complexity Tracking.

## Project Structure

### Documentation (this feature)

```text
specs/001-multi-model-asr/
├── plan.md              # This file
├── spec.md              # Feature spec (already written)
├── research.md          # Phase 0 output (this command)
├── data-model.md        # Phase 1 output (this command)
├── quickstart.md        # Phase 1 output (this command)
├── contracts/
│   └── openapi.json     # Source-of-truth API schema (Phase 1)
├── checklists/
│   └── requirements.md  # Spec quality checklist (already written)
└── tasks.md             # Phase 2 output (NOT created here — /speckit.tasks)
```

### Source Code (repository root)

```text
src/
└── asr/
    ├── __init__.py
    ├── __main__.py                  # `python -m asr` → uvicorn entrypoint
    ├── app.py                       # FastAPI app factory + lifespan (model warm-up, otel init)
    ├── config.py                    # pydantic-settings (env: ASR_*)
    ├── audio/
    │   ├── __init__.py
    │   ├── decode.py                # bytes → mono 16 kHz np.ndarray; raises DecodeError
    │   └── hash.py                  # SHA-256 of raw upload bytes
    ├── models/
    │   ├── __init__.py
    │   ├── base.py                  # ASRModel ABC (load, warm_up, transcribe, identifier, name, languages)
    │   ├── registry.py              # ModelRegistry: register/list/get/default
    │   ├── parakeet.py              # NeMo Parakeet-TDT adapter
    │   └── seamless.py              # Transformers SeamlessM4T-v2 adapter
    ├── pipeline/
    │   ├── __init__.py
    │   ├── queue.py                 # per-model asyncio.Semaphore(1) + bounded asyncio.Queue
    │   └── transcribe.py            # orchestrator: hash → decode → enqueue → run → result
    ├── api/
    │   ├── __init__.py
    │   ├── errors.py                # structured ErrorEnvelope (code, message, details)
    │   └── v1/
    │       ├── __init__.py
    │       ├── routes.py            # POST /transcribe, GET /models, GET /healthz
    │       └── schemas.py           # pydantic request/response models
    ├── ui/
    │   ├── __init__.py
    │   └── gradio_app.py            # Blocks UI; calls pipeline.transcribe directly
    └── observability/
        ├── __init__.py
        ├── logging.py               # structlog config + correlation-ID middleware
        ├── metrics.py               # prometheus collectors + /metrics route
        └── tracing.py               # OTel resource, span helpers, FastAPI instrumentation

tests/
├── conftest.py                      # shared fixtures (StubModel, app, client)
├── contract/
│   ├── test_openapi_freshness.py    # live schema == checked-in openapi.json
│   ├── test_transcribe_contract.py  # response/error envelopes match schema
│   └── test_models_contract.py
├── unit/
│   ├── test_audio_decode.py         # WAV/MP3/FLAC/OGG; stereo→mono; resample
│   ├── test_audio_hash.py           # determinism; collision-free for distinct bytes
│   ├── test_model_registry.py       # registration; default resolution; missing-model error
│   ├── test_queue.py                # serial per model; overflow → MODEL_BUSY; concurrency across models
│   ├── test_errors.py               # error envelope shape and machine-readable codes
│   └── test_observability.py        # log fields present; no transcription text in logs
├── integration/
│   ├── test_api_transcribe.py       # POST /transcribe with stub model (CPU)
│   ├── test_smoke_parakeet.py       # @gpu_required @model_required
│   ├── test_smoke_seamless.py       # @gpu_required @model_required
│   └── test_cuda_presence.py        # startup behavior with/without GPU and ASR_ALLOW_CPU
├── e2e/
│   └── test_container.py            # testcontainers: build, run, hit /api/v1/transcribe
└── perf/
    ├── baseline.json                # prior-tagged-release SLO numbers
    └── test_slos.py                 # RTF, overhead, cold-start, VRAM regression gate

audio/                               # golden fixtures (already in repo; expand as needed)
Dockerfile                           # existing; updated for new entrypoint + Seamless deps
pyproject.toml                       # existing; add seamless deps + dev/test extras
```

**Structure Decision**: Single Python package under `src/asr/`. The current
top-level `main.py` is removed; the entrypoint is `python -m asr` (or
`uvicorn asr.app:create_app --factory`). The `audio/` directory of
existing fixtures is reused. Tests live at the repository root in `tests/`,
mirroring the production module structure.

## Complexity Tracking

> No Constitution Check violations. Table intentionally empty.
