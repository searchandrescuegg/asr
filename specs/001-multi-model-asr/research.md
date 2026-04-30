# Phase 0 Research: Multi-Model Speech Transcription

**Feature**: `001-multi-model-asr`
**Date**: 2026-04-29

This document resolves the open technical questions surfaced when filling
the Technical Context of `plan.md`. Each entry follows the format:
**Decision** → **Rationale** → **Alternatives considered**.

---

## R1. Seamless-M4T-v2 integration path

**Decision**: Use Hugging Face Transformers' `SeamlessM4Tv2ForSpeechToText`
class loaded from `facebook/seamless-m4t-v2-large` (or `…-medium` for
tighter VRAM budgets) with `AutoProcessor` for pre-processing. Pin the
`transformers` version in `pyproject.toml`.

**Rationale**: The Transformers integration is officially supported, ships
with documented inference APIs, integrates directly with `torch.cuda`, and
keeps our dependency footprint to one Hugging Face package. It exposes
exactly the surface our adapter needs (`generate(...)` over a tokenized
audio batch).

**Alternatives considered**:
- `seamless-communication` upstream package (Meta) — heavier install
  (fairseq2, custom CUDA ops), more rapid version churn, and overlapping
  dependencies that conflict with NeMo's pins.
- ONNX/CTranslate2 ports — none currently published with parity for the v2
  ASR head; deferred until/unless performance forces it.

---

## R2. Audio decoding & resampling library

**Decision**: Continue using `librosa` for decode + resample to 16 kHz mono,
backed by `soundfile`/`libsndfile` for native formats and `audioread` +
`ffmpeg` for compressed formats (MP3/OGG). `numpy.mean(axis=0)` for
deterministic stereo-to-mono downmix.

**Rationale**: librosa is already in the dep tree and works against all
formats the spec lists (FR-008). Switching to `torchaudio` would gain
~2× decode speed on small files but adds another path through libsox/ffmpeg
that complicates the Dockerfile. We don't yet have evidence decode is the
bottleneck — inference dominates RTF.

**Alternatives considered**:
- `torchaudio.load` — fastest pure-tensor path, but format support depends
  on the host's libsox build; introduces a second resampler implementation
  alongside librosa.
- `pydub` + `numpy` — clean for MP3 but adds an ffmpeg shell-out and
  doesn't natively handle FLAC/OGG without explicit codecs.

The Dockerfile must continue to install `ffmpeg` (already present) for MP3
support via `audioread`.

---

## R3. Per-model concurrency primitive

**Decision**: One `asyncio.Semaphore(1)` per model paired with a bounded
`asyncio.Queue(maxsize=N)` (default `N=4`, configurable via `ASR_QUEUE_DEPTH`).
Inference runs in `asyncio.to_thread(...)` so the event loop stays
responsive. Queue overflow returns immediately with the structured
`MODEL_BUSY` error (FR-017).

**Rationale**: Native to FastAPI's asyncio runtime; no extra thread-pool
or process-pool framework required. The semaphore enforces FR-016 ("at most
one inference in flight per model"). The bounded queue gives back-pressure
without blocking the event loop or drowning GPU memory.

**Alternatives considered**:
- `concurrent.futures.ThreadPoolExecutor(max_workers=1)` per model — works,
  but mixes thread-pool semantics with async, and queueing/back-pressure
  becomes manual.
- `anyio` capacity limiters — adds another async runtime; no benefit over
  stdlib here.
- Process-per-model worker via `multiprocessing` — would isolate failures
  but doubles GPU memory pressure (CUDA contexts) and complicates startup.

---

## R4. Structured logging library

**Decision**: `structlog` configured to emit JSON lines, with a FastAPI
middleware that injects a UUIDv4 correlation ID into each request's
contextvar and onto the response header (`X-Correlation-Id`). The same
ID is set as the OpenTelemetry span attribute and the `correlation_id`
log field.

**Rationale**: structlog produces stable JSON, supports contextvar-based
log enrichment without explicit threading, and is already broadly used in
async Python services. We keep stdlib `logging` underneath so library logs
(uvicorn, transformers) are captured uniformly.

**Alternatives considered**:
- `python-json-logger` — produces JSON but lacks contextvar enrichment;
  more boilerplate to thread the correlation ID through every log site.
- `loguru` — convenient but its bundled formatting and sink model is
  harder to align with OTel + Prometheus middleware patterns.

---

## R5. Metrics

**Decision**: `prometheus-client` exposed at `/metrics` (no path version —
operational endpoint, not part of `/api/v1`). Collectors:
- `asr_requests_total{model, status}` (Counter)
- `asr_request_duration_seconds{model, stage}` (Histogram; stage ∈
  `{decode, inference, total}`)
- `asr_audio_duration_seconds` (Histogram)
- `asr_queue_depth{model}` (Gauge, sampled before enqueue)
- `asr_queue_rejections_total{model}` (Counter)
- `asr_gpu_memory_used_bytes` (Gauge; via `pynvml`)
- `asr_gpu_utilization_ratio` (Gauge; via `pynvml`)

Metric names are stable; renames are MAJOR per Constitution V.

**Rationale**: prometheus-client is the canonical Python instrumentation
library; integrates trivially with FastAPI as a mounted route. The chosen
labels match the cardinality limits of a small-team deployment (model
names are bounded; status is bounded).

**Alternatives considered**:
- OpenTelemetry metrics SDK + OTLP exporter — would unify with the trace
  exporter, but scrape-friendly Prometheus is the de-facto standard for
  the SRE workflows we expect, and `pynvml` integrates more smoothly via
  prometheus collectors.

---

## R6. Distributed tracing

**Decision**: OpenTelemetry SDK with the OTLP/HTTP exporter, plus
`opentelemetry-instrumentation-fastapi` for automatic request spans.
Manual child spans created explicitly inside the pipeline for the four
stages required by the constitution: `request_received`, `audio_decoded`,
`model_inference`, `response_serialized`. Exporter endpoint configurable
via the standard `OTEL_EXPORTER_OTLP_ENDPOINT` env var; tracing is a no-op
when unset (avoids noise in dev).

**Rationale**: Standard, vendor-neutral, zero-config when the env var is
unset, and the FastAPI instrumentation captures middleware behavior we'd
otherwise have to wire by hand. Span attributes carry the same
correlation ID present in logs (Constitution V).

**Alternatives considered**:
- Sentry tracing — vendor-locked; we don't yet need the error-aggregation
  side, and the constitution names OTel by name.
- No tracing in v1 — would violate Constitution V (PASS gate would fail).

---

## R7. OpenAPI source-of-truth & freshness check

**Decision**: A small command (`python -m asr.scripts.dump_openapi`) writes
`app.openapi()` to `specs/001-multi-model-asr/contracts/openapi.json`. A
contract test (`tests/contract/test_openapi_freshness.py`) loads the live
app, calls `app.openapi()`, normalizes (sort keys, drop ephemeral fields
if any), and asserts deep-equality with the checked-in file. CI fails on
divergence.

**Rationale**: Constitution II requires the schema to be the contract
checked into the repo. The freshness test catches accidental schema drift
in PRs and forces a deliberate `dump_openapi` step when intentional.

**Alternatives considered**:
- Hand-written OpenAPI YAML — duplicates work; allows the live app to
  drift from the contract silently.
- Auto-regenerate in CI without checking in — loses the diffability of
  schema changes during review.

---

## R8. Test stack & markers

**Decision**: `pytest` + `pytest-asyncio` + `httpx.AsyncClient` for in-process
API tests. `testcontainers-python` for the E2E layer. Two custom markers
registered in `pyproject.toml`:
- `gpu_required` — skipped when `torch.cuda.is_available()` is `False`.
- `model_required` — skipped when the model weights cannot be loaded
  (deferred via fixture; treats download/disk failure as skip-not-fail
  for CPU CI).

A third marker, `slo`, gates the performance regression suite from the
default `pytest` run.

**Rationale**: Keeps the standard `pytest` run fast and offline-safe on
CPU CI, while letting GPU runners opt in to the heavy layers.

**Alternatives considered**:
- Splitting tests into separate `tox` environments — heavier than needed
  for a single repo; markers + a `pytest -m "not gpu_required"` default
  in CI achieves the same separation.

---

## R9. Default audio normalization (constitution alignment)

**Decision**: Multi-channel inputs are downmixed (not rejected) to mono
deterministically using channel-mean. The response carries a
`downmix_applied: bool` flag. This choice is recorded in the OpenAPI
schema (Constitution: "the choice MUST be documented in the OpenAPI
schema; silent behavior changes are forbidden").

**Rationale**: Downmix is more user-friendly than rejection for the typical
use case (someone records on a stereo phone or laptop and uploads
directly); the surfaced flag preserves the "no silent behavior changes"
principle.

**Alternatives considered**:
- Reject multi-channel with HTTP 400 — better for users who care about
  channel-specific transcription, but no one in scope does, and rejection
  would make the UI flow brittle.

---

## R10. Configuration & secrets

**Decision**: `pydantic-settings` with the `ASR_` env prefix. Settings:
- `ASR_HOST`, `ASR_PORT` (default `0.0.0.0:8000`).
- `ASR_ALLOW_CPU` (default `false`; constitutional opt-in for non-GPU dev).
- `ASR_MAX_FILE_BYTES` (default `100 * 1024 * 1024`).
- `ASR_MAX_AUDIO_SECONDS` (default `600`).
- `ASR_QUEUE_DEPTH` (default `4`).
- `ASR_DEFAULT_MODEL` (default `parakeet-tdt-0.6b-v2`).
- `ASR_ENABLED_MODELS` (CSV list; default `parakeet-tdt-0.6b-v2,seamless-m4t-v2`).
- `OTEL_EXPORTER_OTLP_ENDPOINT` (standard OTel var; unset → tracing no-op).

No secrets are required for v1 (no auth, no external API keys). Hugging
Face model downloads use anonymous access for the chosen public weights.

**Rationale**: Config-via-env is the path of least surprise for a
container-deployed service; pydantic-settings gives type-checked parsing
with explicit defaults and validation.

**Alternatives considered**:
- TOML/YAML config file — adds an artifact to manage and complicates
  Docker overrides without buying anything for a service this size.

---

## R11. "Defensive programming" boundaries

**Decision**: Validate at exactly two boundaries — (a) HTTP intake
(file size, content type, audio decode, duration cap) and (b) model
adapter boundary (model-specific exceptions wrapped in
`TranscriptionError` with a machine-readable code). No internal
re-validation: trust pydantic's parsing, trust adapter contracts, trust
the registry. Helper functions assume their preconditions because the
boundary already enforced them.

**Rationale**: The user explicitly asked for "all defensive programming
noted before". The "noted before" is the spec's edge cases and FR-011/012/
017, not blanket null-checks throughout internal code. Adding redundant
checks inside trusted internals is the anti-pattern called out in our
top-level guidance ("don't add error handling for scenarios that can't
happen"). This decision keeps the surface focused: every error path the
user can hit is named, structured, and tested; internal helpers stay
clean.

**Alternatives considered**:
- Pervasive try/except at every layer — produces a logspam pattern, hides
  real failures, and makes the structured error contract harder to keep
  consistent.
- No structured errors — fails FR-006/012, fails Constitution III's
  "error handling" critical-path coverage.

---

## Resolved unknowns checklist

- [x] How to load Seamless-M4T-v2 in Python (R1)
- [x] Audio decoding stack (R2)
- [x] Per-model serial-queue primitive (R3)
- [x] Logging stack (R4)
- [x] Metrics stack (R5)
- [x] Tracing stack (R6)
- [x] OpenAPI freshness mechanism (R7)
- [x] Test framework + markers (R8)
- [x] Multi-channel handling (R9)
- [x] Configuration surface (R10)
- [x] Scope of defensive checks (R11)

No NEEDS CLARIFICATION markers remain.
