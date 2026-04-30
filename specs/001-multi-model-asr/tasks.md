---
description: "Task list for feature 001-multi-model-asr"
---

# Tasks: Multi-Model Speech Transcription Application

**Input**: Design documents from `/specs/001-multi-model-asr/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/openapi.json, quickstart.md

**Tests**: REQUIRED. Constitution III mandates tests on every critical path; Constitution IV mandates four integration layers. Test tasks are interleaved with implementation tasks within each story phase, not optional.

**Organization**: Tasks are grouped by user story so each story can be independently implemented, tested, and demonstrated.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (US1, US2, US3); omitted for Setup, Foundational, and Polish phases
- File paths are absolute repo-relative paths (no leading `/`)

## Path Conventions

Repository root: `/Users/michaelpeters/go/src/github.com/searchandrescuegg/asr`
- Production code: `src/asr/...`
- Tests: `tests/...`
- API contract source-of-truth: `specs/001-multi-model-asr/contracts/openapi.json`
- Audio fixtures: `audio/` (existing)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Land the new package layout, the required dependencies, and the test/CI plumbing before any business logic is written.

- [X] T001 Create `src/asr/` package skeleton (`__init__.py` files for `asr`, `asr.audio`, `asr.models`, `asr.pipeline`, `asr.api`, `asr.api.v1`, `asr.ui`, `asr.observability`); also create empty `tests/` subtree (`tests/__init__.py`, `tests/contract/`, `tests/unit/`, `tests/integration/`, `tests/e2e/`, `tests/perf/` with `__init__.py` in each)
- [X] T002 Update `pyproject.toml` to add runtime deps (`transformers>=4.45,<5.0`, `sentencepiece`, `soundfile`, `structlog`, `prometheus-client`, `pynvml`, `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`, `opentelemetry-instrumentation-fastapi`, `pydantic-settings`) and a `[dependency-groups].test` group (`pytest`, `pytest-asyncio`, `httpx`, `testcontainers`); add `[project.scripts]` with `asr = "asr.__main__:main"` and `dump-openapi = "asr.scripts.dump_openapi:main"` (the project currently has no `[project.scripts]` table). Run `uv lock` and commit `uv.lock`. Note: the `transformers` major.minor pin is required because SeamlessM4Tv2 APIs have churned across major versions (research R1).
- [X] T003 [P] Configure pytest in `pyproject.toml`: register markers `gpu_required`, `model_required`, `slo`; set `testpaths = ["tests"]`; default to `addopts = "-m 'not gpu_required and not model_required and not slo'"` so the standard run is offline/CPU-safe.
- [X] T004 [P] Update `Dockerfile` entrypoint from `python main.py` to `python -m asr`; ensure `ffmpeg` is installed (audio decode); copy `src/` instead of root files; verify image still builds.
- [X] T005 [P] Update `.github/workflows/pull_request.yaml` to add a `tests` job that runs `uv run pytest` (CPU, default markers only) alongside the existing `lint` and `commitlint` jobs.
- [X] T057 [P] Author `audio/REFERENCE.md` enumerating the standard reference clip set (filename, source/license, expected text, original sample rate, channel count, duration). At minimum: a short English speech clip (the SC-003 fixture used by both Parakeet and Seamless smoke tests), a silence clip (US1.3 / `no_speech_detected`), a stereo clip (FR-009 downmix path), and a non-16-kHz clip (FR-010 resample path). All other tasks that name `audio/<x>` MUST be consistent with this manifest.

**Checkpoint**: `uv sync && uv run pytest` runs and reports "no tests collected" without errors; `docker build .` succeeds; `audio/REFERENCE.md` is the source of truth for fixture provenance.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the cross-cutting primitives every user story depends on — config, observability, audio pipeline, model abstraction, error envelope, app factory. No story can begin before this phase is complete.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T006 [P] Implement `src/asr/config.py` with `pydantic-settings` `Settings` class: `ASR_HOST`, `ASR_PORT`, `ASR_ALLOW_CPU`, `ASR_MAX_FILE_BYTES`, `ASR_MAX_AUDIO_SECONDS`, `ASR_QUEUE_DEPTH`, `ASR_DEFAULT_MODEL`, `ASR_ENABLED_MODELS`. Defaults per `research.md` R10.
- [X] T007 [P] Implement `src/asr/observability/logging.py`: structlog JSON config, FastAPI middleware that generates a UUIDv4 correlation ID, binds it to a contextvar, sets `X-Correlation-Id` response header, and clears the contextvar after the response. Per Constitution V.
- [X] T008 [P] Implement `src/asr/observability/metrics.py`: prometheus collectors named in `research.md` R5 (`asr_requests_total`, `asr_request_duration_seconds`, `asr_audio_duration_seconds`, `asr_queue_depth`, `asr_queue_rejections_total`, `asr_gpu_memory_used_bytes`, `asr_gpu_utilization_ratio`); a `/metrics` route factory; `pynvml`-backed GPU gauges that no-op gracefully when CUDA is absent.
- [X] T009 [P] Implement `src/asr/observability/tracing.py`: OTel SDK init reading `OTEL_EXPORTER_OTLP_ENDPOINT` (no-op when unset); `instrument_fastapi(app)` helper; `tracer.start_as_current_span` context manager that propagates the contextvar correlation ID as a span attribute.
- [X] T010 [P] Implement `src/asr/api/errors.py`: `ErrorCode` str-enum with the exact eight values from `data-model.md`; `ErrorEnvelope` pydantic model; `TranscriptionError` exception class carrying a code + optional details dict; FastAPI `exception_handler` that maps `TranscriptionError` to the right HTTP status + envelope body.
- [X] T011 [P] Implement `src/asr/audio/hash.py`: `sha256_hex(data: bytes) -> str`. Pure function.
- [X] T012 [P] Implement `src/asr/audio/decode.py`: `decode(data: bytes, *, max_seconds: float) -> DecodedAudio`. Uses `soundfile`/`librosa` per research R2; downmixes to mono via channel-mean; resamples to 16 kHz; raises `TranscriptionError(INVALID_FORMAT)` on decode failure and `TranscriptionError(AUDIO_TOO_LONG)` on duration overflow. Returns the `DecodedAudio` dataclass defined in `data-model.md`.
- [X] T013 [P] Implement `src/asr/models/base.py`: `ASRModel` abstract base class with `identifier`, `name`, `vendor`, `languages`, `expected_sr_hz`, `state`, `last_error` attributes and `load()`, `warm_up()`, `transcribe(samples: np.ndarray) -> ModelOutput` abstract methods; `ModelOutput` dataclass with `text` and optional `language`; `ModelState` enum.
- [X] T014 Implement `src/asr/models/registry.py`: `ModelRegistry` with `register(model)`, `list_available()` (excludes FAILED), `list_all()`, `get(identifier)` (raises `MODEL_NOT_FOUND` or `MODEL_UNAVAILABLE`), `get_default()` (uses `Settings.ASR_DEFAULT_MODEL`, raises `NO_DEFAULT_MODEL` if absent). Depends on T013.
- [X] T015 Implement `src/asr/pipeline/queue.py`: per-model `asyncio.Semaphore(1)` and bounded `asyncio.Queue` keyed by model identifier; `try_acquire(model_id)` raises `MODEL_BUSY` immediately when full; metric `asr_queue_depth` updated before enqueue and `asr_queue_rejections_total` incremented on overflow. Depends on T008, T010.
- [X] T016 Implement `src/asr/pipeline/transcribe.py`: `async def transcribe_audio(file_bytes, requested_model) -> TranscriptionResult` orchestrator following the lifecycle in `data-model.md` (hash → decode → resolve → enqueue → run via `asyncio.to_thread` → build result). Emits OTel spans per stage, structured log lines with required fields (no transcription text per FR-013). Depends on T010, T011, T012, T014, T015.
- [X] T017 [P] Implement `src/asr/app.py`: `create_app() -> FastAPI` factory that installs the correlation-ID middleware (T007), the OTel FastAPI instrumentation (T009), the `TranscriptionError` exception handler (T010), the `/metrics` route (T008), and a lifespan that loads enabled models from the registry on startup and unloads on shutdown. The route modules from later phases plug into this factory.
- [X] T018 [P] Implement `src/asr/__main__.py`: parse settings, fail fast if CUDA is unavailable and `ASR_ALLOW_CPU` is false (log WARN if true), call `uvicorn.run(create_app, factory=True, ...)`.
- [X] T019 Add `tests/conftest.py`: `StubModel` fixture (always-`READY` `ASRModel` returning a fixed string), `app` fixture using `create_app` with the stub registered, `client` fixture wrapping `httpx.AsyncClient(app=app)`. Depends on T013, T014, T017.

**Checkpoint**: Foundation ready — user story implementation can now begin.

---

## Phase 3: User Story 1 - Quickly transcribe an audio file (Priority: P1) 🎯 MVP

**Goal**: Upload an audio file, get back a transcription using the default model (Parakeet), with the busy-state UI behavior and structured errors for invalid input.

**Independent Test**: From a fresh deployment with Parakeet registered, upload `audio/sample.wav` through the UI; transcription text appears within 30 s and matches the expected text. Upload a JPEG and observe the `INVALID_FORMAT` error envelope; upload `audio/silence.wav` and observe `no_speech_detected: true`.

### Tests for User Story 1 (write first; required by Constitution III for critical paths)

- [X] T020 [P] [US1] Unit test `tests/unit/test_audio_hash.py`: determinism (`sha256_hex` returns the same hex for the same bytes); length (64 chars); two distinct inputs produce different hashes.
- [X] T021 [P] [US1] Unit test `tests/unit/test_audio_decode.py`: decode WAV → mono 16 kHz; stereo → mono with `downmix_applied=True`; 44.1 kHz → 16 kHz with `resample_applied=True`; non-audio bytes raise `TranscriptionError(INVALID_FORMAT)`; long audio raises `TranscriptionError(AUDIO_TOO_LONG)`.
- [X] T022 [P] [US1] Unit test `tests/unit/test_observability.py`: a synthetic transcription pipeline run produces a log line with the required fields (`correlation_id`, `model_name`, `audio_duration_s`, `inference_duration_s`, `status`, `audio_sha256`) and does NOT contain the transcription text or the audio bytes.
- [X] T023 [P] [US1] Integration test `tests/integration/test_api_transcribe.py::test_default_model_success`: POST `/api/v1/transcribe` with the stub model registered as default, fixture audio bytes; expect 200, body matches `TranscriptionResult` schema, `model` equals stub identifier, `correlation_id` matches `X-Correlation-Id` header.
- [X] T024 [P] [US1] Integration test `tests/integration/test_api_transcribe.py::test_invalid_format_error`: POST `/api/v1/transcribe` with non-audio bytes; expect 400, body matches `ErrorEnvelope`, `code == INVALID_FORMAT`.
- [X] T025 [P] [US1] Integration test `tests/integration/test_api_transcribe.py::test_no_speech_flag`: POST silence-only audio; expect 200, `text == ""`, `no_speech_detected == True`.

### Implementation for User Story 1

- [X] T026 [P] [US1] Implement `src/asr/api/v1/schemas.py`: pydantic models for `TranscribeRequest`, `TranscriptionResult` matching `contracts/openapi.json` (fields, types, nullables). No timing/segment/word fields per the 2026-04-29 clarification.
- [X] T027 [US1] Implement `src/asr/api/v1/routes.py::transcribe`: `POST /api/v1/transcribe` endpoint that reads the multipart upload (size check against `ASR_MAX_FILE_BYTES` → `FILE_TOO_LARGE`), passes bytes + optional `model` form field to `pipeline.transcribe.transcribe_audio`, returns `TranscriptionResult`. Mount the v1 router into `create_app`. Depends on T016, T017, T026.
- [X] T028 [US1] Implement `src/asr/models/parakeet.py`: `ParakeetModel(ASRModel)` adapter wrapping `nemo.collections.asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")`. `transcribe(samples)` returns `ModelOutput(text=..., language=None)`; `load()` sets state to LOADING then READY (or FAILED with `last_error`). Depends on T013.
- [X] T029 [US1] Wire Parakeet into the lifespan in `src/asr/app.py`: when `ASR_ENABLED_MODELS` includes `parakeet-tdt-0.6b-v2`, instantiate `ParakeetModel`, call `load()`, register with the `ModelRegistry`. Depends on T017, T028.
- [X] T030 [US1] Implement `src/asr/ui/gradio_app.py`: minimal Gradio Blocks UI with file picker, submit button, result panel; submit handler calls `pipeline.transcribe.transcribe_audio` directly (same code path as the API per Constitution II); disables submit + shows busy state during in-flight, replaces with text on completion; no cancel button. Result panel renders the transcribed text with a small caption stating which model produced it (e.g., "via NVIDIA Parakeet-TDT 0.6B v2") and exposes Gradio's built-in copy-to-clipboard affordance. When `no_speech_detected` is true, the panel renders the explicit "No speech detected" message instead of an empty box (US1.3). Mount under `/` in `create_app`. Depends on T016, T017.

**Checkpoint**: At this point, User Story 1 is fully functional — Parakeet transcribes audio via both the API and the UI. The MVP works.

---

## Phase 4: User Story 2 - Choose which model performs the transcription (Priority: P2)

**Goal**: Add Seamless as a second registered model, expose model selection in the API and UI, surface the model in every response.

**Independent Test**: With Parakeet and Seamless both registered, upload the same audio twice — once with `model=parakeet-tdt-0.6b-v2`, once with `model=seamless-m4t-v2`. Both succeed; each response identifies the producing model. Submit `model=does-not-exist` and observe the `MODEL_NOT_FOUND` error envelope.

### Tests for User Story 2

- [X] T031 [P] [US2] Unit test `tests/unit/test_model_registry.py`: register two stubs; `get_default` returns the configured default; `get("does-not-exist")` raises `MODEL_NOT_FOUND`; a FAILED stub is excluded from `list_available` but present in `list_all`; `get_default` raises `NO_DEFAULT_MODEL` when default is unavailable.
- [X] T032 [P] [US2] Unit test `tests/unit/test_queue.py`: two requests for the same model serialize (second waits for first); requests for different models overlap; depth=1 + immediate second submit raises `MODEL_BUSY`; queue depth gauge and rejection counter update correctly.
- [X] T033 [P] [US2] Integration test `tests/integration/test_api_transcribe.py::test_explicit_model_selection`: POST with `model=stub-2` registered alongside `stub-1`; response `model` field equals `stub-2`.
- [X] T034 [P] [US2] Integration test `tests/integration/test_api_models.py::test_list_models`: GET `/api/v1/models`; response shape matches `ModelListResponse`; `default` matches `ASR_DEFAULT_MODEL`; only READY models appear in `models`.
- [X] T035 [P] [US2] Integration test `tests/integration/test_api_transcribe.py::test_model_busy`: with `ASR_QUEUE_DEPTH=1`, dispatch two concurrent requests; one succeeds, the other returns 503 `MODEL_BUSY`.
- [X] T036 [US2] GPU smoke test `tests/integration/test_smoke_seamless.py` marked `@pytest.mark.gpu_required @pytest.mark.model_required`: load the real Seamless model, transcribe the SC-003 reference clip from `audio/REFERENCE.md`, assert (a) `abs(measured_wer - published_wer) <= 0.05` versus Seamless's published English WER recorded in the reference manifest, and (b) `response.language == "eng"` (Seamless reports detected language; Parakeet does not).

### Implementation for User Story 2

- [X] T037 [P] [US2] Extend `src/asr/api/v1/schemas.py` with `ModelDescriptor` and `ModelListResponse` matching `contracts/openapi.json`.
- [X] T038 [US2] Add `GET /api/v1/models` route in `src/asr/api/v1/routes.py` returning `ModelListResponse` (registry's `list_all` mapped through `ModelDescriptor`). Depends on T037.
- [X] T039 [P] [US2] Implement `src/asr/models/seamless.py`: `SeamlessModel(ASRModel)` adapter using `transformers.SeamlessM4Tv2ForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large")` + `AutoProcessor` (research R1). Returns `ModelOutput(text=..., language=detected_lang)`. Depends on T013.
- [X] T040 [US2] Wire Seamless into the lifespan in `src/asr/app.py` analogously to T029. Depends on T017, T039.
- [X] T041 [US2] Update `src/asr/ui/gradio_app.py` to add a model-selector dropdown populated from `ModelRegistry.list_available()`, defaulting to the configured default. Pass selection through to `transcribe_audio`. Depends on T030, T038.

**Checkpoint**: Both Parakeet and Seamless work end-to-end via API and UI; users can pick either; queue back-pressure is enforced.

---

## Phase 5: User Story 3 - Programmatic transcription via API (Priority: P3)

**Goal**: Solidify the API as a first-class, contracted surface — checked-in OpenAPI schema, freshness gate, healthz endpoint, comprehensive error-envelope contract tests, and a dump script for the contract.

**Independent Test**: Using `curl` (no UI), exercise every documented endpoint and every documented error code; assert each response matches the checked-in `contracts/openapi.json`.

### Tests for User Story 3

- [X] T042 [P] [US3] Contract test `tests/contract/test_openapi_freshness.py`: load `create_app().openapi()`, normalize (sort keys), load `specs/001-multi-model-asr/contracts/openapi.json`, deep-equal assert. Failure message instructs the developer to run the dump script.
- [X] T043 [P] [US3] Contract test `tests/contract/test_transcribe_contract.py`: for every documented response code on `POST /api/v1/transcribe` (200, 400 INVALID_FORMAT, 400 AUDIO_TOO_LONG, 404 MODEL_NOT_FOUND, 413 FILE_TOO_LARGE, 500 TRANSCRIPTION_FAILED, 503 MODEL_UNAVAILABLE, 503 MODEL_BUSY, 503 NO_DEFAULT_MODEL), produce the condition and assert the response body validates against the corresponding schema in `contracts/openapi.json`.
- [X] T044 [P] [US3] Contract test `tests/contract/test_models_contract.py`: GET `/api/v1/models` body validates against `ModelListResponse`.
- [X] T045 [P] [US3] Unit test `tests/unit/test_errors.py`: each `ErrorCode` maps to the documented HTTP status; envelope serialization includes `code`, `message`, `correlation_id`, optional `details`; `details` never contains `text`, `filename`, or raw bytes (asserted via key whitelist).

### Implementation for User Story 3

- [X] T046 [US3] Implement `src/asr/scripts/dump_openapi.py` (and `src/asr/scripts/__init__.py`): `python -m asr.scripts.dump_openapi` writes `create_app().openapi()` (sorted keys, 2-space indent) to `specs/001-multi-model-asr/contracts/openapi.json`. Wire into `[project].scripts` as `dump-openapi`.
- [X] T047 [P] [US3] Add `GET /api/v1/healthz` route in `src/asr/api/v1/routes.py` returning `{"status": "ok"}` (matches OpenAPI contract).
- [X] T048 [US3] Run `uv run dump-openapi`, diff against the existing `contracts/openapi.json`, reconcile any drift (the checked-in file should already match by design; this task verifies). Depends on T046, T047.

**Checkpoint**: API is fully contract-tested; OpenAPI schema is the enforced source of truth.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: The four-layer integration suite required by Constitution IV, the performance-regression gate required by Constitution VI, and project hygiene.

- [X] T049 [P] Containerized E2E test `tests/e2e/test_container.py` using `testcontainers`: build the image from `Dockerfile`, run it with `ASR_ALLOW_CPU=1`, hit `/api/v1/healthz` then `/api/v1/transcribe` with a fixture audio, assert 200 and a non-empty transcription. Constitution IV layer 3.
- [X] T050 [P] CUDA-presence test `tests/integration/test_cuda_presence.py`: with `ASR_ALLOW_CPU` unset and CUDA absent, the app exits non-zero at startup; with `ASR_ALLOW_CPU=1`, startup succeeds and a WARN log is emitted. Constitution IV layer 4.
- [X] T051 [P] Real-model smoke test `tests/integration/test_smoke_parakeet.py` marked `@pytest.mark.gpu_required @pytest.mark.model_required`: load the real Parakeet model, transcribe the SC-003 reference clip from `audio/REFERENCE.md`, assert `abs(measured_wer - published_wer) <= 0.05` versus Parakeet's published English WER recorded in the reference manifest, and assert `response.language is None` (Parakeet does not report a detected language). Constitution IV layer 2 (Parakeet half; Seamless half is T036).
- [X] T052 Performance regression suite `tests/perf/test_slos.py` marked `@pytest.mark.slo`: load a fixture set, compute p95 RTF per model, p95 API overhead, cold-start time, steady-state VRAM; load `tests/perf/baseline.json`; fail if any metric regresses by > 10%. Initial `tests/perf/baseline.json` captured from a one-off GPU run; commit it.
- [X] T053 [P] Add a GPU-runner job to `.github/workflows/pull_request.yaml` that runs `uv run pytest -m "gpu_required or model_required or slo"`. Mark it `if: github.event.pull_request.head.repo.full_name == github.repository` so forks don't fail.
- [X] T054 [P] Delete the obsolete top-level `main.py`. The new entrypoint is `python -m asr` (T018).
- [X] T055 [P] Update `README.md` to point at `specs/001-multi-model-asr/quickstart.md` and the constitution.
- [X] T058 [P] Pluggability test `tests/integration/test_pluggability.py` (verifies FR-014 / SC-004): defines a `ThirdPartyStubModel(ASRModel)` *inside the test file*, registers it via `app.dependency_overrides` or a test-scoped lifespan hook, then exercises `GET /api/v1/models` (asserting the stub appears) and `POST /api/v1/transcribe` with `model=<stub-id>` (asserting the response uses it). The test MUST NOT import or modify any source under `src/asr/api/`, `src/asr/ui/`, or `src/asr/pipeline/` — only `src/asr/models/base.py` and `src/asr/models/registry.py`. A static import-graph check at the top of the test module asserts this constraint, so a future regression that smuggles model-specific code into transport/UI breaks the test loudly.
- [ ] T056 Run the full test suite (`uv run pytest && uv run pytest -m "gpu_required or model_required"` on a GPU host); confirm all four constitutional integration layers pass and the FR-014 pluggability test (T058) passes.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately.
- **Foundational (Phase 2)**: Depends on Setup. Blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational. The MVP.
- **User Story 2 (Phase 4)**: Depends on Foundational. Independent of US1 in principle, but in practice US1 is finished first since it carries the default flow.
- **User Story 3 (Phase 5)**: Depends on Foundational + at least US1 (the API surface must exist to test it). The OpenAPI contract is finalized once both API endpoints from US1 and US2 exist; running US3 after US2 avoids reconciling the contract twice.
- **Polish (Phase 6)**: Depends on US1 (E2E needs the API), US2 (real-model smoke for Seamless), and US3 (contract tests).

### Within Each User Story

- Tests written in the same PR as the implementation they cover (Constitution III; not strictly test-first).
- Models before services; services before endpoints; endpoints before UI integration.
- Story complete and demonstrable before moving on.

### Parallel Opportunities

**Phase 1 (Setup)**: T003, T004, T005, T057 can run in parallel after T001 + T002.

**Phase 2 (Foundational)**: T006, T007, T008, T009, T010, T011, T012, T013 are all `[P]` — eight independent files. T014 depends on T013; T015 depends on T008+T010; T016 depends on T010+T011+T012+T014+T015; T017 depends on T007+T008+T009+T010; T018 depends on T017; T019 depends on T013+T014+T017.

**Phase 3 (US1)**: All five test tasks (T020–T025) are `[P]` against each other. T026 is `[P]` with the test tasks. T027 depends on T016+T017+T026. T028 is `[P]` with T026/T027. T029 depends on T017+T028. T030 depends on T016+T017.

**Phase 4 (US2)**: All test tasks T031–T036 are `[P]` against each other. T037 is `[P]` with the test tasks. T038 depends on T037. T039 is `[P]`. T040 depends on T039. T041 depends on T030+T038.

**Phase 5 (US3)**: T042–T045 all `[P]`. T046 sequential. T047 `[P]`. T048 depends on T046+T047.

**Phase 6 (Polish)**: T049, T050, T051, T053, T054, T055, T058 all `[P]`. T052 standalone. T056 final integration check.

---

## Parallel Example: User Story 1 tests

```bash
# All five US1 tests can be authored in parallel (different files / different test functions in distinct files):
Task: "Unit test tests/unit/test_audio_hash.py" (T020)
Task: "Unit test tests/unit/test_audio_decode.py" (T021)
Task: "Unit test tests/unit/test_observability.py" (T022)
Task: "Integration test tests/integration/test_api_transcribe.py::test_default_model_success" (T023)
Task: "Integration test tests/integration/test_api_transcribe.py::test_invalid_format_error" (T024)
Task: "Integration test tests/integration/test_api_transcribe.py::test_no_speech_flag" (T025)
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1 (Setup): T001 → T002 → T003/T004/T005 in parallel.
2. Phase 2 (Foundational): all 14 tasks per the dep graph.
3. Phase 3 (US1): T020–T030.
4. **STOP and validate**: Parakeet transcribes audio via API and UI; structured errors work; logs/metrics/traces flowing. This is a shippable MVP.

### Incremental Delivery

1. After MVP: Phase 4 (US2) → demoable as "now with model selection + Seamless" → ship.
2. Then Phase 5 (US3) → demoable as "API contract locked + healthz" → ship.
3. Then Phase 6 (Polish) → containerized E2E + CUDA-presence + perf gate → constitutional compliance verified.

### Constitution Mapping

| Constitution principle | Tasks that satisfy it |
|---|---|
| I. Multi-Model by Design | T013, T014, T028, T029, T039, T040, T058 (pluggability proof) |
| II. API and UI as First-Class Surfaces | T026, T027, T030, T038, T041, T046, T047, T048 (contract & dump) |
| III. Pragmatic Test Discipline | T020–T025, T031–T035, T042–T045 (per-story coverage) |
| IV. Layered Integration Testing | Layer 1 (contract): T042–T045; Layer 2 (real-model smoke): T036, T051; Layer 3 (containerized E2E): T049; Layer 4 (CUDA-presence): T050 |
| V. Full-Stack Observability | T007, T008, T009, T022, T032 |
| VI. Performance with Enforced SLOs | T052, T053 |

---

## Notes

- Tasks marked `[P]` can run in parallel. Within a phase, parallelism is bounded by the dependency graph above.
- Every story phase yields an independently demonstrable increment.
- Tests are interleaved with implementation (Constitution III is "test-required, not test-first") — author each story's tests in the same PR as its implementation.
- File paths in this document are authoritative; if a task references a path, that's the file to create or modify.
- Avoid: adding internal validation beyond the boundaries enumerated in `data-model.md` (research §R11). Trust pydantic at the API boundary, the registry inside the pipeline, and the adapter contract inside model code.
