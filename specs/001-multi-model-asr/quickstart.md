# Quickstart: Multi-Model Speech Transcription

**Feature**: `001-multi-model-asr`
**Audience**: A new contributor or operator who wants the service running
locally and a transcription back from each model within ~10 minutes.

This document is the runnable end-to-end story. It also doubles as the
script the `tests/integration/` suite exercises in CI; if a step changes,
the test must change with it.

---

## Prerequisites

- Linux or macOS host (Windows works under WSL2).
- Python 3.11 (the project pins this exactly; `pyenv install 3.11`).
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- `ffmpeg` installed (needed for MP3/OGG decoding).
- For full functionality: an NVIDIA GPU with ≥ 8 GB VRAM and CUDA 12.x
  drivers. CPU-only is supported for development behind an explicit env
  flag (see below); Parakeet/Seamless are slow on CPU and may not meet the
  SLO suite, but they will produce correct output.
- A few minutes of free network bandwidth on first run (model weights
  download from Hugging Face the first time each adapter loads).

---

## 1. Install dependencies

```bash
uv sync
```

This reads `pyproject.toml`, resolves against `uv.lock`, and installs
into `.venv/`. No `pip` or `conda` involved.

---

## 2. Configure the runtime

The service reads `ASR_*` environment variables. Defaults are sane for
GPU machines; the only one a non-GPU dev needs to set is
`ASR_ALLOW_CPU=1`.

```bash
# Required for GPU-less hosts (development only)
export ASR_ALLOW_CPU=1

# Optional overrides (defaults shown)
export ASR_HOST=0.0.0.0
export ASR_PORT=8000
export ASR_MAX_FILE_BYTES=$((100 * 1024 * 1024))   # 100 MB
export ASR_MAX_AUDIO_SECONDS=600                   # 10 minutes
export ASR_QUEUE_DEPTH=4
export ASR_DEFAULT_MODEL=parakeet-tdt-0.6b-v3
export ASR_ENABLED_MODELS=parakeet-tdt-0.6b-v3,seamless-m4t-v2

# Optional: enable distributed tracing (no-op when unset)
# export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

---

## 3. Start the server

```bash
uv run python -m asr
```

You should see structured JSON logs of the form:

```json
{"event": "model_loading", "model": "parakeet-tdt-0.6b-v3", ...}
{"event": "model_ready",   "model": "parakeet-tdt-0.6b-v3", "load_seconds": 12.3, ...}
{"event": "model_loading", "model": "seamless-m4t-v2", ...}
{"event": "model_ready",   "model": "seamless-m4t-v2",   "load_seconds": 41.2, ...}
{"event": "server_ready",  "host": "0.0.0.0", "port": 8000}
```

If `ASR_ALLOW_CPU` is unset and CUDA is unavailable, the process exits
non-zero with a clear error — this is intentional (Constitution: GPU
policy).

---

## 4. Verify with the API

In another terminal:

```bash
# Liveness
curl -s http://localhost:8000/api/v1/healthz | jq
# → {"status": "ok"}

# List models
curl -s http://localhost:8000/api/v1/models | jq
# → {"default": "parakeet-tdt-0.6b-v3", "models": [{"identifier": "parakeet-tdt-0.6b-v3", "state": "READY", ...}, ...]}

# Transcribe with the default model (Parakeet)
curl -s -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@audio/sample.wav" | jq
# → {"text": "...", "model": "parakeet-tdt-0.6b-v3", "audio_duration_s": ..., "no_speech_detected": false, "correlation_id": "..."}

# Transcribe with Seamless explicitly
curl -s -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@audio/sample.wav" \
  -F "model=seamless-m4t-v2" | jq

# Trigger a structured error (oversized file)
dd if=/dev/zero of=/tmp/big.wav bs=1M count=200
curl -s -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@/tmp/big.wav" -i
# → HTTP/1.1 413 ; body: {"code": "FILE_TOO_LARGE", "message": "...", "details": {"limit_bytes": 104857600}, "correlation_id": "..."}
```

Every response carries an `X-Correlation-Id` header that matches the
`correlation_id` field in the body. Quote that ID when reporting issues —
it indexes the structured log line.

---

## 5. Verify with the UI

Open `http://localhost:8000/` in a browser. You should see:

- A model dropdown defaulting to "NVIDIA Parakeet-TDT 0.6B v2".
- A file picker / drag-drop area.
- A submit button.
- An empty result panel.

Drop an audio file in, pick a model (or leave the default), click submit:

- The submit control disables.
- A busy indicator appears in the result panel.
- On completion, the busy indicator is replaced by the transcribed text,
  with a small caption indicating which model produced it.

There is no cancel button by design (clarified 2026-04-29).

---

## 6. Verify observability

```bash
# Prometheus metrics
curl -s http://localhost:8000/metrics | grep asr_
# → asr_requests_total{model="parakeet-tdt-0.6b-v3",status="ok"} 1
#   asr_request_duration_seconds_bucket{model="parakeet-tdt-0.6b-v3",stage="inference",le="..."} ...
#   asr_queue_depth{model="parakeet-tdt-0.6b-v3"} 0
#   asr_gpu_memory_used_bytes ...
#   ...
```

Logs are JSON on stdout — pipe to `jq` if eyeballing locally:

```bash
uv run python -m asr 2>&1 | jq -c 'select(.event != null)'
```

---

## 7. Run the tests

```bash
# Default: unit + contract + integration with stub model (CPU, no GPU needed)
uv run pytest

# Add real-model layer (requires GPU + downloaded weights)
uv run pytest -m "gpu_required and model_required"

# E2E containerized run (requires Docker)
uv run pytest tests/e2e

# Performance regression suite (requires GPU; run nightly in CI)
uv run pytest -m slo
```

The default run is offline-safe and finishes in under a minute. The
`gpu_required` / `model_required` markers gate the heavy layers per
constitution §IV.

---

## 8. What to do when things break

| Symptom                                                  | First place to look                                                                 |
|----------------------------------------------------------|-------------------------------------------------------------------------------------|
| Process exits at startup with "CUDA unavailable"         | You're on a non-GPU host. Set `ASR_ALLOW_CPU=1` or run on the GPU runner.            |
| `MODEL_BUSY` errors under low load                       | `ASR_QUEUE_DEPTH` is too small for your traffic, or a stuck inference is hogging the model. Check `asr_queue_depth` metric and the most recent `inference_started`/`inference_completed` log lines for the model's correlation IDs. |
| API returns valid JSON but UI shows nothing              | Open the browser console; the UI calls the same API and surfaces the structured error payload. |
| Contract test fails with "openapi.json drifted"          | You changed the API shape. Run `uv run python -m asr.scripts.dump_openapi` to regenerate `specs/001-multi-model-asr/contracts/openapi.json`, review the diff, and commit it as part of your PR. |
| Performance test fails ">10% regression vs baseline"     | Either you have a real regression — fix it — or you intentionally changed an SLO and need to amend the constitution + update `tests/perf/baseline.json`. |

---

## 9. Mapping to the spec

| Spec acceptance scenario                                   | Quickstart step that exercises it                |
|------------------------------------------------------------|---------------------------------------------------|
| US1.1 — upload & see text within 30 s                      | §5 (UI) or §4 first POST                          |
| US1.2 — non-audio upload returns clear error               | §4 oversized example with a renamed JPEG          |
| US1.3 — empty audio returns "no speech detected"           | §4 with `audio/silence.wav` (no_speech_detected=true) |
| US2.1 — model selector lists available models              | §5 dropdown                                       |
| US2.2 — response identifies which model was used           | §4 second POST (`model=seamless-m4t-v2`)          |
| US2.3 — unknown model returns clear error                  | §4 POST with `model=does-not-exist`               |
| US3.1 — API without model param uses default               | §4 first POST                                     |
| US3.2 — API with model param respects it                   | §4 second POST                                    |
| US3.3 — model list endpoint enumerates models with default | §4 `GET /api/v1/models`                           |
