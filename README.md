# ASR — Multi-Model Speech Transcription

A Python server that transcribes speech with multiple ASR models (NVIDIA
Parakeet and Facebook Seamless in v1), exposing both an HTTP API and a
Gradio UI.

## Project documents

- **Constitution** — non-negotiable engineering principles:
  [`.specify/memory/constitution.md`](.specify/memory/constitution.md)
- **v1 feature spec, plan, and tasks** —
  [`specs/001-multi-model-asr/`](specs/001-multi-model-asr/)
- **Quickstart** (run it locally in 10 minutes) —
  [`specs/001-multi-model-asr/quickstart.md`](specs/001-multi-model-asr/quickstart.md)
- **API contract** (source of truth) —
  [`specs/001-multi-model-asr/contracts/openapi.json`](specs/001-multi-model-asr/contracts/openapi.json)

## Running locally (TL;DR)

```bash
uv sync
ASR_ALLOW_CPU=1 uv run python -m asr   # CPU-only dev mode
# or, with a GPU:
uv run python -m asr
```

Then `http://localhost:8000/` for the UI, `/api/v1/transcribe` for the
API, `/metrics` for Prometheus, `/api/v1/healthz` for liveness.

## Tests

```bash
uv run pytest                                       # CPU-safe, default markers
uv run pytest -m "gpu_required and model_required"  # GPU runner only
uv run pytest -m slo                                # nightly perf regression
```

The default run is offline-safe and finishes in seconds. GPU and SLO
suites are gated to dedicated runners per Constitution IV.
