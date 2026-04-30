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

## Run it locally

Pick the path that matches your machine. All three end with the API at
`http://localhost:8000/api/v1/...` and the UI at `http://localhost:8000/`.

| Path | When to use | Runs Parakeet / Seamless? |
|---|---|---|
| `make docker-cpu` | Fastest "just see it work" — any OS with Docker | No (env `ASR_ENABLED_MODELS=""` skips heavy loads) |
| `make docker-gpu` | Linux host with NVIDIA GPU + nvidia-container-toolkit | Yes (full real-model inference) |
| `make install-mac && make dev-mac` | macOS / Apple Silicon dev loop | Code-only; Parakeet + Seamless go to FAILED state (no CUDA) |
| `make install-linux && make dev` | Linux dev loop with the full GPU stack | Yes |

### Docker (any OS)

```bash
make docker-cpu        # API + UI come up; no model loaded
# or
make docker-gpu        # full stack; needs nvidia-container-toolkit
```

The compose file uses profiles, so the `gpu` and `cpu` services are
mutually exclusive.

### macOS / Apple Silicon

The pinned cu128 nightly torch has no darwin wheel, so `uv sync` does
not work on macOS. Use the pip path instead:

```bash
make install-mac       # creates .venv with CPU torch + dev deps
make dev-mac           # ASR_ALLOW_CPU=1, no real models loaded
make test              # 48 unit/integration/contract tests pass on CPU
```

The Parakeet and Seamless adapters lazy-import their dependencies inside
`load()` — on macOS those imports raise (no NeMo wheel, no CUDA torch),
the models are marked `FAILED`, and the rest of the app keeps running.
You can still exercise the API, UI, queue, observability, and contract
gates.

### Linux

```bash
make install-linux     # uv sync — uses uv.lock (cu128 nightly torch)
make dev               # full GPU; ASR_ALLOW_CPU=1 if no GPU is attached
```

## Tests

```bash
make test              # CPU-safe default markers (48 tests, ~1.5s)
make test-gpu          # gpu_required + model_required + slo (needs GPU)
make lint              # ruff
```

The default `make test` is offline-safe and has no torch/CUDA
requirement. GPU and SLO suites are gated by pytest markers and run
only on dedicated runners per Constitution IV.
