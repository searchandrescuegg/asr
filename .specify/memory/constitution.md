<!--
SYNC IMPACT REPORT
==================
Version change: (uninitialized template) → 1.0.0
Bump rationale: Initial ratification — every placeholder filled for the first time.

Modified principles:
  - [PRINCIPLE_1_NAME] → I. Multi-Model by Design
  - [PRINCIPLE_2_NAME] → II. API and UI as First-Class Surfaces
  - [PRINCIPLE_3_NAME] → III. Pragmatic Test Discipline
  - [PRINCIPLE_4_NAME] → IV. Layered Integration Testing
  - [PRINCIPLE_5_NAME] → V. Full-Stack Observability
  - (added)            → VI. Performance with Enforced SLOs

Added sections:
  - Technology & Runtime Constraints (was [SECTION_2_NAME])
  - Development Workflow & Quality Gates (was [SECTION_3_NAME])
  - Governance (filled)

Removed sections: none.

Templates requiring updates:
  - .specify/templates/plan-template.md ✅ updated (Constitution Check gates filled)
  - .specify/templates/spec-template.md ✅ reviewed — no change needed (technology-agnostic)
  - .specify/templates/tasks-template.md ✅ reviewed — categorization already supports
    contract / integration / unit / observability / performance task types
  - .specify/templates/checklist-template.md ✅ reviewed — no change needed
  - .specify/templates/agent-file-template.md ✅ reviewed — no change needed
  - README.md ⚠ pending — currently a one-line description; consider adding a
    "Project Principles" pointer to this constitution

Follow-up TODOs: none.
-->

# ASR (Automatic Speech Recognition Server) Constitution

## Core Principles

### I. Multi-Model by Design

The system MUST treat the ASR engine as a pluggable component from day one. A
single `ASRModel` (or equivalent) interface MUST encapsulate model loading,
warm-up, and inference. Concrete implementations (e.g., NVIDIA NeMo
Parakeet-TDT, future Whisper, future Canary) MUST sit behind this interface and
MUST be selectable via configuration without modifying transport, UI, or
orchestration code. No business logic in `main.py`, FastAPI route handlers, or
the Gradio interface MAY directly import a model-specific package.

**Rationale**: The project is a "multi-model python server"; without a clean
seam, each new model becomes a fork rather than an addition.

### II. API and UI as First-Class Surfaces

Both the HTTP API and the interactive UI are required, contracted deliverables.

- **HTTP API**: All public endpoints MUST be URL-versioned (`/api/v1/...`).
  Breaking changes to a versioned endpoint MUST introduce a new version
  (`/api/v2/...`); the prior version MUST remain available for at least one
  release cycle. The OpenAPI schema is the source of truth for API shape and
  MUST be checked into the repository (e.g., `contracts/openapi.json`); CI
  MUST fail if the live FastAPI schema diverges from the checked-in copy.
- **UI**: The UI (Gradio or successor) MUST exercise the same code paths users
  rely on for transcription. UI changes MUST go through the same review,
  testing, and observability bar as API changes — UI is not a "demo" tier.

**Rationale**: Both surfaces are user-facing; treating either as second-class
leads to drift between what the API promises and what the UI demonstrates.

### III. Pragmatic Test Discipline

Every critical path MUST have automated tests before merge: API contracts,
transcription correctness on a known fixture, error handling (invalid audio,
oversized uploads, missing GPU), and the model-loading boundary. Internal
helpers SHOULD be unit-tested when logic is non-trivial. Test ordering
(test-first vs. test-alongside) is at author discretion, but a PR that adds or
modifies a critical path without a corresponding test MUST be rejected. CI
MUST run the full test suite on every PR; tests MUST pass before merge.

**Rationale**: TDD is not enforced because much of the work is integration
against external models; test-required (not test-first) keeps the bar high
without ritualizing the order.

### IV. Layered Integration Testing

Integration coverage MUST span four layers, each runnable independently in CI:

1. **API contract tests** — request/response schemas for every `/api/vN/*`
   endpoint, validated against the checked-in OpenAPI schema and golden
   fixtures.
2. **Real-model smoke tests** — at least one test per registered model that
   transcribes a known short audio sample and asserts on the resulting text
   (with tolerance for minor whitespace/punctuation drift).
3. **Containerized end-to-end** — testcontainers-style: build the server
   image, run it in a container, hit `/api/v1/transcribe` from a test client,
   verify the full upload → transcribe → response flow.
4. **GPU/CUDA presence tests** — verify CUDA detection succeeds when a GPU is
   present and that the server fails fast (or skips, per configured policy)
   with a clear error when it is not.

CPU-only environments MAY skip layers 2 and 4 via an explicit pytest marker;
GPU-equipped CI runners MUST run all four layers.

**Rationale**: Each layer catches a class of bug the others cannot — schema
drift, model-output regressions, packaging/runtime errors, and hardware-config
errors are independent failure modes.

### V. Full-Stack Observability

Every production deployment MUST emit:

- **Structured logs** — JSON-formatted, per-request correlation/trace IDs,
  fields for `model_name`, `audio_duration_s`, `inference_duration_s`, and
  `status`.
- **Metrics** — Prometheus-compatible `/metrics` endpoint exposing request
  counts, latency histograms (overall and per model), GPU utilization and
  memory, and error counters by class. Metric names MUST be stable; renames
  count as a MAJOR change.
- **Traces** — OpenTelemetry spans for the transcription pipeline:
  `request_received → audio_decoded → model_inference → response_serialized`.
  Spans MUST carry the same correlation ID present in logs.

Observability is not optional for "internal" tools; the API and UI both ship
with this stack enabled by default.

**Rationale**: ASR latency and accuracy regressions are silent without
telemetry — by the time a user reports them, the regression has been live for
days.

### VI. Performance with Enforced SLOs

The constitution sets numeric SLOs that regression tests MUST enforce. SLOs
apply to a defined reference environment (single NVIDIA GPU with ≥ 8 GB VRAM,
CUDA 12.x, the model named in `pyproject.toml`):

- **Real-time factor (RTF)** — p95 RTF ≤ 0.5 over the standard fixture set
  (RTF = `inference_duration / audio_duration`).
- **API overhead** — p95 non-inference request handling (decode + serialize +
  network within the test client) ≤ 100 ms.
- **Cold start** — model load on process start ≤ 60 s.
- **Steady-state memory** — ≤ 8 GB GPU VRAM at idle after warm-up.

A nightly performance job MUST publish current values and fail when any SLO
regresses by more than 10% relative to the prior tagged release. Tightening an
SLO is a MINOR amendment; loosening one is MAJOR.

**Rationale**: ASR users notice latency immediately; without committed numbers,
"a little slower" accumulates release over release.

## Technology & Runtime Constraints

- **Language/runtime**: Python 3.11 only (`requires-python = "==3.11.*"`).
  Adding support for another minor version requires an amendment.
- **Dependency management**: `uv` is the single source of truth. `pyproject.toml`
  and `uv.lock` MUST stay in sync; PRs that modify deps MUST update the lock.
- **GPU policy**: Production deployments require CUDA-capable hardware. The
  server MUST fail fast at startup if CUDA is unavailable in production mode;
  development/test mode MAY allow CPU fallback behind an explicit
  `ASR_ALLOW_CPU=1` env flag, which MUST be logged at WARN level on startup.
- **Audio input contract**: All transcription endpoints accept a single audio
  file upload, normalized to 16 kHz mono before inference. Multi-channel
  inputs MUST be rejected with HTTP 400 or downmixed deterministically — the
  choice MUST be documented in the OpenAPI schema; silent behavior changes are
  forbidden.
- **Container parity**: The Docker image used in containerized integration
  tests MUST be the same image (or built from the same Dockerfile) used for
  production. "Works on my laptop, fails in container" is treated as a P0 bug.

## Development Workflow & Quality Gates

- **Spec-driven changes**: Non-trivial features follow the spec → plan → tasks
  flow under `/specs/<feature>/`. The plan's Constitution Check section MUST
  pass before implementation begins.
- **Pull requests**: Every PR MUST include (a) what changed, (b) which
  principles it touches, (c) the test evidence (paths to new/updated tests).
  PRs that only refactor MUST state why and MUST not change observable
  behavior.
- **CI gates** — all required to merge:
  1. Lint (`ruff`).
  2. Unit + contract tests on CPU.
  3. Containerized end-to-end test.
  4. (GPU runners) real-model smoke + performance regression check.
- **Reviews**: At least one human review is required. Reviewers MUST verify
  constitution compliance; "looks good" without principle-by-principle check
  is insufficient for changes that touch the model interface, API surface,
  observability stack, or performance-critical paths.
- **Versioning**: The application itself uses semver
  (`pyproject.toml::project.version`). Breaking API changes require a new
  `/api/vN` AND a MAJOR application version bump.

## Governance

This constitution supersedes ad-hoc practice. When a code review, plan, or
spec conflicts with another document, the constitution wins; the conflicting
document MUST be updated.

**Amendment procedure**:

1. Open a PR that modifies `.specify/memory/constitution.md`.
2. The PR description MUST state the proposed version bump (MAJOR / MINOR /
   PATCH) and the rationale.
3. The PR MUST include a Sync Impact Report block (see top of this file) and
   propagate any necessary changes to dependent templates under
   `.specify/templates/` and to user-facing docs.
4. At least one reviewer MUST approve. For MAJOR amendments, the reviewer
   MUST be different from the author.

**Versioning policy**:

- **MAJOR** — backward-incompatible change: principle removed or redefined
  to forbid previously allowed behavior; SLO loosened; required gate dropped.
- **MINOR** — new principle, new section, materially expanded guidance, or
  SLO tightened.
- **PATCH** — clarifications, wording, typo fixes, non-semantic refinements.

**Compliance review**: Every PR review is a compliance review. In addition,
a quarterly walkthrough of this document SHOULD be scheduled to catch drift
between intent and practice; findings result in either a code change, a
constitution amendment, or both.

**Version**: 1.0.0 | **Ratified**: 2026-04-29 | **Last Amended**: 2026-04-29
