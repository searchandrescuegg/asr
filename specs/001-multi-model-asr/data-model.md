# Phase 1 Data Model: Multi-Model Speech Transcription

**Feature**: `001-multi-model-asr`
**Date**: 2026-04-29

This document defines the in-memory entities, their fields and validation
rules, the lifecycle of a transcription request, and the boundary objects
exchanged with API/UI clients. Storage is transient — there is no
database. "Persistence" here means "exists for the duration of one request".

---

## Entities

### Model

A registered ASR engine.

| Field            | Type                | Constraints / Notes                                                                                |
|------------------|---------------------|----------------------------------------------------------------------------------------------------|
| `identifier`     | `str`               | Stable, kebab-case, unique within the registry. Examples: `parakeet-tdt-0.6b-v2`, `seamless-m4t-v2`. Used as the `model` parameter on the API. |
| `name`           | `str`               | Human-readable. Examples: "NVIDIA Parakeet-TDT 0.6B v2", "Facebook Seamless M4T v2 Large".         |
| `vendor`         | `str`               | Free-form provenance label (e.g., `nvidia`, `meta`). Cosmetic only.                                |
| `languages`      | `list[str]`         | ISO-639-1 codes the model is trained on (e.g., `["en"]` for Parakeet, `["en","es","fr",...]` for Seamless). Empty list means "unknown". |
| `expected_sr_hz` | `int`               | Sample rate the model expects after preprocessing (always `16000` in v1).                          |
| `state`          | `ModelState` (enum) | `LOADING` | `READY` | `FAILED`. Mutable; transitions in §Lifecycle.                                |
| `last_error`     | `str \| None`       | Populated when `state == FAILED`. Surfaced in the model-list response.                             |

**ModelState transitions**:
```
                ┌────────┐  load() ok    ┌──────┐
                │LOADING │──────────────▶│READY │
                └────────┘               └──────┘
                     │                       │
                     │ load() raises         │ inference raises terminal
                     ▼                       ▼
                ┌──────────────────────────────┐
                │            FAILED            │
                └──────────────────────────────┘
```

A `FAILED` model is removed from the public model list (FR; "Model
unavailable mid-session" edge case). A model never returns from
`FAILED` to `READY` within a single process — restart is required.

---

### AudioSubmission

A single transcription request, internal-only. Never serialized to clients.

| Field             | Type                  | Constraints / Notes                                                          |
|-------------------|-----------------------|------------------------------------------------------------------------------|
| `correlation_id`  | `str` (UUIDv4)        | Generated at the request middleware; carried in logs, traces, response header. |
| `audio_bytes`     | `bytes`               | Raw upload. **Never logged. Never persisted.** Released after decoding.       |
| `audio_sha256`    | `str` (hex)           | SHA-256 of `audio_bytes`. Logged. Allows dedup-style debugging without revealing content. |
| `requested_model` | `str \| None`         | None means "use default". Validated against the registry at intake.           |
| `received_at`     | `datetime` (UTC)      | Server clock at intake.                                                       |

**Validation at intake (boundary)**:
- `len(audio_bytes) ≤ ASR_MAX_FILE_BYTES` else `ErrorCode.FILE_TOO_LARGE`.
- `Content-Type` parsed from upload; non-audio rejected as `ErrorCode.INVALID_FORMAT`.
- After decode (see DecodedAudio), audio duration checked against
  `ASR_MAX_AUDIO_SECONDS`; over → `ErrorCode.AUDIO_TOO_LONG`.

---

### DecodedAudio

Internal-only. The output of the decode/normalize pipeline stage.

| Field              | Type                | Constraints / Notes                                            |
|--------------------|---------------------|----------------------------------------------------------------|
| `samples`          | `np.ndarray[float32]` | Mono, 16 kHz. Shape `(N,)`.                                  |
| `original_channels`| `int`               | Channel count of the source file (≥ 1).                        |
| `original_sr_hz`   | `int`               | Sample rate of the source file before resampling.              |
| `duration_seconds` | `float`             | `N / 16000`. Computed once and cached.                         |
| `downmix_applied`  | `bool`              | `True` iff `original_channels > 1`.                            |
| `resample_applied` | `bool`              | `True` iff `original_sr_hz != 16000`. Internal only — not surfaced to API; the response only flags downmix per spec/constitution. |

---

### TranscriptionResult

Public — serialized to the API and rendered in the UI. Returned on success.

| Field                 | Type            | Constraints / Notes                                                |
|-----------------------|-----------------|--------------------------------------------------------------------|
| `text`                | `str`           | Plain string. No timing markup, no segment list, no word list. May be the empty string (preserved on `no_speech_detected: false`). |
| `model`               | `str`           | The `Model.identifier` that produced this result.                  |
| `audio_duration_s`    | `float`         | Same as `DecodedAudio.duration_seconds`.                            |
| `inference_duration_s`| `float`         | Wall time of the model's `transcribe(...)` call.                    |
| `downmix_applied`     | `bool`          | Mirrors `DecodedAudio.downmix_applied`.                             |
| `no_speech_detected`  | `bool`          | `True` when the model returned no recognized speech (empty/whitespace text). UI shows the explicit message in this case (FR; "No speech in audio" edge case). |
| `language`            | `str \| None`   | ISO-639-1 if the model returns a detected language; `null` otherwise. |
| `correlation_id`      | `str` (UUIDv4)  | Same value as the response header `X-Correlation-Id`.               |

**Validation**: All fields except `language` are required. `language` is
`null` for Parakeet (English-only model that doesn't return a language
tag) and is populated for Seamless when reported by the model.

---

### ErrorEnvelope

Public — serialized for any non-2xx API response. Identical shape across all error sources.

| Field            | Type                       | Constraints / Notes                                                              |
|------------------|----------------------------|----------------------------------------------------------------------------------|
| `code`           | `ErrorCode` (str enum)     | Machine-readable. See enum below.                                                |
| `message`        | `str`                      | Human-readable. Suitable for end-user display in the UI.                         |
| `details`        | `dict \| None`             | Optional structured context (e.g., `{"limit_bytes": 104857600}`). Never contains audio bytes, transcription text, or filenames. |
| `correlation_id` | `str` (UUIDv4)             | Same value as the response header.                                               |

**ErrorCode enum** (closed set; expanding the set is a MINOR API change):

| Code                       | HTTP | Meaning                                                                  |
|----------------------------|------|--------------------------------------------------------------------------|
| `INVALID_FORMAT`           | 400  | Upload could not be decoded as audio (FR-008; "Unsupported format" edge). |
| `FILE_TOO_LARGE`           | 413  | Upload exceeds `ASR_MAX_FILE_BYTES` (FR-011; "Oversized file" edge).      |
| `AUDIO_TOO_LONG`           | 400  | Decoded duration exceeds `ASR_MAX_AUDIO_SECONDS` ("Long audio" edge).     |
| `MODEL_NOT_FOUND`          | 404  | The requested `model` is not registered (FR-004 explicit-model branch).   |
| `MODEL_UNAVAILABLE`        | 503  | Registered but in `FAILED` state ("Model unavailable mid-session" edge).  |
| `MODEL_BUSY`               | 503  | Per-model queue full (FR-017; "Per-model queue full" edge).               |
| `NO_DEFAULT_MODEL`         | 503  | Default (Parakeet) is unavailable and the user did not specify a model (FR-004 default branch). |
| `NO_SPEECH_DETECTED`       | 200  | NOT an error envelope. Carried as `no_speech_detected: true` on the success response. Listed here only to document it is *not* in this enum. |
| `TRANSCRIPTION_FAILED`     | 500  | Catch-all for unexpected adapter exceptions (FR-012). The original exception is logged with full detail; the envelope carries only the code and a generic message. |

`NO_SPEECH_DETECTED` is intentionally not an error code — empty
transcription is a valid result with explicit signaling via the
`no_speech_detected` flag, per the spec.

---

## Lifecycle: a single transcription request

```
HTTP POST /api/v1/transcribe
        │
        ▼
[middleware] generate correlation_id; bind to contextvar; start root span
        │
        ▼
[boundary] parse multipart upload → AudioSubmission (validate size, content-type)
        │  fail → ErrorEnvelope{INVALID_FORMAT|FILE_TOO_LARGE}
        ▼
[hash]    compute audio_sha256
        │
        ▼
[decode]  bytes → DecodedAudio (validate duration ≤ cap; downmix; resample)
        │  fail → ErrorEnvelope{INVALID_FORMAT|AUDIO_TOO_LONG}
        ▼
[resolve] requested_model → Model (registry lookup; default if None)
        │  fail → ErrorEnvelope{MODEL_NOT_FOUND|MODEL_UNAVAILABLE|NO_DEFAULT_MODEL}
        ▼
[enqueue] try put on per-model bounded queue
        │  full → ErrorEnvelope{MODEL_BUSY}
        ▼
[await]   acquire per-model semaphore (size 1)
        │
        ▼
[run]     model.transcribe(samples) under asyncio.to_thread
        │  raise → ErrorEnvelope{TRANSCRIPTION_FAILED} (logs full stack)
        ▼
[build]   TranscriptionResult (text, model id, durations, flags, language)
        │
        ▼
[respond] release semaphore; emit metrics; close span; return JSON
```

Throughout: every stage emits an OpenTelemetry span with the same
`correlation_id` and writes a structured log line on entry/exit.
The response carries `X-Correlation-Id` so clients can quote it in bug
reports.

---

## Concurrency invariants

- For any `Model`, at most one `model.transcribe(...)` call is in flight at
  any moment in time. Enforced by `asyncio.Semaphore(1)` per model.
- For any `Model`, at most `ASR_QUEUE_DEPTH` requests are *waiting*. The
  next request to arrive while the queue is full is rejected immediately
  with `MODEL_BUSY`.
- Different models run independently; a Parakeet inference never blocks a
  Seamless inference (subject to GPU memory, which is a separate concern
  surfaced by `MODEL_UNAVAILABLE` at load time).
- Cancellation: not supported in v1. A connection drop on the client side
  does NOT abort the in-flight inference; the result is computed and
  discarded on response. (Documented in spec FR-006.)

---

## Validation rules summary (boundary-only)

| Rule                                     | Where enforced                       | On failure                          |
|------------------------------------------|--------------------------------------|-------------------------------------|
| Upload size ≤ `ASR_MAX_FILE_BYTES`       | API intake (multipart parser)        | `FILE_TOO_LARGE`                    |
| Content-Type is audio/*                  | API intake                           | `INVALID_FORMAT`                    |
| Audio decodable                          | `audio.decode.decode(...)`           | `INVALID_FORMAT`                    |
| Decoded duration ≤ `ASR_MAX_AUDIO_SECONDS` | After decode                       | `AUDIO_TOO_LONG`                    |
| `model` (if provided) is registered      | Registry lookup                      | `MODEL_NOT_FOUND`                   |
| Resolved model is `READY`                | Registry lookup                      | `MODEL_UNAVAILABLE`                 |
| Default model is `READY` (when no model) | Registry lookup                      | `NO_DEFAULT_MODEL`                  |
| Per-model queue not full                 | Pipeline enqueue                     | `MODEL_BUSY`                        |

No re-validation occurs inside trusted internals (research §R11).

---

## Out-of-scope (v1)

- Per-segment / word-level timestamps (clarified 2026-04-29).
- Alternative hypotheses (n-best output).
- Persistent storage of audio or transcriptions.
- Cancellation of in-flight inference.
- Authentication / authorization / multi-tenant isolation.
- Asynchronous job-based transcription for long audio.
- Streaming (chunked) transcription.
