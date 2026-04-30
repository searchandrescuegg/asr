# Feature Specification: Multi-Model Speech Transcription Application

**Feature Branch**: `001-multi-model-asr`
**Created**: 2026-04-29
**Status**: Draft
**Input**: User description: "Please provide me an application that can transcribe speech/audio with a number of models, primarily targeting NVIDIA Parakeet and Facebook Seamless, written in python with uv tooling, that has a UI and API for interaction"

## Clarifications

### Session 2026-04-29

- Q: What does the transcription response contain — plain text, or also timing/segment data? → A: Plain text only (no per-segment or word-level timestamps in v1).
- Q: How are concurrent transcription requests handled? → A: Serialize per model (one in-flight per model); different models may overlap; bounded queue with overflow rejection.
- Q: Which model is the default when the user doesn't specify one? → A: NVIDIA Parakeet.
- Q: What does the UI show while a transcription is running? → A: Busy indicator + disabled submit control; no cancel; result replaces the indicator on completion.
- Q: What audio/transcription content is written to logs? → A: Metadata + audio content-hash (SHA-256); no filename, no transcription text, no audio bytes.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Quickly transcribe an audio file (Priority: P1)

A user has an audio recording (a meeting clip, an interview snippet, a voice memo) and
wants the spoken content as text. They open the application, drop the file in, and
within seconds they see the transcription rendered on screen and can copy it.

**Why this priority**: This is the core value of the product. Without this flow
working end-to-end against at least one model, no other story matters. It is the MVP.

**Independent Test**: From a fresh deployment with the default model loaded, upload a
known short audio clip ("the quick brown fox" reference clip) through the UI. The
transcription text appears within the success-criteria latency bound and matches the
expected text within accepted tolerance.

**Acceptance Scenarios**:

1. **Given** the application is running and the default model is loaded, **When** the
   user uploads a 30-second mono speech audio file, **Then** they see the
   transcribed text on screen within 30 seconds and can copy it to the clipboard.
2. **Given** the application is running, **When** the user uploads a file that is
   not an audio file (e.g., a JPEG), **Then** they see a clear error message
   identifying the problem and the upload control returns to a ready state.
3. **Given** the application is running, **When** the user uploads an empty
   audio file (silence), **Then** they see a result with `no_speech_detected`
   set to true, the UI explicitly renders "No speech detected", and the API
   response carries the same flag — not just an empty `text` string.

---

### User Story 2 - Choose which model performs the transcription (Priority: P2)

A user notices that one model handles their accent or domain better than another, or
they want to compare quality. They pick the desired model from a list before
submitting the file, and the result is tagged with which model produced it.

**Why this priority**: Multi-model is the explicit purpose of the project. Without
selection, the "multi" is invisible to the user and the system is no different from a
single-model tool.

**Independent Test**: With at least two models registered (Parakeet and Seamless),
upload the same audio twice — once with each model selected. Verify both produce a
transcription and that each result is labeled with the model that produced it.

**Acceptance Scenarios**:

1. **Given** at least two models are available, **When** the user opens the
   transcription UI, **Then** they see a model selector listing every available
   model with a clear default.
2. **Given** the user has chosen a non-default model and submitted a file, **When**
   the transcription completes, **Then** the response (UI display and API payload)
   identifies the model that was used.
3. **Given** the user requests a model that is not currently available (e.g., not
   registered, failed to load), **When** they submit, **Then** they see a clear
   error naming the model and listing which models *are* available.

---

### User Story 3 - Programmatic transcription via API (Priority: P3)

A developer integrates transcription into their own workflow — a script, an
automation, a downstream service. They send an audio file to the API, optionally
specify the model, and parse the response.

**Why this priority**: The API is a first-class surface per the project constitution.
It must be usable without the UI. This story is P3 only because users on the UI path
(P1, P2) cover the underlying functionality; the API needs its own contract guarantees
on top.

**Independent Test**: Using a generic HTTP client (no UI), submit a known audio file
to the transcription endpoint with and without a model parameter, and verify the
response shape matches a documented schema and contains the expected transcription
text.

**Acceptance Scenarios**:

1. **Given** the API is running, **When** a client POSTs an audio file to the
   transcription endpoint without specifying a model, **Then** the response contains
   the transcription text, the model used, and the audio duration.
2. **Given** the API is running, **When** a client POSTs an audio file with an
   explicit model parameter naming a registered model, **Then** the response uses
   that model and the response payload identifies it.
3. **Given** the API is running, **When** a client requests the list of available
   models, **Then** the response enumerates each model with a stable identifier and
   indicates which is the default.

---

### Edge Cases

- **Multi-channel audio**: User uploads a stereo (or multi-channel) file. System
  downmixes to mono deterministically before transcription and notes in the
  response that downmix occurred. Silent acceptance is forbidden.
- **Sample-rate mismatch**: User uploads audio at a sample rate other than the
  model's expected rate (typically 16 kHz). System resamples transparently.
- **Unsupported format**: User uploads a format the system cannot decode. Response
  is a clear error naming the rejection reason; UI returns to ready state.
- **Oversized file**: User uploads a file larger than the configured maximum.
  System rejects with a size-limit error stating the limit before any transcription
  attempt begins.
- **Long audio (≥ 10 minutes)**: System rejects the upload with a clear
  `AUDIO_TOO_LONG` error stating the configured cap. Asynchronous /
  job-based handling is out of scope for v1 and is not currently planned;
  callers needing long-form transcription must split the audio themselves.
- **No speech in audio**: System returns an explicit "no speech detected" outcome
  rather than empty text with no signal.
- **No GPU available**: Server fails fast at startup with a clear log message;
  development mode may allow CPU operation behind an explicit flag (per
  constitution).
- **Model unavailable mid-session**: User selects a model that was registered at
  startup but has since failed (e.g., OOM, crashed). Submission returns a clear
  error and the model is removed from the list of available models on the next
  query.
- **Per-model queue full**: A second (or Nth) request arrives for a model
  whose queue is already at capacity. System responds immediately with a
  structured "busy" error stating the model and that the queue is full;
  caller may retry. Requests targeting a different (idle) model are
  unaffected.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST allow a user to submit a single audio file and receive a
  transcription as text.
- **FR-002**: System MUST support at least two distinct ASR models in v1: NVIDIA
  Parakeet and Facebook Seamless.
- **FR-003**: System MUST expose a list of currently available models, each with a
  stable identifier and a human-readable name.
- **FR-004**: Users MUST be able to choose which available model performs a given
  transcription. When the user makes no choice, the system MUST default to
  NVIDIA Parakeet. If Parakeet is not registered or not available, submission
  without an explicit model MUST fail with a clear "no default model
  available" error rather than silently substituting another model.
- **FR-005**: System MUST identify, in every transcription response, which model
  produced the result.
- **FR-006**: System MUST provide an interactive UI for upload-and-transcribe with
  a model selector and visible result. While a transcription is in flight,
  the UI MUST show a busy indicator and MUST disable the submit control to
  prevent duplicate submissions; on completion, the result replaces the busy
  indicator. Cancellation of an in-flight transcription is not supported in
  v1.
- **FR-007**: System MUST provide an API for upload-and-transcribe with the same
  capabilities as the UI (model selection, result identification, model listing).
- **FR-008**: System MUST accept common audio formats (at minimum WAV, MP3, FLAC,
  OGG) and decode them to the model's expected input format internally.
- **FR-009**: System MUST downmix multi-channel audio to mono deterministically and
  surface that this transformation occurred in the response.
- **FR-010**: System MUST resample audio to the model's expected sample rate
  internally; users are not required to pre-process audio.
- **FR-011**: System MUST reject submissions that exceed the configured maximum
  file size with a clear error before any transcription attempt.
- **FR-012**: System MUST return a clear, structured error when transcription fails
  (invalid format, model crash, no speech, etc.) rather than a generic 500.
- **FR-013**: System MUST log every transcription request with a correlation
  identifier, the model used, the audio duration, the inference duration,
  and a SHA-256 content hash of the uploaded audio bytes. Logs MUST NOT
  contain the transcription text, the original filename, or the audio bytes
  themselves.
- **FR-014**: System MUST allow a new ASR model to be registered without changes
  to the UI or API surface — adding a model is a configuration/registration task,
  not a code rewrite of transport or interaction layers.
- **FR-015**: API responses describing transcription results MUST conform to a
  documented schema; the schema is the contract and is checked into the
  repository.
- **FR-016**: System MUST serialize transcription requests per model — at most
  one inference is in flight on a given model at any time. Requests for
  different models MAY overlap.
- **FR-017**: System MUST maintain a bounded per-model queue for waiting
  requests. When the queue for a model is full, new requests for that model
  MUST be rejected immediately with a clear "busy / queue full" error
  (machine-readable code) rather than blocking indefinitely.

### Key Entities *(include if feature involves data)*

- **AudioSubmission**: A single transcription request. Attributes: identifier,
  uploaded audio bytes (transient), audio duration, requested model (or default),
  submission timestamp.
- **Model**: A registered ASR engine. Attributes: stable identifier
  (e.g., `parakeet-tdt-0.6b-v2`, `seamless-m4t-v2`), human-readable name,
  availability status, supported languages.
- **TranscriptionResult**: The output of a successful transcription. Attributes:
  text (plain string, no timing markup), model identifier, audio duration,
  inference duration, downmix-applied flag, language (when the model reports
  it). v1 does NOT include per-segment timestamps, word-level timestamps, or
  alternative hypotheses.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can upload an audio file and see the transcription rendered on
  screen in under 30 seconds for clips up to one minute, in 95% of attempts on the
  reference deployment.
- **SC-002**: A user can choose between models in the UI and submit a file in
  under three clicks (file-pick + model-pick + submit).
- **SC-003**: For the standard reference clip set, transcription accuracy (word
  error rate) for each registered model is within 5 percentage points of that
  model's published accuracy on the same content.
- **SC-004**: A new ASR model can be added by registering it (configuration +
  adapter) without modifying any UI or API request/response code.
- **SC-005**: 100% of API responses for successful transcriptions identify which
  model produced the result.
- **SC-006**: 100% of error responses (invalid input, oversize, model failure)
  carry a structured, machine-readable error code in addition to a human-readable
  message.
- **SC-007**: For audio under one minute, the user-perceived end-to-end time
  (upload → result visible) is faster than the audio's own duration in 95% of
  attempts on the reference deployment.

## Assumptions

- The application is operated by a single user or small team behind external
  access controls (e.g., VPN, internal network). v1 does not include
  authentication, authorization, multi-tenant isolation, or rate limiting.
- Synchronous transcription is sufficient for v1. Long-running async/job-based
  transcription is out of scope; long files either complete within the request
  window or are rejected with guidance.
- Audio files in v1 are bounded in size (default cap on the order of 100 MB) and
  duration (default cap on the order of 10 minutes). Both limits are configurable.
- The reference deployment is a single host with one GPU sufficient to load both
  Parakeet and Seamless concurrently; if memory is insufficient, models are
  loaded on demand. The user does not see this distinction.
- The user's audio is in a language supported by their chosen model. Language
  mismatch is a model-quality concern, not a system error; the transcription is
  still returned.
- Persistent storage of audio or transcriptions is out of scope. Submissions are
  processed in memory; results are returned to the caller and not retained.
- The project's existing technology choices (Python 3.11, `uv` for dependency
  management, GPU-first runtime) are inherited from the project constitution and
  not re-litigated by this spec.
