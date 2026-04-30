# Reference Audio Fixtures

This file is the source-of-truth manifest for the audio fixtures the test
suite consumes. Any test that names `audio/<filename>` MUST appear in this
manifest with provenance, expected text, and acoustic metadata.

Adding a fixture is a deliberate act: it joins the SC-003 accuracy gate
(real-model smoke tests assert `abs(measured_wer - published_wer) ≤ 0.05`
versus the per-model published WER recorded below) and may also feed the
SLO regression baseline (`tests/perf/baseline.json`).

## Per-model published WER (English, clean speech)

These published numbers are the reference the `5pp` tolerance compares
against. Update them whenever the bundled model version changes.

| Model identifier              | Published WER | Source                                              |
|-------------------------------|---------------|-----------------------------------------------------|
| `parakeet-tdt-0.6b-v2`        | 0.064         | NVIDIA model card (Parakeet-TDT 0.6B v2, LibriSpeech test-clean) |
| `seamless-m4t-v2`             | 0.090         | Meta SeamlessM4T paper (English ASR, FLEURS test)   |

## Fixtures

### `psern01.wav`

| Field                | Value                                                                  |
|----------------------|------------------------------------------------------------------------|
| Purpose              | SC-003 reference clip used by Parakeet AND Seamless smoke tests        |
| Source / license     | Pre-existing fixture in this repo. Provenance to be confirmed before publishing. |
| Original SR          | TBD — check with `soundfile.info` and update                            |
| Channels             | TBD — confirm and update                                                |
| Duration             | TBD                                                                    |
| Expected text        | TBD — to be transcribed by ear and recorded here as the gold reference |
| Used by              | T036 (Seamless smoke), T051 (Parakeet smoke), T052 (perf baseline)     |

### `psern02.wav`

| Field                | Value                                                                  |
|----------------------|------------------------------------------------------------------------|
| Purpose              | Secondary clip; perf-fixture set extension                              |
| Source / license     | Pre-existing fixture in this repo                                       |
| Original SR          | TBD                                                                    |
| Channels             | TBD                                                                    |
| Duration             | TBD                                                                    |
| Expected text        | TBD                                                                    |
| Used by              | T052 (perf baseline)                                                   |

### `silence.wav` (NOT YET COMMITTED)

| Field                | Value                                                                  |
|----------------------|------------------------------------------------------------------------|
| Purpose              | US1.3 acceptance: empty/silent audio → `no_speech_detected: true`       |
| Source / license     | To be generated locally with `sox -n -r 16000 -c 1 audio/silence.wav trim 0.0 3.0` |
| Original SR          | 16000                                                                   |
| Channels             | 1 (mono)                                                                |
| Duration             | 3.0 s                                                                   |
| Expected text        | `""` (empty); `no_speech_detected` MUST be `true`                       |
| Used by              | T025 (no-speech integration test)                                       |

### `stereo.wav` (NOT YET COMMITTED)

| Field                | Value                                                                  |
|----------------------|------------------------------------------------------------------------|
| Purpose              | FR-009 downmix path test                                                |
| Source / license     | To be generated locally — duplicate `psern01.wav` to a 2-channel stereo file (`sox psern01.wav -c 2 audio/stereo.wav`). |
| Original SR          | matches `psern01.wav`                                                   |
| Channels             | 2                                                                       |
| Duration             | matches `psern01.wav`                                                   |
| Expected text        | matches `psern01.wav`                                                   |
| Used by              | T021 (decode unit test, downmix path)                                  |

### `non16k.wav` (NOT YET COMMITTED)

| Field                | Value                                                                  |
|----------------------|------------------------------------------------------------------------|
| Purpose              | FR-010 resample path test                                               |
| Source / license     | To be generated locally — resample `psern01.wav` to 44.1 kHz (`sox psern01.wav -r 44100 audio/non16k.wav`). |
| Original SR          | 44100                                                                   |
| Channels             | 1 (mono)                                                                |
| Duration             | matches `psern01.wav`                                                   |
| Expected text        | matches `psern01.wav`                                                   |
| Used by              | T021 (decode unit test, resample path)                                  |

## Operator notes

- Fields marked `TBD` are blocking before T036, T051, and T052 can pass.
  Filling them is a one-time task: run `uv run python -c "import soundfile; print(soundfile.info('audio/psern01.wav'))"` and listen to the clip to author the gold transcription.
- Keep the WER table aligned with the model versions actually pinned in
  `pyproject.toml`. A version bump that changes published accuracy MUST
  update this file in the same PR.
- Fixtures used only for the perf SLO suite (no accuracy gate) may omit
  the gold transcription; that fact MUST be noted in the row.
