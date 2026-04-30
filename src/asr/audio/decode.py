from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import librosa
import numpy as np
import soundfile as sf

from asr.api.errors import ErrorCode, TranscriptionError

TARGET_SR_HZ = 16_000


@dataclass(frozen=True)
class DecodedAudio:
    samples: np.ndarray
    original_channels: int
    original_sr_hz: int
    duration_seconds: float
    downmix_applied: bool
    resample_applied: bool


def decode(data: bytes, *, max_seconds: float) -> DecodedAudio:
    if not data:
        raise TranscriptionError(
            ErrorCode.INVALID_FORMAT, "uploaded file is empty"
        )

    try:
        with sf.SoundFile(BytesIO(data)) as snd:
            original_sr = snd.samplerate
            original_channels = snd.channels
            samples = snd.read(dtype="float32", always_2d=True)
    except Exception as ex:
        try:
            samples_la, original_sr = librosa.load(
                BytesIO(data), sr=None, mono=False
            )
            if samples_la.ndim == 1:
                samples = samples_la.reshape(-1, 1)
                original_channels = 1
            else:
                samples = samples_la.T.astype(np.float32, copy=False)
                original_channels = samples.shape[1]
        except Exception:
            raise TranscriptionError(
                ErrorCode.INVALID_FORMAT,
                f"could not decode audio: {ex}",
            ) from ex

    downmix_applied = original_channels > 1
    if downmix_applied:
        mono = samples.mean(axis=1).astype(np.float32, copy=False)
    else:
        mono = samples[:, 0].astype(np.float32, copy=False)

    resample_applied = original_sr != TARGET_SR_HZ
    if resample_applied:
        mono = librosa.resample(mono, orig_sr=original_sr, target_sr=TARGET_SR_HZ)
        mono = mono.astype(np.float32, copy=False)

    duration_seconds = float(mono.shape[0]) / TARGET_SR_HZ
    if duration_seconds > max_seconds:
        raise TranscriptionError(
            ErrorCode.AUDIO_TOO_LONG,
            f"audio duration {duration_seconds:.1f}s exceeds limit {max_seconds:.1f}s",
            details={"duration_seconds": duration_seconds, "limit_seconds": max_seconds},
        )

    return DecodedAudio(
        samples=mono,
        original_channels=original_channels,
        original_sr_hz=original_sr,
        duration_seconds=duration_seconds,
        downmix_applied=downmix_applied,
        resample_applied=resample_applied,
    )
