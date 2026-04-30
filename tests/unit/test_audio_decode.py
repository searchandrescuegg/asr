import pytest

from asr.api.errors import ErrorCode, TranscriptionError
from asr.audio.decode import TARGET_SR_HZ, decode
from tests.conftest import make_silence_wav_bytes, make_tone_wav_bytes


def test_decode_mono_16k_passthrough():
    data = make_tone_wav_bytes(seconds=0.5, sr=TARGET_SR_HZ, channels=1)
    decoded = decode(data, max_seconds=10)
    assert decoded.original_channels == 1
    assert decoded.original_sr_hz == TARGET_SR_HZ
    assert not decoded.downmix_applied
    assert not decoded.resample_applied
    assert decoded.samples.ndim == 1
    assert decoded.samples.dtype.name == "float32"
    assert abs(decoded.duration_seconds - 0.5) < 0.01


def test_decode_stereo_downmixes():
    data = make_tone_wav_bytes(seconds=0.5, sr=TARGET_SR_HZ, channels=2)
    decoded = decode(data, max_seconds=10)
    assert decoded.original_channels == 2
    assert decoded.downmix_applied is True
    assert decoded.samples.ndim == 1


def test_decode_resamples_to_16k():
    data = make_tone_wav_bytes(seconds=0.5, sr=44_100, channels=1)
    decoded = decode(data, max_seconds=10)
    assert decoded.original_sr_hz == 44_100
    assert decoded.resample_applied is True
    # After resampling to 16 kHz, sample count ≈ 8000 for 0.5 s
    assert abs(decoded.samples.shape[0] - int(0.5 * TARGET_SR_HZ)) <= 4


def test_decode_silence_returns_zeros():
    data = make_silence_wav_bytes(seconds=0.5)
    decoded = decode(data, max_seconds=10)
    assert decoded.samples.shape[0] == int(0.5 * TARGET_SR_HZ)
    assert float(decoded.samples.max()) == 0.0


def test_decode_raises_invalid_format_for_garbage():
    with pytest.raises(TranscriptionError) as exc:
        decode(b"not audio at all", max_seconds=10)
    assert exc.value.code == ErrorCode.INVALID_FORMAT


def test_decode_raises_invalid_format_for_empty():
    with pytest.raises(TranscriptionError) as exc:
        decode(b"", max_seconds=10)
    assert exc.value.code == ErrorCode.INVALID_FORMAT


def test_decode_raises_audio_too_long():
    data = make_tone_wav_bytes(seconds=2.0, sr=TARGET_SR_HZ, channels=1)
    with pytest.raises(TranscriptionError) as exc:
        decode(data, max_seconds=1.0)
    assert exc.value.code == ErrorCode.AUDIO_TOO_LONG
    assert exc.value.details is not None
    assert exc.value.details["limit_seconds"] == 1.0
