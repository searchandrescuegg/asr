"""T045 — ErrorCode → HTTP status mapping + envelope sanity."""

from asr.api.errors import (
    ErrorCode,
    ErrorEnvelope,
    TranscriptionError,
    http_status_for,
)

EXPECTED_STATUS = {
    ErrorCode.INVALID_FORMAT: 400,
    ErrorCode.AUDIO_TOO_LONG: 400,
    ErrorCode.FILE_TOO_LARGE: 413,
    ErrorCode.MODEL_NOT_FOUND: 404,
    ErrorCode.MODEL_UNAVAILABLE: 503,
    ErrorCode.MODEL_BUSY: 503,
    ErrorCode.NO_DEFAULT_MODEL: 503,
    ErrorCode.TRANSCRIPTION_FAILED: 500,
}


def test_every_error_code_has_a_documented_status():
    for code in ErrorCode:
        assert code in EXPECTED_STATUS
        assert http_status_for(code) == EXPECTED_STATUS[code]


def test_envelope_serializes_required_fields():
    envelope = ErrorEnvelope(
        code=ErrorCode.MODEL_BUSY,
        message="busy",
        correlation_id="11111111-1111-1111-1111-111111111111",
    )
    payload = envelope.model_dump(mode="json")
    for required in ("code", "message", "correlation_id"):
        assert required in payload
    assert payload["details"] is None


def test_transcription_error_carries_code_and_details():
    err = TranscriptionError(
        ErrorCode.FILE_TOO_LARGE, "too big", details={"limit_bytes": 100}
    )
    assert err.code == ErrorCode.FILE_TOO_LARGE
    assert err.details == {"limit_bytes": 100}


def test_envelope_details_never_contains_forbidden_keys():
    """FR-013: logs/errors must not leak text/filename/audio bytes."""
    forbidden = {"text", "transcription", "filename", "audio_bytes"}
    envelope = ErrorEnvelope(
        code=ErrorCode.INVALID_FORMAT,
        message="bad",
        correlation_id="00000000-0000-0000-0000-000000000000",
        details={"limit_bytes": 100},
    )
    keys = set((envelope.details or {}).keys())
    assert keys.isdisjoint(forbidden), f"details leaked forbidden keys: {keys & forbidden}"
