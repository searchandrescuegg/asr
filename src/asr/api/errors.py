from __future__ import annotations

from enum import StrEnum
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    INVALID_FORMAT = "INVALID_FORMAT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    AUDIO_TOO_LONG = "AUDIO_TOO_LONG"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_BUSY = "MODEL_BUSY"
    NO_DEFAULT_MODEL = "NO_DEFAULT_MODEL"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"


_HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_FORMAT: 400,
    ErrorCode.FILE_TOO_LARGE: 413,
    ErrorCode.AUDIO_TOO_LONG: 400,
    ErrorCode.MODEL_NOT_FOUND: 404,
    ErrorCode.MODEL_UNAVAILABLE: 503,
    ErrorCode.MODEL_BUSY: 503,
    ErrorCode.NO_DEFAULT_MODEL: 503,
    ErrorCode.TRANSCRIPTION_FAILED: 500,
}


class ErrorEnvelope(BaseModel):
    code: ErrorCode
    message: str
    details: dict[str, Any] | None = None
    correlation_id: str = Field(description="UUIDv4 mirroring X-Correlation-Id")


class TranscriptionError(Exception):
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def http_status_for(code: ErrorCode) -> int:
    return _HTTP_STATUS[code]


async def transcription_error_handler(
    request: Request, exc: TranscriptionError
) -> JSONResponse:
    from asr.observability.logging import current_correlation_id

    correlation_id = current_correlation_id() or ""
    envelope = ErrorEnvelope(
        code=exc.code,
        message=exc.message,
        details=exc.details,
        correlation_id=correlation_id,
    )
    return JSONResponse(
        status_code=http_status_for(exc.code),
        content=envelope.model_dump(mode="json"),
        headers={"X-Correlation-Id": correlation_id} if correlation_id else None,
    )
