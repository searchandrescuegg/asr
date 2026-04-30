from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from asr.api.errors import ErrorCode, ErrorEnvelope, TranscriptionError
from asr.api.v1.schemas import (
    HealthzResponse,
    ModelDescriptor,
    ModelListResponse,
    TranscriptionResult,
)
from asr.config import Settings, get_settings
from asr.models.registry import ModelRegistry
from asr.pipeline.transcribe import Pipeline

router = APIRouter(prefix="/api/v1")

_error_responses = {
    400: {
        "model": ErrorEnvelope,
        "description": "Invalid input (bad format or audio too long).",
    },
    404: {
        "model": ErrorEnvelope,
        "description": "Requested model is not registered.",
    },
    413: {
        "model": ErrorEnvelope,
        "description": "Upload exceeds the configured maximum file size.",
    },
    500: {
        "model": ErrorEnvelope,
        "description": "Unexpected transcription failure.",
    },
    503: {
        "model": ErrorEnvelope,
        "description": "Model is unavailable, busy, or no default model is available.",
    },
}


def _pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


def _registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


@router.post(
    "/transcribe",
    response_model=TranscriptionResult,
    responses=_error_responses,
    summary="Transcribe a single audio file",
)
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Form(default=None),
    settings: Settings = Depends(get_settings),
) -> TranscriptionResult:
    data = await file.read()
    if len(data) > settings.max_file_bytes:
        raise TranscriptionError(
            ErrorCode.FILE_TOO_LARGE,
            f"upload exceeds {settings.max_file_bytes} bytes",
            details={
                "limit_bytes": settings.max_file_bytes,
                "received_bytes": len(data),
            },
        )

    pipeline = _pipeline(request)
    return await pipeline.transcribe(data, model)


@router.get("/models", response_model=ModelListResponse)
async def list_models(request: Request) -> ModelListResponse:
    registry = _registry(request)
    descriptors = [
        ModelDescriptor(
            identifier=m.identifier,
            name=m.name,
            vendor=m.vendor,
            languages=list(m.languages),
            state=m.state,
            last_error=m.last_error,
        )
        for m in registry.list_all()
    ]
    return ModelListResponse(default=registry.default_identifier, models=descriptors)


@router.get("/healthz", response_model=HealthzResponse)
async def healthz() -> HealthzResponse:
    return HealthzResponse(status="ok")
