from __future__ import annotations

from pydantic import BaseModel, Field

from asr.models.base import ModelState


class TranscriptionResult(BaseModel):
    text: str
    model: str
    audio_duration_s: float = Field(ge=0)
    inference_duration_s: float = Field(ge=0)
    downmix_applied: bool
    no_speech_detected: bool
    language: str | None = None
    correlation_id: str


class ModelDescriptor(BaseModel):
    identifier: str
    name: str
    vendor: str
    languages: list[str]
    state: ModelState
    last_error: str | None = None


class ModelListResponse(BaseModel):
    default: str
    models: list[ModelDescriptor]


class HealthzResponse(BaseModel):
    status: str
