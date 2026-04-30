from __future__ import annotations

import asyncio
import time

from asr.api.errors import ErrorCode, TranscriptionError
from asr.api.v1.schemas import TranscriptionResult
from asr.audio.decode import decode
from asr.audio.hash import sha256_hex
from asr.config import get_settings
from asr.models.registry import ModelRegistry
from asr.observability import metrics
from asr.observability.logging import current_correlation_id, get_logger
from asr.observability.tracing import stage_span
from asr.pipeline.queue import QueueRouter

_log = get_logger("asr.pipeline.transcribe")


class Pipeline:
    def __init__(self, registry: ModelRegistry, queue_router: QueueRouter) -> None:
        self._registry = registry
        self._queue_router = queue_router

    async def transcribe(
        self, file_bytes: bytes, requested_model: str | None
    ) -> TranscriptionResult:
        settings = get_settings()
        cid = current_correlation_id() or ""

        with stage_span("request_received"):
            audio_sha = sha256_hex(file_bytes)
            _log.info(
                "transcription_request_received",
                audio_bytes=len(file_bytes),
                audio_sha256=audio_sha,
                requested_model=requested_model,
            )

        with stage_span("audio_decoded"):
            decode_start = time.perf_counter()
            decoded = decode(file_bytes, max_seconds=settings.max_audio_seconds)
            decode_seconds = time.perf_counter() - decode_start
            metrics.request_duration_seconds.labels(
                model=requested_model or self._registry.default_identifier,
                stage="decode",
            ).observe(decode_seconds)
            metrics.audio_duration_seconds.observe(decoded.duration_seconds)

        model = self._registry.resolve(requested_model)
        queue = await self._queue_router.for_model(model.identifier)

        with stage_span("model_inference"):
            async with queue.slot():
                infer_start = time.perf_counter()
                try:
                    output = await asyncio.to_thread(model.transcribe, decoded.samples)
                except TranscriptionError:
                    metrics.requests_total.labels(
                        model=model.identifier, status="error"
                    ).inc()
                    raise
                except Exception as ex:
                    metrics.requests_total.labels(
                        model=model.identifier, status="error"
                    ).inc()
                    _log.error(
                        "transcription_inference_failed",
                        model=model.identifier,
                        exc_type=type(ex).__name__,
                    )
                    raise TranscriptionError(
                        ErrorCode.TRANSCRIPTION_FAILED,
                        "transcription failed",
                        details={"model": model.identifier},
                    ) from ex
                inference_seconds = time.perf_counter() - infer_start
                metrics.request_duration_seconds.labels(
                    model=model.identifier, stage="inference"
                ).observe(inference_seconds)

        with stage_span("response_serialized"):
            text = (output.text or "").strip()
            no_speech = text == ""
            result = TranscriptionResult(
                text=text,
                model=model.identifier,
                audio_duration_s=decoded.duration_seconds,
                inference_duration_s=inference_seconds,
                downmix_applied=decoded.downmix_applied,
                no_speech_detected=no_speech,
                language=output.language,
                correlation_id=cid,
            )
            metrics.requests_total.labels(model=model.identifier, status="ok").inc()
            metrics.request_duration_seconds.labels(
                model=model.identifier, stage="total"
            ).observe(decode_seconds + inference_seconds)
            _log.info(
                "transcription_completed",
                model=model.identifier,
                audio_duration_s=decoded.duration_seconds,
                inference_duration_s=inference_seconds,
                audio_sha256=audio_sha,
                downmix_applied=decoded.downmix_applied,
                no_speech_detected=no_speech,
                status="ok",
            )
            return result
