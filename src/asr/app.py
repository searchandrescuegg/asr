from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from asr.api.errors import TranscriptionError, transcription_error_handler
from asr.config import get_settings
from asr.models.registry import ModelRegistry
from asr.observability.logging import (
    CorrelationIdMiddleware,
    configure_logging,
    get_logger,
)
from asr.observability.metrics import metrics_router
from asr.observability.tracing import configure_tracing, instrument_fastapi
from asr.pipeline.queue import QueueRouter
from asr.pipeline.transcribe import Pipeline

_log = get_logger("asr.app")


def _build_models() -> list:
    """Instantiate concrete model adapters from settings.enabled_model_ids.

    Adapter imports are local so a misconfigured model never blocks startup
    of the others. Each adapter is tried; failures are logged and the model
    is left out of the registry.
    """
    settings = get_settings()
    models = []
    for model_id in settings.enabled_model_ids:
        try:
            if model_id == "parakeet-tdt-0.6b-v3":
                from asr.models.parakeet import ParakeetModel

                models.append(ParakeetModel())
            elif model_id == "seamless-m4t-v2":
                from asr.models.seamless import SeamlessModel

                models.append(SeamlessModel())
            else:
                _log.warning("unknown_model_id", model_id=model_id)
        except Exception as ex:
            _log.error(
                "model_instantiation_failed",
                model_id=model_id,
                exc_type=type(ex).__name__,
            )
    return models


def create_app(models: list | None = None) -> FastAPI:
    configure_logging()
    configure_tracing()
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        registry = ModelRegistry(default_identifier=settings.default_model)
        for model in models if models is not None else _build_models():
            try:
                _log.info("model_loading", model=model.identifier)
                model.load()
                model.warm_up()
                _log.info("model_ready", model=model.identifier)
            except Exception as ex:
                model.last_error = str(ex)
                from asr.models.base import ModelState

                model.state = ModelState.FAILED
                _log.error(
                    "model_load_failed",
                    model=model.identifier,
                    exc_type=type(ex).__name__,
                )
            registry.register(model)

        app.state.registry = registry
        app.state.queue_router = QueueRouter(depth=settings.queue_depth)
        app.state.pipeline = Pipeline(registry, app.state.queue_router)
        _log.info("server_ready", host=settings.host, port=settings.port)
        yield

    app = FastAPI(
        title="ASR — Multi-Model Speech Transcription",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(CorrelationIdMiddleware)
    app.add_exception_handler(TranscriptionError, transcription_error_handler)
    app.include_router(metrics_router())

    from asr.api.v1.routes import router as v1_router

    app.include_router(v1_router)

    try:
        instrument_fastapi(app)
    except Exception:
        pass

    try:
        from asr.ui.gradio_app import mount_ui

        mount_ui(app)
    except Exception as ex:
        _log.warning("ui_mount_skipped", exc_type=type(ex).__name__)

    return app
