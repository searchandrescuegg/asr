from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from asr.observability.logging import get_logger

REGISTRY = CollectorRegistry()
_log = get_logger("asr.observability.metrics")

requests_total = Counter(
    "asr_requests_total",
    "Total transcription requests handled.",
    ("model", "status"),
    registry=REGISTRY,
)

request_duration_seconds = Histogram(
    "asr_request_duration_seconds",
    "Time spent in each pipeline stage.",
    ("model", "stage"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
    registry=REGISTRY,
)

audio_duration_seconds = Histogram(
    "asr_audio_duration_seconds",
    "Distribution of submitted audio durations.",
    buckets=(1, 5, 15, 30, 60, 120, 300, 600),
    registry=REGISTRY,
)

queue_depth = Gauge(
    "asr_queue_depth",
    "Current depth of the per-model wait queue.",
    ("model",),
    registry=REGISTRY,
)

queue_rejections_total = Counter(
    "asr_queue_rejections_total",
    "Requests rejected with MODEL_BUSY because the per-model queue was full.",
    ("model",),
    registry=REGISTRY,
)

gpu_memory_used_bytes = Gauge(
    "asr_gpu_memory_used_bytes",
    "GPU memory currently in use, in bytes (0 when no GPU).",
    registry=REGISTRY,
)

gpu_utilization_ratio = Gauge(
    "asr_gpu_utilization_ratio",
    "GPU utilization as a fraction in [0, 1] (0 when no GPU).",
    registry=REGISTRY,
)


def sample_gpu_metrics() -> None:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_memory_used_bytes.set(mem.used)
            gpu_utilization_ratio.set(util.gpu / 100.0)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        gpu_memory_used_bytes.set(0)
        gpu_utilization_ratio.set(0)


def metrics_router() -> APIRouter:
    router = APIRouter()

    @router.get("/metrics", include_in_schema=False)
    def _metrics() -> Response:
        sample_gpu_metrics()
        return Response(
            content=generate_latest(REGISTRY),
            media_type=CONTENT_TYPE_LATEST,
        )

    return router
