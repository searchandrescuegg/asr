from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from asr.observability.logging import current_correlation_id

_initialised = False


def configure_tracing(service_name: str = "asr") -> None:
    global _initialised
    if _initialised:
        return
    _initialised = True

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


def instrument_fastapi(app) -> None:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    FastAPIInstrumentor.instrument_app(app)


def get_tracer(name: str = "asr"):
    return trace.get_tracer(name)


@contextmanager
def stage_span(name: str) -> Iterator[trace.Span]:
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        cid = current_correlation_id()
        if cid:
            span.set_attribute("correlation_id", cid)
        yield span
