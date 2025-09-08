"""
OpenTelemetry tracing setup for FastAPI application.

This module configures distributed tracing integration with the Vitea observability stack.
Traces are exported to the central OpenTelemetry collector via OTLP HTTP protocol.
"""

import os
from typing import Optional

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.core.logging import logger


def setup_tracing(
    app: FastAPI, 
    service_name: str, 
    service_version: str = "1.0.0",
    environment: Optional[str] = None
) -> None:
    """
    Configure OpenTelemetry tracing for FastAPI application.
    
    This sets up:
    - Resource identification with service metadata
    - OTLP HTTP span exporter to observability collector
    - Automatic FastAPI request instrumentation
    - Automatic HTTP client instrumentation
    
    Args:
        app: FastAPI application instance
        service_name: Name of the service (e.g., "testing-observability-platform")
        service_version: Version of the service
        environment: Deployment environment (defaults to APP_ENV or "development")
    """
    
    if environment is None:
        environment = os.getenv("APP_ENV", "development")
    
    # Create resource with service identification
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
            "deployment.environment": environment,
            "service.namespace": "vitea",
        }
    )

    # Configure tracer provider
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    # OTLP HTTP exporter configuration
    # Use localhost for local development, host.docker.internal for Docker
    default_endpoint = "http://localhost:10318/v1/traces"
    if os.path.exists("/.dockerenv"):
        default_endpoint = "http://host.docker.internal:10318/v1/traces"
    
    otlp_endpoint = os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        default_endpoint,
    )
    
    span_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        timeout=10,
        headers={}
    )
    
    # Batch span processor for performance (non-blocking)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            span_exporter,
            max_queue_size=4096,
            max_export_batch_size=512,
            schedule_delay_millis=1000,  # Export every 1 second
            export_timeout_millis=5000,
        )
    )

    # Instrument FastAPI for automatic request/response spans
    FastAPIInstrumentor.instrument_app(app)
    
    # Instrument HTTP requests for automatic client spans
    RequestsInstrumentor().instrument()

    logger.info(
        "OpenTelemetry tracing configured",
        service_name=service_name,
        service_version=service_version,
        environment=environment,
        otlp_endpoint=otlp_endpoint
    )


def get_tracer(name: str) -> trace.Tracer:
    """
    Get a tracer instance for manual span creation.
    
    Args:
        name: Tracer name (typically __name__ or service component)
        
    Returns:
        Configured tracer instance
    """
    return trace.get_tracer(name)


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID for correlation.
    
    Returns:
        Hex-encoded trace ID if available, None otherwise
    """
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        return f"{current_span.get_span_context().trace_id:032x}"
    return None


def get_current_span_id() -> Optional[str]:
    """
    Get the current span ID for correlation.
    
    Returns:
        Hex-encoded span ID if available, None otherwise
    """
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        return f"{current_span.get_span_context().span_id:016x}"
    return None