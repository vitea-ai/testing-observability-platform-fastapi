"""
Monitoring setup for observability (Tier 3+).
"""

from typing import Optional

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import settings
from app.core.logging import logger


def setup_monitoring(app: FastAPI) -> None:
    """
    Setup monitoring and observability tools.
    
    Args:
        app: FastAPI application instance
    """
    # Setup Prometheus metrics
    if settings.prometheus_enabled:
        setup_prometheus(app)
    
    # Setup Sentry error tracking
    if settings.sentry_dsn:
        setup_sentry()
    
    # Setup OpenTelemetry
    if settings.opentelemetry_enabled:
        setup_opentelemetry(app)


def setup_prometheus(app: FastAPI) -> None:
    """
    Setup Prometheus metrics collection.
    
    Args:
        app: FastAPI application instance
    """
    try:
        instrumentator = Instrumentator(
            should_group_status_codes=True,
            should_ignore_untemplated=True,
            should_respect_env_var=True,
            should_instrument_requests_inprogress=True,
            excluded_handlers=[".*health.*", ".*metrics.*", ".*docs.*"],
            env_var_name="ENABLE_METRICS",
            inprogress_name="fastapi_inprogress",
            inprogress_labels=True,
        )
        
        instrumentator.instrument(app).expose(app, endpoint="/metrics")
        logger.info("Prometheus metrics enabled at /metrics")
    except Exception as e:
        logger.error("Failed to setup Prometheus", error=str(e))


def setup_sentry() -> None:
    """Setup Sentry error tracking."""
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.starlette import StarletteIntegration
        from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
        
        sentry_sdk.init(
            dsn=settings.sentry_dsn,
            environment=settings.deployment_tier,
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                StarletteIntegration(transaction_style="endpoint"),
                SqlalchemyIntegration(),
            ],
            traces_sample_rate=0.1 if settings.is_production else 1.0,
            profiles_sample_rate=0.1 if settings.is_production else 1.0,
            before_send=filter_sensitive_data,
            attach_stacktrace=True,
            send_default_pii=False,  # Don't send personally identifiable information
        )
        logger.info("Sentry error tracking enabled")
    except Exception as e:
        logger.error("Failed to setup Sentry", error=str(e))


def filter_sensitive_data(event: dict, hint: dict) -> Optional[dict]:
    """
    Filter sensitive data from Sentry events.
    
    Args:
        event: Sentry event data
        hint: Additional context
    
    Returns:
        Filtered event or None to drop
    """
    # Remove sensitive fields
    sensitive_fields = [
        "password", "token", "secret", "api_key", "authorization",
        "ssn", "credit_card", "email", "phone"
    ]
    
    if "request" in event and "data" in event["request"]:
        for field in sensitive_fields:
            if field in event["request"]["data"]:
                event["request"]["data"][field] = "[REDACTED]"
    
    # Remove sensitive headers
    if "request" in event and "headers" in event["request"]:
        for header in ["authorization", "cookie", "x-api-key"]:
            if header in event["request"]["headers"]:
                event["request"]["headers"][header] = "[REDACTED]"
    
    return event


def setup_opentelemetry(app: FastAPI) -> None:
    """
    Setup OpenTelemetry tracing.
    
    Args:
        app: FastAPI application instance
    """
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        
        # Create resource
        resource = Resource.create({
            "service.name": settings.name,
            "service.version": settings.version,
            "deployment.environment": settings.deployment_tier,
        })
        
        # Setup tracer
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # Add OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint="http://localhost:4317",  # Configure based on your setup
            insecure=True,
        )
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)
        
        logger.info("OpenTelemetry tracing enabled")
    except Exception as e:
        logger.error("Failed to setup OpenTelemetry", error=str(e))
