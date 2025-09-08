"""
Enhanced logging configuration for observability stack integration.

This module configures structured JSON logging with the required service_name
field for proper log attribution in the Vitea observability stack.
"""

import logging
import sys
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger

from app.core.config import settings


class ServiceNameFilter(logging.Filter):
    """
    Logging filter that adds service_name to all log records.
    This is CRITICAL for the observability stack to properly route logs.
    """
    
    def __init__(self, service_name: str):
        super().__init__()
        self.service_name = service_name
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add service_name to every log record."""
        record.service_name = self.service_name
        return True


def configure_json_logging(
    service_name: str = "testing-observability-platform",
    log_level: Optional[str] = None
) -> None:
    """
    Configure JSON logging with required service_name field.
    
    This ensures all logs include the service_name field which is used
    by the observability stack for log routing and attribution.
    
    Args:
        service_name: Name of the service (CRITICAL for log routing)
        log_level: Logging level (defaults to settings)
    """
    
    if log_level is None:
        log_level = settings.log_level.value
    
    # Create JSON formatter with custom format
    # IMPORTANT: service_name is included via the filter, not the format string
    json_formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(service_name)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        json_ensure_ascii=False
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Create console handler with JSON formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    
    # Add the critical service_name filter
    service_filter = ServiceNameFilter(service_name)
    console_handler.addFilter(service_filter)
    
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    configure_logger_levels()
    
    # Log initial startup message with all required fields
    logger = logging.getLogger(service_name)
    logger.info(
        "Service started",
        extra={
            "service_name": service_name,  # Redundant but ensures it's there
            "component": "api",
            "deployment_tier": settings.deployment_tier,
            "environment": settings.env,
            "version": settings.version,
            "startup": True
        }
    )


def configure_logger_levels() -> None:
    """Configure log levels for specific loggers to reduce noise."""
    
    if settings.is_production:
        # Reduce noise in production
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    else:
        # More verbose in development/staging
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)
        logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    
    # Always suppress these noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    
    # OpenTelemetry loggers
    logging.getLogger("opentelemetry").setLevel(logging.WARNING)
    logging.getLogger("opentelemetry.instrumentation").setLevel(logging.INFO)


def get_logger_with_context(
    name: str,
    service_name: str = "testing-observability-platform",
    **context: Any
) -> logging.LoggerAdapter:
    """
    Get a logger with additional context that includes service_name.
    
    Args:
        name: Logger name (typically __name__)
        service_name: Service name for attribution
        **context: Additional context to include in all logs
        
    Returns:
        Logger adapter with context
    """
    base_logger = logging.getLogger(name)
    
    # Always include service_name in context
    full_context = {
        "service_name": service_name,
        **context
    }
    
    return logging.LoggerAdapter(base_logger, full_context)


# Convenience function for backwards compatibility
def setup_logging() -> None:
    """
    Setup logging with JSON format and service_name.
    This is called from main.py during application startup.
    """
    configure_json_logging(
        service_name=settings.name or "testing-observability-platform"
    )