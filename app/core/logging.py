"""
Logging configuration with structured logging support.
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level

from app.core.config import settings


def setup_logging() -> None:
    """
    Configure structured logging based on deployment tier.
    """
    # Configure structlog
    processors = [
        TimeStamper(fmt="iso"),
        add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.set_exc_info,
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
    ]
    
    # Add tier-specific processors
    if settings.is_production:
        processors.append(JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level.value)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=settings.log_level.value,
    )
    
    # Set specific logger levels
    if settings.is_production:
        # Reduce noise in production
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    else:
        # More verbose in development
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)
        logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    
    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return structlog.get_logger(name).bind(
        tier=settings.deployment_tier,
        environment=settings.env,
        service=settings.name,
    )


# Create default logger instance
logger = get_logger(__name__)
