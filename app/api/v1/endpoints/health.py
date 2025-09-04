"""
Health check endpoints for monitoring and diagnostics.
"""

from typing import Dict, Any
import time
import psutil

from fastapi import APIRouter, status

from app.core.config import settings
from app.core.logging import logger

router = APIRouter()


@router.get(
    "/status",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Get detailed health status"
)
async def get_health_status() -> Dict[str, Any]:
    """
    Get detailed health status including system metrics.
    
    Returns:
        Detailed health information
    """
    # Get system metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    health_data = {
        "status": "healthy",
        "timestamp": time.time(),
        "service": {
            "name": settings.name,
            "version": settings.version,
            "tier": settings.deployment_tier,
            "environment": settings.env,
        },
        "system": {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
        },
        "features": settings.get_enabled_features(),
    }
    
    # Add database status if enabled
    if settings.deployment_tier != "development" or settings.database_url:
        from app.db.session import check_database_connection
        health_data["database"] = {
            "connected": await check_database_connection(),
            "url": settings.database_url.split("@")[-1] if settings.database_url else None
        }
    
    # Add cache status if enabled
    if settings.is_feature_enabled("caching") and settings.redis_url:
        from app.utils.cache import check_cache_connection
        health_data["cache"] = {
            "connected": await check_cache_connection(),
            "url": settings.redis_url.split("@")[-1] if settings.redis_url else None
        }
    
    logger.info("Health check performed", **health_data)
    
    return health_data


@router.get(
    "/ping",
    status_code=status.HTTP_200_OK,
    summary="Simple ping endpoint"
)
async def ping() -> Dict[str, str]:
    """
    Simple ping endpoint for basic availability check.
    
    Returns:
        Pong response
    """
    return {"ping": "pong"}
