"""
Audit logging middleware for compliance (Tier 2+).
"""

import time
import json
from typing import Dict, Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.config import settings
from app.core.logging import logger


class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for audit logging of all API requests.
    
    Logs all requests and responses for compliance and debugging.
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Log request and response details."""
        if not settings.is_feature_enabled("audit_logging"):
            return await call_next(request)
        
        # Skip logging for health checks and docs
        if request.url.path in ["/health", "/ready", "/docs", "/openapi.json", "/redoc"]:
            return await call_next(request)
        
        # Start timing
        start_time = time.time()
        
        # Extract request details
        request_details = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None,
            "headers": {
                k: v for k, v in request.headers.items()
                if k.lower() not in ["authorization", "cookie"]  # Don't log sensitive headers
            }
        }
        
        # Get user info if available
        user_info = None
        if hasattr(request.state, "user"):
            user_info = request.state.user
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log audit event
        audit_event = {
            "timestamp": time.time(),
            "request": request_details,
            "response": {
                "status_code": response.status_code,
                "duration_seconds": duration
            },
            "user": user_info,
            "request_id": getattr(request.state, "request_id", None),
            "tier": settings.deployment_tier
        }
        
        # Log based on response status
        if response.status_code >= 500:
            logger.error("AUDIT: Server error", **audit_event)
        elif response.status_code >= 400:
            logger.warning("AUDIT: Client error", **audit_event)
        else:
            logger.info("AUDIT: Request processed", **audit_event)
        
        # In production, send to audit log storage
        if settings.is_production:
            await self._store_audit_log(audit_event)
        
        return response
    
    async def _store_audit_log(self, event: Dict[str, Any]) -> None:
        """
        Store audit log in persistent storage.
        
        In production, this would write to a database or log aggregation service.
        """
        # Placeholder for production audit log storage
        pass
