"""
Rate limiting middleware (Tier 3+).
"""

import time
from typing import Dict, List
from collections import defaultdict

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.config import settings
from app.core.logging import logger


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.
    
    In production, use Redis for distributed rate limiting.
    """
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            app: FastAPI application
            calls: Number of calls allowed
            period: Time period in seconds
        """
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients: Dict[str, List[float]] = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply rate limiting."""
        if not settings.is_feature_enabled("rate_limiting"):
            return await call_next(request)
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/ready", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        # Get client identifier (IP address or user ID)
        client_id = request.client.host if request.client else "unknown"
        
        # Get current time
        now = time.time()
        
        # Clean old entries
        self.clients[client_id] = [
            timestamp for timestamp in self.clients[client_id]
            if now - timestamp < self.period
        ]
        
        # Check rate limit
        if len(self.clients[client_id]) >= self.calls:
            logger.warning(
                "Rate limit exceeded",
                client=client_id,
                calls=len(self.clients[client_id]),
                period=self.period
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Add current request
        self.clients[client_id].append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(
            self.calls - len(self.clients[client_id])
        )
        response.headers["X-RateLimit-Reset"] = str(int(now + self.period))
        
        return response
