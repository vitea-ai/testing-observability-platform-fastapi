"""
Security headers middleware for enhanced security (Tier 3+).
"""

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.config import settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to responses.
    
    These headers help protect against various attacks like XSS, clickjacking, etc.
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Add security headers to response."""
        response = await call_next(request)
        
        # Add security headers based on tier
        if settings.deployment_tier in ["staging", "production"]:
            # Prevent MIME type sniffing
            response.headers["X-Content-Type-Options"] = "nosniff"
            
            # Prevent clickjacking
            response.headers["X-Frame-Options"] = "DENY"
            
            # Enable XSS protection
            response.headers["X-XSS-Protection"] = "1; mode=block"
            
            # Control referrer information
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            # Content Security Policy
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "img-src 'self' data: https:; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline';"
            )
            
            # Add HSTS in production only
            if settings.is_production:
                response.headers["Strict-Transport-Security"] = (
                    "max-age=31536000; includeSubDomains; preload"
                )
            
            # Permissions Policy (formerly Feature Policy)
            response.headers["Permissions-Policy"] = (
                "geolocation=(), microphone=(), camera=()"
            )
        
        return response
