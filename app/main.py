"""
Main FastAPI application with tier-based configuration.

This module creates and configures the FastAPI application with progressive
enhancement based on deployment tier.
"""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.api.v1.router import api_router
from app.db.session import init_db, close_db
from app.middleware.security import SecurityHeadersMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.audit import AuditLoggingMiddleware
# from app.utils.monitoring import setup_monitoring  # Disabled for now


# ==========================================
# Application Metadata
# ==========================================
APP_METADATA = {
    "title": settings.name,
    "description": """
    ## FastAPI Service Template
    
    A production-ready FastAPI template with progressive enhancement through deployment tiers.
    
    ### Features
    - 🚀 **Fast**: Built with FastAPI and uv for maximum performance
    - 🔒 **Secure**: Progressive security features based on deployment tier
    - 📊 **Observable**: Built-in monitoring and logging
    - 🏗️ **Scalable**: Tiered architecture for gradual complexity
    
    ### Current Tier
    **{tier}** - {features}
    """.format(
        tier=settings.deployment_tier.value.upper(),
        features=", ".join([
            k.replace("_", " ").title() 
            for k, v in settings.get_enabled_features().items() 
            if v
        ]) or "Basic Features"
    ),
    "version": settings.version,
    "contact": {
        "name": "API Support",
        "email": "api@example.com",
    },
    "license_info": {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
}


# ==========================================
# Lifespan Management
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle with tier-appropriate setup and teardown.
    """
    # ========== STARTUP ==========
    logger.info(
        "Starting application",
        tier=settings.deployment_tier,
        features=settings.get_enabled_features()
    )
    
    # Initialize database (Tier 2+)
    if settings.deployment_tier != "development" or settings.database_url:
        await init_db()
        logger.info("Database initialized")
    
    # Setup monitoring (Tier 3+)
    # if settings.is_feature_enabled("monitoring"):
    #     setup_monitoring(app)
    #     logger.info("Monitoring initialized")
    
    # Setup cache (Tier 3+)
    if settings.is_feature_enabled("caching") and settings.redis_url:
        from app.utils.cache import init_cache
        await init_cache()
        logger.info("Cache initialized")
    
    # Application is ready
    logger.info("Application startup complete")
    
    yield
    
    # ========== SHUTDOWN ==========
    logger.info("Shutting down application")
    
    # Close database connections
    if settings.deployment_tier != "development" or settings.database_url:
        await close_db()
        logger.info("Database connections closed")
    
    # Close cache connections
    if settings.is_feature_enabled("caching") and settings.redis_url:
        from app.utils.cache import close_cache
        await close_cache()
        logger.info("Cache connections closed")
    
    logger.info("Application shutdown complete")


# ==========================================
# Application Factory
# ==========================================
def create_application() -> FastAPI:
    """
    Create FastAPI application with tier-appropriate configuration.
    
    Returns:
        FastAPI: Configured application instance
    """
    # Setup logging
    setup_logging()
    
    # Create app with conditional documentation
    app = FastAPI(
        **APP_METADATA,
        docs_url="/docs" if settings.docs_enabled else None,
        redoc_url="/redoc" if settings.docs_enabled else None,
        openapi_url="/openapi.json" if settings.docs_enabled else None,
        lifespan=lifespan,
    )
    
    # ==========================================
    # Middleware Configuration
    # ==========================================
    
    # CORS middleware (all tiers)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Trusted host middleware (Tier 2+)
    if settings.deployment_tier != "development":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.allowed_hosts
        )
    
    # Security headers (Tier 3+)
    if settings.is_feature_enabled("monitoring"):
        app.add_middleware(SecurityHeadersMiddleware)
    
    # Rate limiting (Tier 3+)
    if settings.is_feature_enabled("rate_limiting"):
        app.add_middleware(
            RateLimitMiddleware,
            calls=settings.rate_limit_calls,
            period=settings.rate_limit_period
        )
    
    # Audit logging (Tier 2+)
    if settings.is_feature_enabled("audit_logging"):
        app.add_middleware(AuditLoggingMiddleware)
    
    # Request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Add unique request ID to each request."""
        request_id = f"{time.time()}-{id(request)}"
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    
    # ==========================================
    # Routes Configuration
    # ==========================================
    
    # Health check endpoint (all tiers)
    @app.get(
        "/health",
        tags=["Health"],
        summary="Health Check",
        response_model=Dict[str, Any]
    )
    async def health_check():
        """
        Check application health and return tier information.
        """
        return {
            "status": "healthy",
            "tier": settings.deployment_tier,
            "version": settings.version,
            "features": settings.get_enabled_features(),
            "timestamp": time.time()
        }
    
    # Ready check endpoint (Kubernetes readiness probe)
    @app.get(
        "/ready",
        tags=["Health"],
        summary="Readiness Check"
    )
    async def readiness_check():
        """
        Check if application is ready to serve requests.
        """
        # Add checks for database, cache, etc. based on tier
        checks = {"app": True}
        
        if settings.deployment_tier != "development":
            # Check database connection
            from app.db.session import check_database_connection
            checks["database"] = await check_database_connection()
        
        if settings.is_feature_enabled("caching") and settings.redis_url:
            # Check Redis connection
            from app.utils.cache import check_cache_connection
            checks["cache"] = await check_cache_connection()
        
        all_ready = all(checks.values())
        
        return JSONResponse(
            status_code=status.HTTP_200_OK if all_ready else status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "ready": all_ready,
                "checks": checks,
                "timestamp": time.time()
            }
        )
    
    # API routes
    app.include_router(
        api_router,
        prefix=settings.api_v1_prefix
    )
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """
        Root endpoint with API information.
        """
        return {
            "message": f"Welcome to {settings.name}",
            "version": settings.version,
            "tier": settings.deployment_tier,
            "docs": "/docs" if settings.docs_enabled else None,
            "health": "/health",
            "api": settings.api_v1_prefix
        }
    
    # ==========================================
    # Exception Handlers
    # ==========================================
    
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        """Handle 404 errors."""
        return JSONResponse(
            status_code=404,
            content={
                "error": "Not Found",
                "message": f"The requested URL {request.url.path} was not found.",
                "path": request.url.path
            }
        )
    
    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc):
        """Handle 500 errors."""
        logger.error(
            "Internal server error",
            exc_info=exc,
            path=request.url.path,
            method=request.method
        )
        
        # Don't expose internal details in production
        if settings.is_production:
            message = "An internal error occurred. Please try again later."
        else:
            message = str(exc)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": message,
                "request_id": getattr(request.state, "request_id", None)
            }
        )
    
    return app


# ==========================================
# Create Application Instance
# ==========================================
app = create_application()


# ==========================================
# Main Entry Point
# ==========================================
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.value.lower(),
        access_log=settings.debug,
    )
