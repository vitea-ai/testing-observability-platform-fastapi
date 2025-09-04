"""
API v1 main router that aggregates all endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    health,
    datasets,
    experiments,
    evaluators,
    evaluation_analysis,
)
from app.core.config import settings

# Create main API router
api_router = APIRouter()

# Include health check router
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)

# Include core platform routers
api_router.include_router(
    datasets.router,
    prefix="/datasets",
    tags=["Datasets"]
)

api_router.include_router(
    experiments.router,
    prefix="/experiments",
    tags=["Experiments"]
)

api_router.include_router(
    evaluators.router,
    prefix="/evaluators",
    tags=["Evaluators"]
)

api_router.include_router(
    evaluation_analysis.router,
    prefix="/evaluation-analysis",
    tags=["Evaluation Analysis"]
)

# Include user routes only if authentication is enabled (Tier 2+)
if settings.is_feature_enabled("authentication"):
    from app.api.v1.endpoints import users
    api_router.include_router(
        users.router,
        prefix="/users",
        tags=["Users"]
    )
