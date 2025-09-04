# Testing Observability Platform - FastAPI Backend

## Overview

This is the FastAPI version of the Testing Observability Platform, converted from the original Node.js implementation. It follows the FastAPI template structure with progressive tier-based development.

## Quick Start

```bash
# Using Docker (recommended)
./scripts/start.sh

# Or manually with uv
uv sync
uv run uvicorn app.main:app --reload --port 9000
```

## API Structure

The platform provides the following main endpoints:

### Core Endpoints
- `/api/v1/datasets` - Test dataset management
- `/api/v1/experiments` - Experiment execution and management
- `/api/v1/evaluators` - Available evaluation metrics
- `/api/v1/evaluations` - Evaluation runs (to be implemented)
- `/api/v1/evaluation-analysis` - Result analysis (to be implemented)

### Development Tiers

Following the FastAPI template approach:

**Tier 1 (Development)** - Current implementation
- In-memory storage for quick prototyping
- Basic CRUD operations
- No authentication required
- Swagger UI enabled

**Tier 2 (Integration)** - Next steps
- PostgreSQL database integration
- SQLAlchemy models
- Alembic migrations
- Basic JWT authentication

**Tier 3 (Staging)**
- Redis caching
- Rate limiting
- Monitoring with Prometheus
- Background task processing

**Tier 4 (Production)**
- Full HIPAA compliance
- End-to-end encryption
- Audit logging
- High availability

## Key Differences from Node.js Version

1. **Framework**: FastAPI instead of Express
2. **Type Safety**: Pydantic models for validation
3. **Async**: Native async/await support
4. **Documentation**: Auto-generated OpenAPI docs
5. **Tiered Architecture**: Progressive enhancement approach

## Database Schema (Tier 2+)

Main tables:
- `datasets` - Test datasets with JSONB data field
- `experiments` - Experiment configurations
- `test_results` - Individual test results
- `evaluations` - Evaluation runs
- `evaluation_metrics` - Available metrics

## Configuration

Environment variables use `APP_` prefix:
- `APP_DATABASE_URL` - PostgreSQL connection
- `APP_PORT` - API port (default: 9000)
- `APP_DEPLOYMENT_TIER` - Current tier
- `APP_EVALUATOR_SERVICE_URL` - Evaluator service URL

## Testing

```bash
# Run tests
uv run pytest

# With coverage
uv run pytest --cov=app

# Lint and format
uv run black app/
uv run ruff check app/
```

## Next Steps

1. Implement database models and migrations
2. Add service layer for business logic
3. Integrate with evaluator service
4. Add evaluation and analysis endpoints
5. Implement authentication (tier 2)
6. Add comprehensive tests