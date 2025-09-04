# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a FastAPI template implementing a tiered architecture approach for building HIPAA-compliant governance platforms. The project uses a progressive enhancement strategy with 4 deployment tiers, from local development to production-ready HIPAA compliance.

## Core Architecture

### Tiered Deployment Strategy
The application follows a 4-tier progressive enhancement model:
- **Tier 1 (Development)**: In-memory storage, basic CRUD, Swagger UI
- **Tier 2 (Integration)**: PostgreSQL, JWT auth, Alembic migrations
- **Tier 3 (Staging)**: Redis caching, rate limiting, monitoring
- **Tier 4 (Production)**: Full HIPAA compliance, encryption, audit logging

### Application Structure
The FastAPI application (`app/main.py`) uses:
- **Lifespan Context Manager**: Handles startup/shutdown based on deployment tier
- **Middleware Stack**: CORS, RequestID, Timing, and tier-specific security middleware
- **Dependency Injection**: Environment-based service initialization
- **Structured Logging**: Using `structlog` for consistent log formatting

### Key Design Patterns
- **Settings Management**: Pydantic settings with environment-based configuration
- **Service Layer**: Business logic separated from API endpoints
- **Schema/Model Separation**: Pydantic schemas for validation, SQLAlchemy models for DB
- **Mock Services**: In-memory implementations for Tier 1 development

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv sync

# Install dev dependencies
uv sync --dev
```

### Running the Application
```bash
# Development server with auto-reload
uv run uvicorn app.main:app --reload --port 8000

# Direct Python execution (uses settings from app.core.config)
uv run python app/main.py

# Production server (no reload)
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=app --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_main.py -v

# Run tests by marker
uv run pytest -m unit  # Unit tests only
uv run pytest -m integration  # Integration tests only
```

### Code Quality
```bash
# Format code (required before commits)
uv run black app/ tests/
uv run isort app/ tests/

# Lint code
uv run ruff check app/ tests/

# Type checking
uv run mypy app/
```

### Database Operations (Tier 2+)
```bash
# Create new migration
uv run alembic revision --autogenerate -m "Description"

# Apply migrations
uv run alembic upgrade head

# Rollback one migration
uv run alembic downgrade -1

# View migration history
uv run alembic history
```

### Dependency Management
```bash
# Add production dependency
uv add package-name

# Add development dependency
uv add --dev package-name

# Update all dependencies
uv sync

# Show installed packages
uv pip list
```

## Configuration Management

The application uses Pydantic Settings with the following key environment variables:
- `APP_ENV`: Environment (development/integration/staging/production)
- `DEBUG`: Enable debug mode and auto-reload
- `DATABASE_URL`: PostgreSQL connection string (Tier 2+)
- `REDIS_URL`: Redis connection string (Tier 3+)
- `SECRET_KEY`: JWT signing key (Tier 2+)
- `CORS_ORIGINS`: Allowed CORS origins (JSON array)

Configuration is loaded from:
1. Environment variables
2. `.env` file (create from `.env.example`)
3. Default values in `app/core/config.py`

## Testing Strategy

### Test Organization
- `tests/unit/`: Fast, isolated unit tests
- `tests/integration/`: Tests requiring database/external services
- `tests/conftest.py`: Shared pytest fixtures

### Running Tests for Specific Tiers
```bash
# Tier 1 (in-memory only)
APP_ENV=development uv run pytest tests/unit

# Tier 2+ (with database)
APP_ENV=integration uv run pytest tests/
```

## API Documentation

When `SHOW_DOCS=true` (default in development):
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Important Patterns and Conventions

### File Organization
- API endpoints go in `app/api/v1/endpoints/`
- Business logic in `app/services/`
- Database models in `app/models/`
- Pydantic schemas in `app/schemas/`
- Utility functions in `app/utils/`

### Dependency Injection Pattern
The application uses FastAPI's dependency injection system. Common dependencies are defined in `app/core/dependencies.py`.

### Error Handling
Custom exceptions are defined in `app/core/exceptions.py` with corresponding handlers registered in the main application.

### Middleware Activation
Middleware is progressively activated based on the deployment tier, configured in `app/main.py:setup_middleware()`.

## Development Workflow

1. **Start Development**: Set `APP_ENV=development` and use in-memory storage
2. **Add Features**: Implement endpoints, services, and schemas
3. **Test Locally**: Run tests with `uv run pytest`
4. **Format & Lint**: Run formatters before committing
5. **Tier Progression**: Move to higher tiers by updating environment configuration

## Tier-Specific Considerations

### Tier 1 (Development)
- No authentication required
- In-memory storage via mock services
- All API documentation enabled
- Focus on rapid prototyping

### Tier 2+ (Integration and above)
- PostgreSQL required (using psycopg3 driver for better SQLAlchemy compatibility)
- Run migrations with Alembic
- JWT authentication enabled
- Integration tests become relevant
- Note: For maximum performance scenarios, asyncpg is available as an optional dependency

### Tier 3+ (Staging and above)
- Redis required for caching
- Rate limiting activated
- Security headers enabled
- Monitoring endpoints available

### Tier 4 (Production)
- Full HIPAA compliance features
- Audit logging enabled
- Encryption at rest and in transit
- Disaster recovery procedures