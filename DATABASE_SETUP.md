# Database Setup Guide

## Overview

The Testing Observability Platform uses PostgreSQL as its primary database with SQLAlchemy ORM and Alembic for migrations, following the FastAPI template's tier-based architecture.

## Quick Start

### 1. Using Docker (Recommended)

```bash
# Start the database with Docker Compose
docker-compose up -d postgres

# Wait for database to be ready
sleep 5

# Initialize the database
uv run python scripts/init_db.py

# Test the connection
uv run python scripts/test_db.py
```

### 2. Manual Setup

If you have PostgreSQL installed locally:

```bash
# Create database
createdb -U dbadmin evaluation_db

# Set environment variable
export APP_DATABASE_URL="postgresql://dbadmin:postgres123@localhost:5433/evaluation_db"

# Initialize tables
uv run python scripts/init_db.py
```

## Database Configuration

The database connection is configured in `app/core/config.py` and uses environment variables:

```python
# Default connection (development)
DATABASE_URL=postgresql://dbadmin:postgres123@localhost:5433/evaluation_db

# For Docker
APP_DATABASE_URL=postgresql://dbadmin:postgres123@postgres:5432/evaluation_db
```

## Database Models

Following the template pattern, models are defined in `app/models/`:

### Core Models

1. **Dataset** (`app/models/dataset.py`)
   - Stores test datasets with JSONB data field
   - Includes status tracking and versioning
   - Uses UUID primary keys

2. **Experiment** (`app/models/experiment.py`)
   - Manages experiment runs
   - Links to datasets
   - Tracks execution status and progress

3. **TestResult** (`app/models/experiment.py`)
   - Individual test case results
   - Links to experiments
   - Stores input/output and execution details

4. **Evaluation** (`app/models/evaluation.py`)
   - Evaluation runs and results
   - Links to experiments
   - Stores evaluator configuration and scores

## Database Patterns

### Session Management

The platform uses async SQLAlchemy with the following pattern:

```python
from app.core.dependencies import get_db
from sqlalchemy.ext.asyncio import AsyncSession

@router.get("/items")
async def get_items(db: AsyncSession = Depends(get_db)):
    service = ItemService(db)
    return await service.list_items()
```

### Service Layer

All database operations go through service classes:

```python
from app.services.dataset_service import DatasetService

service = DatasetService(db)
dataset = await service.create_dataset(dataset_data)
```

### Tier-Based Features

- **Tier 1 (Development)**: In-memory storage fallback
- **Tier 2 (Integration)**: Full PostgreSQL with async support
- **Tier 3 (Staging)**: Add Redis caching
- **Tier 4 (Production)**: Full audit logging and encryption

## Migrations

### Creating Migrations

```bash
# Auto-generate migration from model changes
uv run alembic revision --autogenerate -m "Add new field to dataset"

# Apply migrations
uv run alembic upgrade head

# Rollback one migration
uv run alembic downgrade -1
```

### Migration Files

Migrations are stored in `alembic/versions/`. The system automatically detects model changes when using `--autogenerate`.

## Testing Database Operations

### Unit Tests

```python
# tests/test_database.py
import pytest
from app.services.dataset_service import DatasetService

@pytest.mark.asyncio
async def test_create_dataset(db_session):
    service = DatasetService(db_session)
    dataset = await service.create_dataset(...)
    assert dataset.id is not None
```

### Integration Tests

```bash
# Run database integration tests
uv run pytest tests/integration/test_db_operations.py
```

## Troubleshooting

### Connection Issues

1. Check PostgreSQL is running:
   ```bash
   docker-compose ps postgres
   ```

2. Verify connection settings:
   ```bash
   uv run python scripts/test_db.py
   ```

3. Check logs:
   ```bash
   docker-compose logs postgres
   ```

### Common Errors

- **"relation does not exist"**: Run migrations or init script
- **"connection refused"**: Check if PostgreSQL is running on correct port (5433)
- **"password authentication failed"**: Verify credentials in .env file

## Performance Optimization

### Indexes

Key indexes are automatically created:
- UUID primary keys
- Foreign key relationships
- Status and type fields for filtering

### Connection Pooling

Configured in `app/db/session.py`:
- Pool size: 20 connections
- Max overflow: 30 connections
- Pool recycle: 3600 seconds

### JSONB Queries

For efficient JSONB queries:
```python
# Query JSONB field
query = select(Dataset).where(
    Dataset.data.op('@>')([{'input': 'test'}])
)
```

## Backup and Restore

### Backup

```bash
# Backup database
docker-compose exec postgres pg_dump -U dbadmin evaluation_db > backup.sql
```

### Restore

```bash
# Restore database
docker-compose exec -T postgres psql -U dbadmin evaluation_db < backup.sql
```

## Security

- Passwords are never stored in code
- Use environment variables for credentials
- Enable SSL in production
- Implement row-level security for multi-tenancy

## Next Steps

1. Run initial migration to create tables
2. Test database connection with test script
3. Configure appropriate tier settings
4. Implement caching layer (Tier 3+)
5. Add audit logging (Tier 4)