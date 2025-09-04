# API Configuration Guide

## Current Architecture (FastAPI Only - No Node.js)

This project uses **FastAPI** exclusively. There is no Node.js backend component.

## Running Services

### Core Services (Docker Containers)
- **FastAPI Application**: `eval-api-fastapi`
  - Port: 9000
  - URL: http://localhost:9000
  - Docs: http://localhost:9000/docs
  - Purpose: Main API server handling all business logic

- **PostgreSQL Database**: `eval-postgres-fastapi`
  - Port: 5433
  - Credentials: dbadmin/postgres123
  - Database: evaluation_db

- **Evaluator Service**: `eval-evaluator-service-fastapi`
  - Port: 9002
  - Purpose: AI evaluation service

## API Endpoints

All endpoints are served by FastAPI on port 9000:

### Health & Status
- `GET /health` - Health check
- `GET /ready` - Readiness probe

### Experiments
- `GET /api/v1/experiments/` - List experiments
- `POST /api/v1/experiments/` - Create experiment
- `GET /api/v1/experiments/{id}` - Get experiment
- `POST /api/v1/experiments/{id}/execute` - Execute experiment
- `GET /api/v1/experiments/{id}/results` - Get results
- `POST /api/v1/experiments/import-csv` - Import CSV results

### Datasets
- `GET /api/v1/datasets/` - List datasets
- `POST /api/v1/datasets/` - Create dataset
- `GET /api/v1/datasets/{id}` - Get dataset
- `GET /api/v1/datasets/{id}/entries` - Get dataset entries

### Evaluations
- `GET /api/v1/experiments/all-evaluations/` - List all evaluations
- `GET /api/v1/experiments/evaluations/{id}` - Get specific evaluation

### Evaluators
- `GET /api/v1/evaluators/` - List available evaluators
- `GET /api/v1/evaluators/types` - Get evaluator types
- `GET /api/v1/evaluators/categories` - Get evaluator categories

## Database Configuration

All database operations use SQLAlchemy ORM with psycopg3 driver:
- Connection: `postgresql+psycopg://dbadmin:postgres123@postgres:5432/evaluation_db`
- All CRUD operations use ORM methods (no raw SQL)
- Models defined in `app/models/`
- Services in `app/services/` handle business logic

## Important Notes

1. **No Node.js Backend**: This project does not use Express, Node.js, or any JavaScript backend
2. **ORM Only**: All database operations use SQLAlchemy ORM - no raw SQL queries
3. **Single API**: All functionality is provided by the FastAPI application on port 9000
4. **Deployment Tiers**: The app supports multiple tiers (development, integration, staging, production)
   - Currently running in: **integration** tier
   - Features enabled: Database, Audit Logging

## Starting the Services

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Restart a specific service
docker restart eval-api-fastapi

# Stop all services
docker compose down
```

## Testing the API

```bash
# Health check
curl http://localhost:9000/health

# List experiments
curl http://localhost:9000/api/v1/experiments/

# View API documentation
open http://localhost:9000/docs
```

## Frontend Integration

If you have a frontend application, configure it to:
- API Base URL: `http://localhost:9000`
- API Prefix: `/api/v1`
- No authentication required (in integration tier)

## Troubleshooting

### Database Connection Issues
- Ensure PostgreSQL container is running: `docker ps | grep postgres`
- Check logs: `docker logs eval-postgres-fastapi`
- Verify network: Containers must be on `vitea-shared-network`

### API Errors
- Check API logs: `docker logs eval-api-fastapi`
- Verify tier configuration: Should be "integration" or higher for database
- Ensure volume mounts are correct for code updates

### Port Conflicts
- API: Port 9000 (change in docker-compose.yml if needed)
- Database: Port 5433 (external), 5432 (internal)
- Evaluator: Port 9002