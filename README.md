# Testing Observability Platform - FastAPI

A production-ready FastAPI service for dataset management, CSV processing, and evaluation workflows.

## Features

- CSV upload and processing with streaming support
- Dataset management with CRUD operations  
- Evaluation and experiment tracking
- Progressive tier-based architecture
- Production-ready observability and monitoring

## Development

```bash
# Install dependencies
uv sync

# Run development server
uv run python app/main.py
```

## Docker

```bash
# Build image
docker build -t eval-api-fastapi .

# Run container
docker run -p 8000:8000 eval-api-fastapi
```
