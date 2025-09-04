#!/bin/bash

# Stop script for Testing Observability Platform FastAPI backend

set -e

echo "🛑 Stopping Testing Observability Platform (FastAPI)..."

# Stop services
docker-compose down

echo "✅ Services stopped successfully!"