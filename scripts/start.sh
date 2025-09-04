#!/bin/bash

# Start script for Testing Observability Platform FastAPI backend

set -e

echo "🚀 Starting Testing Observability Platform (FastAPI)..."

# Check if .env file exists, if not copy from example
if [ ! -f .env ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
fi

# Check if shared network exists
if ! docker network inspect vitea-shared-network >/dev/null 2>&1; then
    echo "🔗 Creating shared Docker network..."
    docker network create vitea-shared-network
fi

# Start services with docker-compose
echo "🐳 Starting Docker containers..."
docker-compose up -d --build

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 5

# Check service health
echo "✅ Checking service health..."
curl -s http://localhost:9000/health | jq . || echo "⚠️  API not responding yet, may still be starting..."

echo ""
echo "✨ Services started successfully!"
echo ""
echo "📚 Access points:"
echo "  - API Documentation: http://localhost:9000/docs"
echo "  - Alternative Docs: http://localhost:9000/redoc"
echo "  - Health Check: http://localhost:9000/health"
echo "  - API Base URL: http://localhost:9000/api/v1"
echo ""
echo "📝 Available endpoints:"
echo "  - Datasets: http://localhost:9000/api/v1/datasets"
echo "  - Experiments: http://localhost:9000/api/v1/experiments"
echo "  - Evaluators: http://localhost:9000/api/v1/evaluators"
echo ""
echo "To view logs: docker-compose logs -f"
echo "To stop: ./scripts/stop.sh or docker-compose down"