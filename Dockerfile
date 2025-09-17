FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files and README
COPY pyproject.toml uv.lock README.md ./

# Install dependencies using uv with staging extras
RUN uv sync --frozen --extra staging

# Copy application code
COPY . .

# Create data directory for SQLite and file uploads
RUN mkdir -p /app/data

# Expose port (can be overridden with build arg)
ARG APP_PORT=8000
EXPOSE ${APP_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${APP_PORT:-8000}/health || exit 1

# Run the application
CMD ["sh", "-c", "uv run --no-sync python -m uvicorn app.main:app --host 0.0.0.0 --port ${APP_PORT:-8000}"]