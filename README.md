# FastAPI Service Template - ViTea

## 🚀 Quick Start

This is a production-ready FastAPI template with a tiered architecture approach for building HIPAA-compliant governance platforms. Start simple, add complexity as needed.

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (optional, for containerized development)
- PostgreSQL (for Tier 2+)
- Redis (for Tier 3+)

### Installation

You should clone and remove the .git folder and do git init after you have changed to folder name to make it your new repo.

```bash
# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Clone this template
git clone <repository>
cd FastAPI_template

# Install dependencies (automatically creates venv)
uv sync

# Copy environment template
cp .env.example .env

# Run the development server
uv run uvicorn app.main:app --reload --port 8000
```

### Access the Application

- **API Documentation**: <http://localhost:8000/docs>
- **Alternative Docs**: <http://localhost:8000/redoc>
- **Health Check**: <http://localhost:8000/health>

## 📁 Project Structure

```
FastAPI_template/
├── app/                        # Application code
│   ├── api/                   # API routes
│   │   └── v1/                # API version 1
│   │       ├── endpoints/     # Individual endpoints
│   │       └── router.py      # Main API router
│   ├── core/                  # Core functionality
│   │   ├── config.py         # Configuration management
│   │   ├── security.py       # Security utilities
│   │   └── dependencies.py   # Dependency injection
│   ├── models/               # Database models
│   ├── schemas/              # Pydantic schemas
│   ├── services/             # Business logic
│   ├── db/                   # Database utilities
│   ├── utils/                # Utility functions
│   └── main.py              # FastAPI application entry
├── tests/                    # Test files
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── conftest.py         # Pytest fixtures
├── alembic/                 # Database migrations
│   └── versions/           # Migration files
├── scripts/                 # Utility scripts
├── docs/                    # Documentation
│   ├── architecture.md     # Architecture decisions
│   ├── deployment.md       # Deployment guide
│   ├── development.md      # Development guide
│   ├── security.md         # Security guidelines
│   └── tiers.md           # Tier progression guide
├── docker/                  # Docker configurations
│   ├── Dockerfile.dev      # Development Dockerfile
│   └── Dockerfile.prod     # Production Dockerfile
├── .env.example            # Environment variables template
├── pyproject.toml          # Project dependencies
├── alembic.ini            # Alembic configuration
└── README.md              # This file
```

## 🎯 Development Tiers

This template follows a 4-tier progressive enhancement strategy:

### Tier 1: Local Development (Current)

- Simple FastAPI with basic CRUD
- In-memory storage for quick prototyping
- Swagger UI enabled
- No authentication
- Focus on business logic

### Tier 2: Integration Testing

- PostgreSQL database (with psycopg3 driver)
- Basic JWT authentication
- Alembic migrations
- Integration tests
- SQLAlchemy async ORM support

### Tier 3: Staging Deployment

- Redis caching
- Rate limiting
- Monitoring (Prometheus/Grafana)
- Security headers
- Performance optimization

### Tier 4: Production

- Full HIPAA compliance
- End-to-end encryption
- Audit logging
- High availability
- Disaster recovery

See [docs/tiers.md](docs/tiers.md) for detailed tier progression guide.

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run unit tests only
uv run pytest tests/unit

# Run with coverage
uv run pytest --cov=app --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_main.py -v
```

## 🐳 Docker Development

```bash
# Build and run with Docker Compose
docker-compose -f docker-compose.dev.yml up --build

# Run in detached mode
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f app

# Stop services
docker-compose -f docker-compose.dev.yml down
```

## 🔧 Configuration

Configuration is managed through environment variables. See `.env.example` for all available options.

Key configurations:

- `APP_NAME`: Application name
- `APP_ENV`: Environment (development/staging/production)
- `DEBUG`: Debug mode (true/false)
- `DATABASE_URL`: Database connection string
- `SECRET_KEY`: JWT secret key

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- [Architecture Guide](docs/architecture.md) - System design and patterns
- [Development Guide](docs/development.md) - Development workflow and best practices
- [Deployment Guide](docs/deployment.md) - Deployment strategies
- [Security Guide](docs/security.md) - Security best practices
- [API Guide](docs/api.md) - API documentation
- [Tier Progression](docs/tiers.md) - Moving between tiers

## 🛠️ Development Commands

```bash
# Format code
uv run black app/ tests/
uv run isort app/ tests/

# Lint code
uv run ruff check app/ tests/
uv run mypy app/

# Create new migration
uv run alembic revision --autogenerate -m "Description"

# Apply migrations
uv run alembic upgrade head

# Rollback migration
uv run alembic downgrade -1

# Start development server
uv run python app/main.py

# Run interactive shell
uv run python -i scripts/shell.py
```

## 📦 Adding Dependencies

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

## 🚀 Deployment

See [docs/deployment.md](docs/deployment.md) for detailed deployment instructions for each tier.

### Quick Deploy to Staging

```bash
# Build Docker image
docker build -f docker/Dockerfile.prod -t fastapi-app:latest .

# Run container
docker run -p 8000:8000 --env-file .env fastapi-app:latest
```

## 🤝 Contributing

1. Create a feature branch (`git checkout -b feature/amazing-feature`)
2. Commit your changes (`git commit -m 'Add amazing feature'`)
3. Push to the branch (`git push origin feature/amazing-feature`)
4. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Team

- **Your Name** - Initial work - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- FastAPI for the amazing framework
- Astral for the uv package manager
- The Python community

---

**Note**: This is a Tier 1 development template. For production deployment with HIPAA compliance, follow the tier progression guide in [docs/tiers.md](docs/tiers.md).
