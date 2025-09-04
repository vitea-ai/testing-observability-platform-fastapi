
# Optimal FastAPI Starter Template: A Guide

This document outlines a recommended structure and set of best practices for creating a robust, scalable, and developer-friendly FastAPI starter template. The philosophy is to start simple, be explicit, and provide a clear path for growth.

## 1. Philosophy and Goals

*   **Simplicity First:** The template should be runnable out-of-the-box with a minimal "Hello World" example. Avoid unnecessary complexity and "magic."
*   **Scalability by Design:** The structure should naturally evolve from a simple service to a complex application without requiring a major refactor.
*   **Developer Experience:** Prioritize clear code, good tooling, and straightforward debugging.
*   **Flexibility:** The template should provide a solid foundation but not be overly prescriptive about every library and tool.

## 2. Project Initialization and Tooling

With recent updates, the Python tooling ecosystem has been massively simplified. We can now recommend a single, unified tool for the entire workflow.

### Recommended Tool: uv

[**uv**](https://github.com/astral-sh/uv) is an all-in-one project and package manager that is incredibly fast and robust. It is built by Astral, the same team behind the popular linter **Ruff**. `uv` now handles:

*   **Python Version Management:** It can discover, download, and use specific Python versions.
*   **Virtual Environments:** It creates and manages virtual environments for your projects.
*   **Dependency Management:** It installs and manages packages from `pyproject.toml`, replacing `pip` and `pip-tools`.
*   **Script Running:** It can run scripts and tasks defined in your `pyproject.toml`.

This means `uv` is the only tool you need to install to get a project running, creating a simple and consistent developer experience.

### Initialization Steps

1.  **Install `uv`:**
    Follow the official instructions to install `uv` on your system:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Set Up the Project:**
    Create your project directory. `uv` will handle the rest.
    ```bash
    mkdir FastAPI_template
    cd FastAPI_template
    ```

3.  **Create `pyproject.toml`:**
    This file is the heart of your project. It defines your project's metadata, dependencies, and scripts. `uv` will use this file to create the correct environment.

    ```toml
    [project]
    name = "fastapi-template"
    version = "0.1.0"
    description = "A modern FastAPI template with best practices"
    requires-python = ">=3.11"
    dependencies = [
        "fastapi[standard]",
        "uvicorn[standard]",
        "pydantic-settings",
        "structlog",
        "sqlalchemy",
        "alembic",
        "asyncpg",
    ]

    [project.optional-dependencies]
    dev = [
        "pytest",
        "pytest-asyncio",
        "httpx",
        "ruff",
        "mypy",
        "black",
    ]

    [build-system]
    requires = ["hatchling"]
    build-backend = "hatchling.build"

    # Tool configurations
    [tool.ruff]
    line-length = 88
    target-version = "py311"

    [tool.ruff.lint]
    select = ["E", "W", "F", "I", "B", "C4", "UP"]
    ignore = ["E501", "B008"]

    [tool.pytest.ini_options]
    minversion = "8.0"
    testpaths = ["tests"]
    addopts = "--asyncio-mode=auto"
    ```

4.  **Install Dependencies:**
    This single command is all a new developer needs to run. `uv` will automatically:
    - Read the `requires-python` version from `pyproject.toml`
    - Download that Python version if it's not already installed
    - Create a `.venv` virtual environment using that Python version
    - Install all dependencies from `pyproject.toml` into the venv

    ```bash
    # Install all dependencies (creates venv automatically)
    uv sync
    
    # Or with dev dependencies
    uv sync --dev
    ```
    Ensure `.venv` is added to your `.gitignore` file.

5.  **Workflow:**
    You can run commands directly with `uv run`:
    ```bash
    # Run the dev server
    uv run uvicorn src.main:app --reload

    # Run tests
    uv run pytest

    # Lint your code
    uv run ruff check .
    
    # Format your code
    uv run black .
    uv run ruff format .
    ```
    
    For frequently used commands, you can create shell aliases or use a task runner like `just` or `make`.

## 3. Recommended Directory Structure (Feature-Based)

This structure organizes code by business domain/feature, which is more scalable than organizing by file type.

```
/FastAPI_template/
├── .env.example          # Example environment variables
├── .gitignore
├── pyproject.toml        # Project metadata and dependencies (managed by uv)
├── README.md
├── alembic/              # Database migrations
│   └── ...
├── src/
│   ├── __init__.py
│   ├── main.py             # FastAPI app instantiation and startup logic
│   │
│   ├── core/               # Application-wide concerns
│   │   ├── __init__.py
│   │   ├── config.py       # Configuration management (Pydantic settings)
│   │   └── db.py           # Database session management
│   │
│   ├── features/           # Main application code, organized by feature
│   │   ├── __init__.py
│   │   │
│   │   └── items/          # Example feature: "items"
│   │       ├── __init__.py
│   │       ├── router.py     # API endpoints for items
│   │       ├── schemas.py    # Pydantic schemas for items
│   │       ├── service.py    # Business logic for items
│   │       └── models.py     # SQLAlchemy models for items
│   │
│   └── health/             # A simple health check feature
│       ├── __init__.py
│       └── router.py
│
└── tests/
    ├── __init__.py
    ├── conftest.py         # Global test fixtures
    │
    └── features/
        └── test_items.py   # Tests for the "items" feature
```

### Key Advantages of this Structure

*   **High Cohesion, Low Coupling:** All the code for a single feature (e.g., `items`) is located in one place, making it easy to find and modify.
*   **Scalability:** Adding a new feature is as simple as adding a new directory under `src/features/`.
*   **Clear Ownership:** It's easier to assign ownership of features to different teams or developers.

## 4. Core Concepts and Best Practices

### Configuration Management (`core/config.py`)

Use `pydantic-settings` to manage configuration from environment variables. This provides type validation and a single source of truth for settings.

```python
# src/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    
    app_name: str = "My FastAPI App"
    database_url: Optional[str] = None
    debug: bool = False
    
    # For Tier 2+
    secret_key: Optional[str] = None
    
    # For Tier 3+
    redis_url: Optional[str] = None

settings = Settings()
```

### Database Session Management (`core/db.py`)

Use FastAPI's dependency injection system to manage database sessions. This ensures that a session is created for each request and closed afterward.

```python
# src/core/db.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator
from .config import settings

engine = create_async_engine(settings.DATABASE_URL)
AsyncSessionFactory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionFactory() as session:
        yield session
```

### Dependency Injection

Leverage FastAPI's `Depends` for everything from database sessions to authentication. This makes your code more modular, testable, and explicit.

### Testing

*   Write tests for your business logic (`service.py`) and your API endpoints (`router.py`).
*   Use an in-memory SQLite database or a separate test database for integration tests.
*   Use `httpx` and `AsyncClient` to make requests to your application in tests.

### Async All the Way

Use `async def` for all I/O-bound operations (database queries, API calls, etc.). This is crucial for performance. Use `asyncpg` for your PostgreSQL driver.

## 5. Example: A Runnable "Hello World"

This template should include a simple, runnable example to get developers started immediately.

**1. `.env.example`:**
```
APP_NAME="My FastAPI App"
DEBUG=true
# DATABASE_URL="postgresql+asyncpg://user:password@localhost/mydatabase"  # Uncomment for Tier 2+
# SECRET_KEY="your-secret-key-here"  # Uncomment for Tier 2+
# REDIS_URL="redis://localhost:6379"  # Uncomment for Tier 3+
```

**2. `src/main.py`:**
```python
from fastapi import FastAPI
from src.features.items import router as items_router
from src.health import router as health_router

app = FastAPI(title="Optimal FastAPI Template")

app.include_router(health_router.router, tags=["Health"])
app.include_router(items_router.router, prefix="/api/v1/items", tags=["Items"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Optimal FastAPI Template"}
```

**3. `src/health/router.py`:**
```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}
```

**4. `src/features/items/router.py` (A simple placeholder):**
```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def get_items():
    return [{"id": 1, "name": "Example Item"}]
```

By following this guide, you can create a FastAPI starter template that is simple, scalable, and a pleasure to work with.
