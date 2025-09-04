"""
Test configuration and fixtures.
"""

import pytest
from typing import Generator, AsyncGenerator
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import asyncio

from app.main import app
from app.core.config import settings
from app.db.session import get_db
from app.models.base import Base


# Override settings for testing
settings.testing = True
settings.database_url = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def test_db() -> Generator[Session, None, None]:
    """Create test database session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(test_db: Session) -> Generator[TestClient, None, None]:
    """Create test client with overridden database."""
    
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers() -> dict:
    """Get authentication headers for testing."""
    from app.api.v1.endpoints.users import create_access_token
    
    token = create_access_token(data={"sub": "testuser"})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_item() -> dict:
    """Sample item data for testing."""
    return {
        "name": "Test Item",
        "description": "This is a test item",
        "price": 99.99,
        "tax": 9.99,
        "tags": ["test", "sample"]
    }
