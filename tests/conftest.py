"""
Test configuration and fixtures.
"""

import pytest
import pytest_asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import WebSocket
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import asyncio
from celery import Celery
from celery.result import AsyncResult
import fakeredis

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


@pytest.fixture
def mock_celery_app():
    """Create a mock Celery application for testing."""
    mock_app = MagicMock(spec=Celery)
    mock_app.send_task = MagicMock()
    mock_app.control = MagicMock()
    mock_app.control.inspect = MagicMock()
    mock_app.control.revoke = MagicMock()
    mock_app.control.purge = MagicMock()
    
    # Mock inspect methods
    mock_inspect = MagicMock()
    mock_inspect.active = MagicMock(return_value={})
    mock_inspect.scheduled = MagicMock(return_value={})
    mock_inspect.reserved = MagicMock(return_value={})
    mock_inspect.registered = MagicMock(return_value={"worker1": ["task1", "task2"]})
    mock_inspect.stats = MagicMock(return_value={"worker1": {}})
    mock_app.control.inspect.return_value = mock_inspect
    
    return mock_app


@pytest.fixture
def mock_celery_task():
    """Create a mock Celery task result."""
    mock_task = MagicMock(spec=AsyncResult)
    mock_task.id = "test-task-id-123"
    mock_task.status = "PENDING"
    mock_task.info = None
    mock_task.result = None
    mock_task.update_state = MagicMock()
    mock_task.revoke = MagicMock()
    
    # Create a mock delay method that returns the task
    mock_delay = MagicMock(return_value=mock_task)
    
    return mock_task, mock_delay


@pytest.fixture
def mock_redis():
    """Create a mock Redis client using fakeredis."""
    return fakeredis.FakeRedis()


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    websocket = MagicMock(spec=WebSocket)
    websocket.accept = AsyncMock()
    websocket.send_json = AsyncMock()
    websocket.receive_json = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client for external API calls."""
    mock_client = AsyncMock()
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = AsyncMock(return_value={
        "evaluators": [
            {"id": "quality_evaluator", "name": "Quality Evaluator"},
            {"id": "factuality_evaluator", "name": "Factuality Evaluator"}
        ]
    })
    mock_response.text = "Mock response"
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()
    
    return mock_client


@pytest.fixture
def sample_experiment_data():
    """Sample experiment data for testing."""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "Test Experiment",
        "description": "Test experiment for unit tests",
        "dataset_id": "660e8400-e29b-41d4-a716-446655440001",
        "agent_config": {
            "model": "gpt-4",
            "provider": "openai",
            "temperature": 0.7,
            "max_tokens": 100
        },
        "status": "completed",
        "progress": 1.0,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T01:00:00"
    }


@pytest.fixture
def sample_test_results():
    """Sample test results for evaluation."""
    return [
        {
            "test_id": "test-1",
            "input": "What is 2+2?",
            "expected_output": "4",
            "actual_output": "The answer is 4",
            "context": [],
            "test_case_type": "single_turn",
            "metadata": {}
        },
        {
            "test_id": "test-2",
            "input": "What is the capital of France?",
            "expected_output": "Paris",
            "actual_output": "Paris is the capital of France",
            "context": ["Geography facts"],
            "test_case_type": "single_turn",
            "metadata": {"category": "geography"}
        }
    ]


@pytest.fixture
def sample_evaluation_results():
    """Sample evaluation results from evaluator service."""
    return {
        "overall_score": 0.85,
        "results": [
            {
                "test_id": "test-1",
                "passed": True,
                "score": 0.9,
                "feedback": "Good answer"
            },
            {
                "test_id": "test-2",
                "passed": True,
                "score": 0.8,
                "feedback": "Correct"
            }
        ]
    }
