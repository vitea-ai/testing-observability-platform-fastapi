"""
Unit tests for main application.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings


def test_app_creation():
    """Test that app is created successfully."""
    assert app is not None
    assert app.title == settings.name
    assert app.version == settings.version


def test_health_endpoint(client: TestClient):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["tier"] == settings.deployment_tier
    assert data["version"] == settings.version
    assert "features" in data
    assert "timestamp" in data


def test_ready_endpoint(client: TestClient):
    """Test readiness check endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    
    data = response.json()
    assert data["ready"] is True
    assert "checks" in data
    assert data["checks"]["app"] is True


def test_root_endpoint(client: TestClient):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "message" in data
    assert data["version"] == settings.version
    assert data["tier"] == settings.deployment_tier
    assert data["api"] == settings.api_v1_prefix


def test_404_handler(client: TestClient):
    """Test 404 error handler."""
    response = client.get("/nonexistent")
    assert response.status_code == 404
    
    data = response.json()
    assert data["error"] == "Not Found"
    assert "/nonexistent" in data["message"]


def test_cors_headers(client: TestClient):
    """Test CORS headers are present."""
    response = client.options("/health")
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers


@pytest.mark.parametrize("tier,docs_enabled", [
    ("development", True),
    ("integration", True),
    ("staging", True),
    ("production", False),
])
def test_docs_availability(tier: str, docs_enabled: bool):
    """Test documentation availability based on tier."""
    original_tier = settings.deployment_tier
    settings.deployment_tier = tier
    
    try:
        client = TestClient(app)
        
        if docs_enabled:
            response = client.get("/docs")
            # In test mode, docs might redirect
            assert response.status_code in [200, 307]
        else:
            response = client.get("/docs")
            assert response.status_code == 404
    finally:
        settings.deployment_tier = original_tier
