"""
[IMPLEMENTED] Unit tests for system health check endpoint and core configuration.
"""
from fastapi.testclient import TestClient

from backend.config import settings
from backend.main import app

client = TestClient(app)


def test_root_endpoint():
    """Verify that root endpoint responds with basic metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == settings.APP_NAME
    assert data["version"] == settings.VERSION
    assert "docs" in data
    assert "health" in data


def test_system_health_endpoint():
    """Verify /api/v1/system/health returns correct structure and values."""
    response = client.get("/api/v1/system/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == settings.APP_NAME
    assert data["version"] == settings.VERSION
    assert data["environment"] == settings.APP_ENV
    assert data["quantum_simulator"] == settings.QUANTUM_SIMULATOR_BACKEND
    assert "timestamp" in data
