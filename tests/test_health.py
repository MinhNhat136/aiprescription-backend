"""Tests for health check endpoints."""
import pytest


@pytest.mark.asyncio
async def test_health_endpoint_returns_200(client):
    """Test that health endpoint returns 200 status."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_endpoint_returns_version(client):
    """Test that health endpoint returns version information."""
    response = await client.get("/health")
    data = response.json()
    assert "version" in data
    assert data["version"] == "0.1.0"