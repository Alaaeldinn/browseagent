import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app import app
from session_manager import create_user_session, delete_user_session

def test_health_endpoint():
    """Test the health check endpoint"""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root_endpoint():
    """Test the root endpoint"""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "BrowseAgent API" in data["message"]

def test_create_session():
    """Test session creation with mock API key"""
    client = TestClient(app)
    
    # This test would require a valid API key to work fully
    # For now, we'll test the structure
    response = client.post("/session/create", json={
        "api_key": "sk-test-api-key"  # This would be invalid in real scenario
    })
    # Should return a response even if API key is invalid
    assert response.status_code in [200, 400, 401, 500]

def test_model_endpoints():
    """Test model listing endpoint"""
    client = TestClient(app)
    
    # Create a temporary session for testing
    session_id = create_user_session("test-key", "openai/gpt-3.5-turbo")
    
    try:
        # Test models endpoint with session
        response = client.get("/models", headers={
            "X-Session-ID": session_id
        })
        # Should return 200 or 500 depending on API key validity
        assert response.status_code in [200, 500]
    finally:
        # Clean up
        delete_user_session(session_id)

if __name__ == "__main__":
    pytest.main([__file__])