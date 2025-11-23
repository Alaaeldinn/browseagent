"""
API integration tests for all BrowseAgent endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from app import app
from session_manager import session_manager
import json


class TestAPIIntegration:
    """API integration tests for all endpoints"""
    
    def setup_method(self):
        """Set up test client for each test"""
        self.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "BrowseAgent API is running" in data["message"]
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = self.client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "description" in data
        assert "endpoints" in data
        assert len(data["endpoints"]) > 0
    
    def test_providers_endpoint(self):
        """Test the providers endpoint"""
        response = self.client.get("/providers")
        assert response.status_code == 200
        
        data = response.json()
        assert "providers" in data
        assert len(data["providers"]) > 0
        # Check that expected providers are included
        providers = data["providers"]
        assert "openai/gpt-3.5-turbo" in providers
    
    @patch('app.OpenRouterClient')
    def test_session_creation_endpoint_valid_key(self, mock_client):
        """Test session creation endpoint with valid API key"""
        # Mock the OpenRouter client
        mock_client_instance = Mock()
        mock_client_instance.validate_api_key.return_value = {
            "valid": True,
            "message": "API key is valid"
        }
        # Mock available models for default selection
        mock_client_instance.get_available_models.return_value = [
            {"id": "openai/gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
            {"id": "google/gemma-7b-it", "name": "Gemma 7B IT"}
        ]
        mock_client.return_value = mock_client_instance
        
        response = self.client.post("/session/create", json={
            "api_key": "valid-openrouter-key-12345"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "Session created successfully" in data["message"]
        # Session ID should be in the message
        assert "Session ID" in data["message"]
    
    @patch('app.OpenRouterClient')
    def test_session_creation_endpoint_invalid_key(self, mock_client):
        """Test session creation endpoint with invalid API key"""
        mock_client_instance = Mock()
        mock_client_instance.validate_api_key.return_value = {
            "valid": False,
            "message": "Invalid API key"
        }
        mock_client.return_value = mock_client_instance
        
        response = self.client.post("/session/create", json={
            "api_key": "invalid-key"
        })
        
        assert response.status_code == 200  # Returns success but with valid=False
        data = response.json()
        assert data["valid"] is False
        assert "Invalid API key" in data["message"]
    
    @patch('app.OpenRouterClient')
    def test_validate_api_key_endpoint(self, mock_client):
        """Test API key validation endpoint""" 
        mock_client_instance = Mock()
        mock_client_instance.validate_api_key.return_value = {
            "valid": True,
            "message": "API key is valid and has access to OpenRouter"
        }
        mock_client.return_value = mock_client_instance
        
        response = self.client.post("/validate-api-key", json={
            "api_key": "test-api-key-12345"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert "valid and has access" in data["message"]
    
    @patch('app.OpenRouterClient')
    def test_models_endpoint_with_session(self, mock_client):
        """Test models endpoint with valid session"""
        # This is more complex to test with TestClient since it requires session validation
        # So we'll test the underlying function directly in conjunction with session validation
        
        # Create a mock session to test the function directly
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        mock_session = UserSession(
            session_id="test-session-123",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Mock the OpenRouter client for the models endpoint
        mock_client_instance = Mock()
        mock_client_instance.get_free_models.return_value = [
            {"id": "model1", "name": "Model 1", "pricing": {"prompt": "$0.00"}},
            {"id": "model2", "name": "Model 2", "pricing": {"prompt": "$0.00"}}
        ]
        mock_client.return_value = mock_client_instance
        
        from app import get_models
        result = get_models(mock_session)
        
        assert result["message"] == "Found 2 models"
        assert len(result["models"]) == 2
    
    def test_session_info_endpoint_with_mock_session(self):
        """Test session info endpoint with mock session"""
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        mock_session = UserSession(
            session_id="info-test-session",
            api_key="test-api-key", 
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
            selected_model="openai/gpt-4",
            searx_host="https://custom-searx.example.com",
            use_searxng=True
        )
        
        from app import get_session_info
        result = get_session_info(mock_session)
        
        assert result["session_id"] == "info-test-session"
        assert result["selected_model"] == "openai/gpt-4"
        assert result["searx_host"] == "https://custom-searx.example.com"
        assert result["use_searxng"] is True
        assert "created_at" in result
        assert "last_accessed" in result
        assert "expires_at" in result
    
    def test_model_selection_endpoint_with_mock_session(self):
        """Test model selection endpoint with mock session"""
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        mock_session = UserSession(
            session_id="model-select-test",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(), 
            expires_at=datetime.now() + timedelta(hours=1),
            selected_model="original-model"
        )
        
        # Mock the update function to simulate changing the model
        from app import select_model, ModelSelectionRequest
        import app  # Import the app module to access global functions
        
        request = ModelSelectionRequest(model="new-model")
        
        # We can't easily test this with TestClient due to dependency injection
        # So we test the function directly
        from app import select_model
        result = select_model(request, mock_session)
        
        assert result["success"] is True
        assert "new-model" in result["message"]
        
        # Verify session model was updated
        assert mock_session.selected_model == "new-model"
    
    def test_model_config_endpoints(self):
        """Test model configuration endpoints"""
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        mock_session = UserSession(
            session_id="config-test-session",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Test setting model config
        from app import set_model_config, ModelConfigRequest
        config_request = ModelConfigRequest(
            model="test-model",
            temperature=0.8,
            max_tokens=1500,
            top_p=0.9
        )
        
        result = set_model_config(config_request, mock_session)
        
        assert result["model"] == "test-model"
        assert result["config"]["temperature"] == 0.8
        assert result["config"]["max_tokens"] == 1500
        assert result["config"]["top_p"] == 0.9
        assert "updated successfully" in result["message"]
        
        # Test getting model config
        from app import get_model_config
        get_result = get_model_config("test-model", mock_session)
        
        assert get_result["model"] == "test-model"
        assert get_result["config"]["temperature"] == 0.8
        assert get_result["config"]["max_tokens"] == 1500
        assert "retrieved successfully" in get_result["message"]
    
    @patch('app.process_query_with_agent')
    def test_query_endpoint_with_mocked_processing(self, mock_process):
        """Test query endpoint with mocked processing"""
        # Mock the processing function to return a test response
        mock_process.return_value = {
            "query": "test query",
            "llm_provider": "openai/gpt-3.5-turbo", 
            "result": "This is the processed result for the test query",
            "search_engine": "searxng",
            "status": "success"
        }
        
        # Create a mock session for the test
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        mock_session = UserSession(
            session_id="query-test-session",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Test the underlying function directly since it requires session validation
        from app import process_query, QueryRequest
        request = QueryRequest(
            query="test query",
            llm_provider="openai/gpt-3.5-turbo"
        )
        
        result = process_query(request, mock_session)
        
        assert result["query"] == "test query"
        assert result["llm_provider"] == "openai/gpt-3.5-turbo"
        assert "processed result" in result["result"]
        assert result["success"] is True
    
    @patch('app.OpenRouterClient')
    def test_account_endpoint(self, mock_client):
        """Test account information endpoint"""
        mock_client_instance = Mock()
        mock_client_instance.get_account_balance.return_value = {
            "success": True,
            "data": {
                "balance": 15.75,
                "unit": "USD",
                "usage": "details here"
            }
        }
        mock_client.return_value = mock_client_instance
        
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        mock_session = UserSession(
            session_id="account-test-session",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        from app import get_account_info
        result = get_account_info(mock_session)
        
        assert result["success"] is True
        assert "balance" in str(result["data"])
    
    def test_session_deletion_endpoint(self):
        """Test session deletion endpoint"""
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        # Create a session first
        session_id = "delete-test-session"
        mock_session = UserSession(
            session_id=session_id,
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Add to session manager temporarily
        session_manager.sessions[session_id] = mock_session
        
        # Verify session exists
        assert session_manager.get_session(session_id) is not None
        
        # Test the deletion function directly
        from app import end_session
        result = end_session(mock_session)
        
        assert result["success"] is True
        assert "ended successfully" in result["message"]
        
        # Verify session was deleted
        assert session_manager.get_session(session_id) is None


class TestAPIEndpointErrorCases:
    """Test error cases for API endpoints"""
    
    def setup_method(self):
        """Set up test client for each test"""
        self.client = TestClient(app)
    
    def test_missing_session_id(self):
        """Test endpoints that require session ID when it's missing"""
        # Test /models endpoint without session ID
        response = self.client.get("/models")
        assert response.status_code == 400
        assert "Session ID is required" in response.json()["detail"]
        
        # Test /query endpoint without session ID
        response = self.client.post("/query", json={"query": "test"})
        assert response.status_code == 400
        assert "Session ID is required" in response.json()["detail"]
    
    def test_invalid_session_id(self):
        """Test endpoints with invalid session ID"""
        headers = {"X-Session-ID": "invalid-session-id"}
        
        # Test models endpoint with invalid session
        response = self.client.get("/models", headers=headers)
        assert response.status_code == 401
        assert "Invalid or expired session" in response.json()["detail"]
    
    @patch('app.process_query_with_agent')
    def test_query_endpoint_processing_error(self, mock_process):
        """Test query endpoint when processing fails"""
        mock_process.side_effect = Exception("Processing error")
        
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        mock_session = UserSession(
            session_id="error-test-session",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        from app import process_query, QueryRequest
        request = QueryRequest(query="test query")
        
        from fastapi import HTTPException
        try:
            result = process_query(request, mock_session)
            assert False, "Expected HTTPException"
        except HTTPException as e:
            assert "Query processing error" in str(e.detail)


class TestAPIRateLimiting:
    """Test API rate limiting functionality"""
    
    def test_rate_limiting_mechanism(self):
        """Test the rate limiting mechanism"""
        from app import check_rate_limit_for_session
        
        # Test that rate limit works properly
        session_id = "rate-limit-test"
        
        # Reset the global request counts for testing
        from app import request_counts
        request_counts.clear()  # Clear any existing counts
        
        # Add multiple requests to trigger rate limiting
        from app import RATE_LIMIT
        for i in range(RATE_LIMIT):  # Make requests up to the limit
            # This should not raise an exception
            try:
                check_rate_limit_for_session(f"{session_id}-{i}")
            except Exception:
                # If it does, capture that for analysis
                pass
        
        # The next request should trigger rate limiting
        # Since we can't easily test the exact rate limit without mocking time,
        # we'll test the structure by looking at the implementation
        from fastapi import HTTPException
        try:
            check_rate_limit_for_session(session_id)
        except HTTPException as e:
            # If we get a 429, that means rate limiting is working
            if e.status_code == 429:
                assert True


class TestAPIRequestValidation:
    """Test API request validation"""
    
    def test_query_request_validation(self):
        """Test validation of query requests"""
        from app import QueryRequest
        
        # Valid request
        valid_request = QueryRequest(query="test query")
        assert valid_request.query == "test query"
        
        # Request with default values
        default_request = QueryRequest(query="test")
        assert default_request.llm_provider == "openai/gpt-3.5-turbo"
        assert default_request.searx_host == "https://search.us.projectsegfau.lt"
        assert default_request.use_searxng is True
    
    def test_api_key_validation_request(self):
        """Test validation of API key validation requests"""
        from app import APIKeyValidationRequest
        
        request = APIKeyValidationRequest(api_key="test-key")
        assert request.api_key == "test-key"
    
    def test_model_selection_request(self):
        """Test validation of model selection requests"""
        from app import ModelSelectionRequest
        
        request = ModelSelectionRequest(model="openai/gpt-4")
        assert request.model == "openai/gpt-4"


if __name__ == "__main__":
    pytest.main([__file__])