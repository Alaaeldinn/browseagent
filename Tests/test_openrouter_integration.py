"""
Unit tests for OpenRouter integration functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from openrouter import OpenRouterClient


class TestOpenRouterClient:
    """Unit tests for OpenRouterClient class"""
    
    def test_initialization_with_api_key(self):
        """Test OpenRouterClient initializes with provided API key"""
        api_key = "test-api-key-123"
        client = OpenRouterClient(api_key=api_key)
        
        assert client.api_key == api_key
        assert "Bearer test-api-key-123" in str(client.headers)
        assert client.base_url == "https://openrouter.ai/api/v1"
        
    def test_initialization_with_env_api_key(self):
        """Test OpenRouterClient initializes with environment API key"""
        import os
        original_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "env-api-key-456"
        
        try:
            client = OpenRouterClient()
            assert client.api_key == "env-api-key-456"
        finally:
            if original_key is not None:
                os.environ["OPENROUTER_API_KEY"] = original_key
            else:
                del os.environ["OPENROUTER_API_KEY"]
    
    def test_initialization_with_no_api_key(self):
        """Test OpenRouterClient handles missing API key gracefully"""
        import os
        original_key = os.environ.get("OPENROUTER_API_KEY")
        
        # Remove the env var if it exists
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]
        
        try:
            client = OpenRouterClient()
            assert client.api_key is None
            assert "Bearer None" in str(client.headers)  # This will occur but be handled properly
        finally:
            if original_key is not None:
                os.environ["OPENROUTER_API_KEY"] = original_key
    
    @patch('openrouter.requests.get')
    def test_validate_api_key_success_via_user_endpoint(self, mock_get):
        """Test API key validation succeeds via user endpoint"""
        # Setup mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {"email": "test@example.com", "models_used": []}
        }
        mock_get.return_value = mock_response
        
        client = OpenRouterClient(api_key="test-key")
        result = client.validate_api_key()
        
        assert result["valid"] is True
        assert result["message"] == "API key is valid"
        assert "test@example.com" in str(result["user_info"])
        mock_get.assert_called_with(
            "https://openrouter.ai/api/v1/user",
            headers=client.headers
        )
        
    @patch('openrouter.requests.get')
    def test_validate_api_key_success_via_models_endpoint(self, mock_get):
        """Test API key validation succeeds via models endpoint when user fails"""
        # First call (user endpoint) fails, second call (models) succeeds
        user_response = Mock()
        user_response.status_code = 403  # Not 401 or 200
        
        models_response = Mock()
        models_response.status_code = 200
        models_response.json.return_value = {"data": []}
        
        mock_get.side_effect = [user_response, models_response]
        
        client = OpenRouterClient(api_key="test-key")
        result = client.validate_api_key()
        
        assert result["valid"] is True
        assert result["message"] == "API key is valid"
        
    @patch('openrouter.requests.get')
    def test_validate_api_key_unauthorized(self, mock_get):
        """Test API key validation fails with 401 status"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        client = OpenRouterClient(api_key="test-key")
        result = client.validate_api_key()
        
        assert result["valid"] is False
        assert result["message"] == "Invalid API key"
        
    @patch('openrouter.requests.get')
    def test_validate_api_key_rate_limited(self, mock_get):
        """Test API key validation handles rate limiting"""
        mock_response = Mock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response
        
        client = OpenRouterClient(api_key="test-key")
        result = client.validate_api_key()
        
        assert result["valid"] is True  # Key is valid, just rate limited
        assert "rate limited" in result["message"]
        
    @patch('openrouter.requests.get')
    def test_validate_api_key_no_key_provided(self, mock_get):
        """Test API key validation with no key provided"""
        client = OpenRouterClient(api_key=None)
        result = client.validate_api_key()
        
        assert result["valid"] is False
        assert result["message"] == "API key is required"
        mock_get.assert_not_called()
        
    @patch('openrouter.requests.get')
    def test_validate_api_key_connection_error(self, mock_get):
        """Test API key validation handles connection errors"""
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        client = OpenRouterClient(api_key="test-key")
        result = client.validate_api_key()
        
        assert result["valid"] is False
        assert "Could not connect" in result["message"]
        
    @patch('openrouter.requests.get')
    def test_validate_api_key_general_exception(self, mock_get):
        """Test API key validation handles general exceptions"""
        mock_get.side_effect = Exception("Unexpected error")
        
        client = OpenRouterClient(api_key="test-key")
        result = client.validate_api_key()
        
        assert result["valid"] is False
        assert "Error validating API key" in result["message"]
        
    @patch('openrouter.requests.get')
    def test_get_available_models_success(self, mock_get):
        """Test getting available models succeeds"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "model1", "name": "Test Model 1"},
                {"id": "model2", "name": "Test Model 2"}
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        client = OpenRouterClient(api_key="test-key")
        models = client.get_available_models()
        
        assert len(models) == 2
        assert models[0]["id"] == "model1"
        assert models[1]["id"] == "model2"
        mock_get.assert_called_with(
            "https://openrouter.ai/api/v1/models",
            headers=client.headers
        )
        
    @patch('openrouter.requests.get')
    def test_get_available_models_error(self, mock_get):
        """Test getting available models handles errors"""
        mock_get.side_effect = Exception("API error")
        
        client = OpenRouterClient(api_key="test-key")
        models = client.get_available_models()
        
        assert models == []
        
    @patch.object(OpenRouterClient, 'get_available_models')
    def test_get_free_models_success(self, mock_get_available):
        """Test getting free models works correctly"""
        # Mock available models with different pricing
        mock_models = [
            {
                "id": "free-model",
                "name": "Free Model",
                "pricing": {"prompt": "$0.00"}
            },
            {
                "id": "paid-model",
                "name": "Paid Model",
                "pricing": {"prompt": "$0.01"}
            },
            {
                "id": "another-free",
                "name": "Another Free Model",
                "pricing": {"prompt": "$0.00000"}
            }
        ]
        mock_get_available.return_value = mock_models
        
        client = OpenRouterClient(api_key="test-key")
        free_models = client.get_free_models()
        
        assert len(free_models) == 2
        free_ids = [model["id"] for model in free_models]
        assert "free-model" in free_ids
        assert "another-free" in free_ids
        
    @patch.object(OpenRouterClient, 'get_available_models')
    def test_get_free_models_error(self, mock_get_available):
        """Test getting free models handles errors"""
        mock_get_available.side_effect = Exception("API error")
        
        client = OpenRouterClient(api_key="test-key")
        free_models = client.get_free_models()
        
        assert free_models == []
        
    @patch.object(OpenRouterClient, 'get_available_models')
    def test_get_model_pricing_info_success(self, mock_get_available):
        """Test getting pricing info for specific model"""
        mock_models = [
            {
                "id": "test-model",
                "name": "Test Model",
                "pricing": {"prompt": "$0.00", "completion": "$0.00"},
                "description": "A test model",
                "context_length": 4096
            },
            {
                "id": "other-model",
                "name": "Other Model",
                "pricing": {"prompt": "$0.01", "completion": "$0.02"},
                "description": "Another model",
                "context_length": 2048
            }
        ]
        mock_get_available.return_value = mock_models
        
        client = OpenRouterClient(api_key="test-key")
        pricing_info = client.get_model_pricing_info("test-model")
        
        assert pricing_info is not None
        assert pricing_info["id"] == "test-model"
        assert pricing_info["name"] == "Test Model"
        assert pricing_info["pricing"]["prompt"] == "$0.00"
        
    @patch.object(OpenRouterClient, 'get_available_models')
    def test_get_model_pricing_info_not_found(self, mock_get_available):
        """Test getting pricing info for non-existent model"""
        mock_models = [
            {
                "id": "existing-model",
                "name": "Existing Model",
                "pricing": {"prompt": "$0.00"},
                "description": "An existing model",
                "context_length": 4096
            }
        ]
        mock_get_available.return_value = mock_models
        
        client = OpenRouterClient(api_key="test-key")
        pricing_info = client.get_model_pricing_info("non-existent-model")
        
        assert pricing_info is None
        
    @patch.object(OpenRouterClient, 'get_available_models')
    def test_get_model_pricing_info_error(self, mock_get_available):
        """Test getting pricing info handles errors"""
        mock_get_available.side_effect = Exception("API error")
        
        client = OpenRouterClient(api_key="test-key")
        pricing_info = client.get_model_pricing_info("test-model")
        
        assert pricing_info is None
        
    @patch('openrouter.requests.post')
    def test_test_model_access_success(self, mock_post):
        """Test model access testing works"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        client = OpenRouterClient(api_key="test-key")
        result = client.test_model_access("test-model")
        
        assert result is True
        mock_post.assert_called()
        call_args = mock_post.call_args
        assert call_args[1]['headers'] == client.headers
        assert "test-model" in str(call_args[1]['json'])
        
    @patch('openrouter.requests.post')
    def test_test_model_access_error_codes(self, mock_post):
        """Test model access testing handles different error codes"""
        # Test various status codes that indicate the model exists but has access issues
        for status_code in [400, 401, 402, 403]:
            with patch('openrouter.requests.post') as mock_post_inner:
                mock_response = Mock()
                mock_response.status_code = status_code
                mock_post_inner.return_value = mock_response
                
                client = OpenRouterClient(api_key="test-key")
                result = client.test_model_access("test-model")
                
                # All these codes mean the model exists but there are access/permission issues
                # The function should return True (model exists) for all of these
                assert result is True, f"Failed for status code {status_code}"
        
    @patch('openrouter.requests.post')
    def test_test_model_access_connection_error(self, mock_post):
        """Test model access testing handles connection errors"""
        mock_post.side_effect = Exception("Connection error")
        
        client = OpenRouterClient(api_key="test-key")
        result = client.test_model_access("test-model")
        
        assert result is False
        

# Test the default free models utility function
def test_get_default_free_models():
    """Test the get_default_free_models function"""
    from openrouter import get_default_free_models
    
    default_models = get_default_free_models()
    
    assert len(default_models) > 0
    assert isinstance(default_models, list)
    
    # Check that each model has required fields
    for model in default_models:
        assert "id" in model
        assert "name" in model
        assert "pricing" in model
        assert "context_length" in model


if __name__ == "__main__":
    pytest.main([__file__])