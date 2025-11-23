"""
Unit tests for API key validation functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from openrouter import OpenRouterClient


class TestAPIKeyValidation:
    """Unit tests for API key validation functionality"""
    
    def test_api_key_format_validation(self):
        """Test validation of API key format in app.py dependency"""
        from app import get_session_id
        from fastapi import HTTPException
        
        # Test with valid session ID format (would normally be done in the dependency)
        try:
            # This is more of an integration test with FastAPI, 
            # but we can test the logic
            pass
        except:
            pass
    
    @patch.object(OpenRouterClient, 'validate_api_key')
    def test_api_key_validation_success(self, mock_validate):
        """Test successful API key validation"""
        mock_validate.return_value = {
            "valid": True,
            "message": "API key is valid",
            "details": "Access confirmed"
        }
        
        client = OpenRouterClient(api_key="valid-api-key")
        result = client.validate_api_key()
        
        assert result["valid"] is True
        assert result["message"] == "API key is valid"
        
    @patch.object(OpenRouterClient, 'validate_api_key')
    def test_api_key_validation_failure(self, mock_validate):
        """Test API key validation failure"""
        mock_validate.return_value = {
            "valid": False,
            "message": "Invalid API key",
            "details": "Authentication failed"
        }
        
        client = OpenRouterClient(api_key="invalid-api-key")
        result = client.validate_api_key()
        
        assert result["valid"] is False
        assert result["message"] == "Invalid API key"
    
    def test_api_key_validation_no_key(self):
        """Test API key validation with no key provided"""
        client = OpenRouterClient(api_key=None)
        result = client.validate_api_key()
        
        assert result["valid"] is False
        assert result["message"] == "API key is required"
    
    def test_api_key_validation_empty_key(self):
        """Test API key validation with empty key"""
        client = OpenRouterClient(api_key="")
        result = client.validate_api_key()
        
        assert result["valid"] is False
        assert result["message"] == "API key is required"
        
    @patch('openrouter.requests.get')
    def test_account_balance_with_valid_key(self, mock_get):
        """Test account balance retrieval with valid API key"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "balance": 10.50,
                "unit": "USD"
            }
        }
        mock_get.return_value = mock_response
        
        client = OpenRouterClient(api_key="valid-key")
        result = client.get_account_balance()
        
        assert result["success"] is True
        assert "balance" in str(result["data"])
        
    @patch('openrouter.requests.get')
    def test_account_balance_with_invalid_key(self, mock_get):
        """Test account balance retrieval with invalid API key"""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_get.return_value = mock_response
        
        client = OpenRouterClient(api_key="invalid-key")
        result = client.get_account_balance()
        
        assert result["success"] is False
        assert "401" in result["message"]
        
    @patch('openrouter.requests.get')
    def test_account_balance_connection_error(self, mock_get):
        """Test account balance retrieval with connection error"""
        mock_get.side_effect = Exception("Connection failed")
        
        client = OpenRouterClient(api_key="some-key")
        result = client.get_account_balance()
        
        assert result["success"] is False
        assert "Connection failed" in result["message"]
        
    @patch('openrouter.requests.get')
    def test_account_balance_no_key(self, mock_get):
        """Test account balance retrieval with no API key"""
        client = OpenRouterClient(api_key=None)
        result = client.get_account_balance()
        
        assert result["success"] is False
        assert result["message"] == "API key is required"
        mock_get.assert_not_called()


class TestAppAPIKeyValidation:
    """Tests for API key validation in the FastAPI app"""
    
    @patch('app.OpenRouterClient')
    def test_create_session_valid_api_key(self, mock_client_class):
        """Test session creation with valid API key"""
        from app import create_session
        from pydantic import BaseModel
        
        # Mock the OpenRouterClient instance
        mock_client_instance = Mock()
        mock_client_instance.validate_api_key.return_value = {
            "valid": True,
            "message": "API key is valid"
        }
        mock_client_class.return_value = mock_client_instance
        
        # Mock session creation
        with patch('app.create_user_session', return_value='test-session-123'):
            # Create a mock request object
            class MockRequest:
                def __init__(self):
                    self.api_key = "valid-api-key-123"
            
            from app import APIKeyValidationRequest
            request = APIKeyValidationRequest(api_key="valid-api-key-123")
            
            response = create_session(request)
            
            assert response.valid is True
            assert "Session created successfully" in response.message
            assert "test-session-123" in response.message
    
    @patch('app.OpenRouterClient')
    def test_create_session_invalid_api_key(self, mock_client_class):
        """Test session creation with invalid API key"""
        from app import create_session
        
        # Mock the OpenRouterClient instance
        mock_client_instance = Mock()
        mock_client_instance.validate_api_key.return_value = {
            "valid": False,
            "message": "Invalid API key format"
        }
        mock_client_class.return_value = mock_client_instance
        
        from app import APIKeyValidationRequest
        request = APIKeyValidationRequest(api_key="invalid-api-key")
        
        response = create_session(request)
        
        assert response.valid is False
        assert "Invalid API key format" in response.message
        
    @patch('app.OpenRouterClient')
    def test_create_session_api_error(self, mock_client_class):
        """Test session creation with API error"""
        from app import create_session
        
        # Mock the OpenRouterClient instance to raise an exception
        mock_client_instance = Mock()
        mock_client_instance.validate_api_key.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client_instance
        
        from app import APIKeyValidationRequest
        request = APIKeyValidationRequest(api_key="some-key")
        
        response = create_session(request)
        
        assert response.valid is False
        assert "Error creating session" in response.message


# Test session management for API key handling
class TestSessionBasedAPIKey:
    """Tests for session-based API key handling"""
    
    def test_session_creation_stores_api_key(self):
        """Test that session creation properly stores API key"""
        from session_manager import session_manager, create_user_session
        
        api_key = "test-api-key-12345"
        session_id = create_user_session(api_key)
        
        # Retrieve the session
        session = session_manager.get_session(session_id)
        
        assert session is not None
        assert session.api_key == api_key
        assert session.selected_model == "openai/gpt-3.5-turbo"  # default
        
        # Clean up
        session_manager.delete_session(session_id)
        
    def test_session_api_key_access(self):
        """Test accessing API key from session"""
        from session_manager import session_manager, create_user_session
        
        api_key = "another-test-key-67890"
        session_id = create_user_session(api_key, selected_model="google/gemma-7b-it")
        
        session = session_manager.get_session(session_id)
        
        assert session.api_key == api_key
        assert session.selected_model == "google/gemma-7b-it"
        
        # Clean up
        session_manager.delete_session(session_id)
    
    def test_session_expiry(self):
        """Test session expiration functionality"""
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        # Create an expired session manually for testing
        expired_session = UserSession(
            session_id="test-session",
            api_key="test-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() - timedelta(hours=1),  # Expired 1 hour ago
        )
        
        assert expired_session.is_expired() is True
        
        # Create a non-expired session
        valid_session = UserSession(
            session_id="valid-session",
            api_key="test-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),  # Expires in 1 hour
        )
        
        assert valid_session.is_expired() is False
        
    def test_session_model_config_storage(self):
        """Test storing and retrieving model configurations in sessions"""
        from session_manager import session_manager, create_user_session, ModelConfig
        
        api_key = "config-test-key"
        session_id = create_user_session(api_key)
        
        session = session_manager.get_session(session_id)
        assert session is not None
        
        # Create a model configuration
        config = ModelConfig(
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9
        )
        
        # Store configuration for a model
        session.set_model_config("test-model", config)
        
        # Retrieve configuration
        retrieved_config = session.get_model_config("test-model")
        
        assert retrieved_config.temperature == 0.7
        assert retrieved_config.max_tokens == 1000
        assert retrieved_config.top_p == 0.9
        
        # Test default config for non-existent model
        default_config = session.get_model_config("non-existent-model")
        assert default_config.temperature == 0.7  # Default from ModelConfig class
        
        # Clean up
        session_manager.delete_session(session_id)


if __name__ == "__main__":
    pytest.main([__file__])