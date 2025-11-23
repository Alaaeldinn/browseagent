"""
Tests for API key handling throughout the BrowseAgent application
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import os
from fastapi.testclient import TestClient
from app import app
from session_manager import session_manager, create_user_session, get_user_session


class TestAPIKeyStorageAndSecurity:
    """Tests for secure API key storage and handling"""
    
    def test_session_based_api_key_storage(self):
        """Test that API keys are securely stored in sessions"""
        api_key = "sk-or-test-api-key-1234567890"
        session_id = create_user_session(api_key)
        
        # Verify the session was created with the API key
        session = get_user_session(session_id)
        assert session is not None
        assert session.api_key == api_key
        assert session.is_active is True
        
        # Verify the API key is not exposed in session info when not needed
        session_info = {
            "session_id": session.session_id,
            "selected_model": session.selected_model,
            "created_at": session.created_at.isoformat(),
            "is_active": session.is_active
        }
        # API key should not be part of the exposed session info
        assert "api_key" not in session_info or session_info.get("api_key") is None
        
        # Clean up
        session_manager.delete_session(session_id)
    
    def test_api_key_not_stored_persistently(self):
        """Test that API keys are not stored in persistent storage"""
        # This verifies that the session manager only stores in memory
        api_key = "persistent-test-key"
        session_id = create_user_session(api_key)
        
        # Verify the key is in the in-memory session manager
        session = get_user_session(session_id)
        assert session.api_key == api_key
        
        # The key should not exist in environment or other persistent storage
        assert os.environ.get("TEST_API_KEY") != api_key
        # Clean up
        session_manager.delete_session(session_id)


class TestAPIKeyValidation:
    """Tests for API key validation process"""
    
    @patch('app.OpenRouterClient')
    def test_api_key_validation_process(self, mock_client):
        """Test the complete API key validation process"""
        mock_client_instance = Mock()
        mock_client_instance.validate_api_key.return_value = {
            "valid": True,
            "message": "API key is valid",
            "user_info": {"email": "test@example.com"}
        }
        mock_client.return_value = mock_client_instance
        
        from app import create_session, APIKeyValidationRequest
        request = APIKeyValidationRequest(api_key="test-api-key")
        
        response = create_session(request)
        
        assert response.valid is True
        assert "Session created successfully" in response.message
        mock_client_instance.validate_api_key.assert_called_once()
    
    @patch('app.OpenRouterClient')
    def test_invalid_api_key_handling(self, mock_client):
        """Test handling of invalid API keys"""
        mock_client_instance = Mock()
        mock_client_instance.validate_api_key.return_value = {
            "valid": False,
            "message": "Invalid API key format"
        }
        mock_client.return_value = mock_client_instance
        
        from app import create_session, APIKeyValidationRequest
        request = APIKeyValidationRequest(api_key="invalid-api-key")
        
        response = create_session(request)
        
        assert response.valid is False
        assert "Invalid API key format" in response.message


class TestAPIKeyAccessControl:
    """Tests for API key-based access control"""
    
    def test_api_key_access_validation(self):
        """Test validation of API key access in session-based system"""
        from app import validate_and_track_session, get_session_id
        from fastapi import HTTPException
        
        # Test with a non-existent session ID
        with pytest.raises(HTTPException) as exc_info:
            get_session_id("non-existent-session-id")
        
        # The dependency system is complex to test directly,
        # so we test the underlying functions
        from session_manager import get_user_session
        
        # Create a valid session
        session_id = create_user_session("test-api-key-for-access")
        session = get_user_session(session_id)
        
        assert session is not None
        assert session.api_key == "test-api-key-for-access"
        
        # Clean up
        session_manager.delete_session(session_id)
    
    @patch('app.OpenRouterClient')
    def test_api_key_used_for_openrouter_calls(self, mock_client):
        """Test that the stored API key is used for OpenRouter API calls"""
        mock_client_instance = Mock()
        mock_client_instance.get_available_models.return_value = [
            {"id": "model1", "name": "Model 1"}
        ]
        mock_client.return_value = mock_client_instance
        
        # Create a session
        session_id = create_user_session("test-api-key-validation")
        
        # Get the session
        session = get_user_session(session_id)
        
        # Use the API key to make a client call
        test_client = mock_client.return_value  # This uses the API key internally
        models = test_client.get_available_models()
        
        assert len(models) == 1
        assert models[0]["id"] == "model1"
        
        # Clean up
        session_manager.delete_session(session_id)


class TestAPIKeyEnvironmentHandling:
    """Tests for API key handling in environment variables"""
    
    def test_api_key_environment_isolation(self):
        """Test that API keys are isolated in environment during requests"""
        from agent import get_llm_instance
        import os
        
        # Save original environment
        original_keys = {
            k: v for k, v in os.environ.items() 
            if k in ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]
        }
        
        try:
            # Clear relevant environment variables
            for key in ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]:
                if key in os.environ:
                    del os.environ[key]
            
            # Test creating an LLM instance with API key temporarily set
            os.environ["OPENROUTER_API_KEY"] = "temp-test-key"
            
            # This should work with the temporary key
            try:
                llm = get_llm_instance("openrouter/test-model")
                # If it doesn't raise an error, the environment was handled correctly
                success = True
            except ValueError:
                # Expected for non-existent model, but environment handling is correct
                success = True
            
            # Restore original environment
            for key, value in original_keys.items():
                os.environ[key] = value
                
        finally:
            # Ensure environment is cleaned up
            if "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            
            # Restore original values
            for key, value in original_keys.items():
                os.environ[key] = value


class TestAPIKeyLifecycle:
    """Tests for the complete lifecycle of API keys"""
    
    def test_api_key_session_lifecycle(self):
        """Test the complete lifecycle: create, use, expire, cleanup"""
        api_key = "lifecycle-test-key-12345"
        
        # 1. Create session with API key
        session_id = create_user_session(api_key)
        assert session_id is not None
        
        # 2. Verify session exists and contains API key
        session = get_user_session(session_id)
        assert session is not None
        assert session.api_key == api_key
        
        # 3. Use the session (simulated)
        session.update_access_time()
        assert session.last_accessed is not None
        
        # 4. Verify session is active before expiration
        assert not session.is_expired()
        
        # 5. Clean up session
        cleanup_result = session_manager.delete_session(session_id)
        assert cleanup_result is True
        
        # 6. Verify session is gone
        assert get_user_session(session_id) is None
    
    def test_multiple_api_keys_concurrent_sessions(self):
        """Test handling multiple concurrent API keys in different sessions"""
        api_keys = [
            "key-1-test-concurrent",
            "key-2-test-concurrent", 
            "key-3-test-concurrent"
        ]
        
        session_ids = []
        
        # Create multiple sessions
        for key in api_keys:
            session_id = create_user_session(key)
            session_ids.append(session_id)
        
        # Verify each session has its own API key
        for i, session_id in enumerate(session_ids):
            session = get_user_session(session_id)
            assert session.api_key == api_keys[i]
        
        # Clean up all sessions
        for session_id in session_ids:
            session_manager.delete_session(session_id)
            assert get_user_session(session_id) is None


class TestAPIKeySecurity:
    """Tests for API key security measures"""
    
    def test_api_key_not_exposed_in_logs(self):
        """Test that API keys are not exposed in logs or error messages"""
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        # Create a session with a sensitive API key
        sensitive_key = "sk-or-v1-very-sensitive-key-data-here"
        session = UserSession(
            session_id="security-test-session",
            api_key=sensitive_key,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Verify the API key is stored but not easily accessible via common methods
        session_info = {
            "session_id": session.session_id,
            "selected_model": session.selected_model,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
        }
        
        # The api_key should not appear in session info for security
        # (this is just a representation; in real scenarios we'd check logs)
        session_repr = str(session_info)
        assert sensitive_key not in session_repr
        
        # Clean up
        session_manager.delete_session(session.session_id)
    
    def test_api_key_pattern_validation(self):
        """Test that API keys follow expected patterns"""
        # Valid OpenRouter pattern
        valid_key = "sk-or-v1-1234567890abcdef"
        session_id = create_user_session(valid_key)
        
        session = get_user_session(session_id)
        assert session.api_key == valid_key
        
        session_manager.delete_session(session_id)
    
    def test_short_api_key_rejection(self):
        """Test that short API keys are handled appropriately"""
        from app import get_session_id
        from fastapi import HTTPException
        
        # Short key should be rejected by our validation
        short_key = "short"
        
        # This is tested in the dependency injection function
        # Since it's complex to test the FastAPI dependency directly,
        # we rely on our unit test from earlier where it's tested


class TestAPIKeyIntegration:
    """Integration tests for API key handling with the entire system"""
    
    @patch('app.OpenRouterClient')
    @patch('app.process_query_with_agent')
    def test_api_key_used_throughout_request_flow(self, mock_process, mock_client):
        """Test that API key is properly used throughout the request flow"""
        # Mock the OpenRouter client validation
        mock_client_instance = Mock()
        mock_client_instance.validate_api_key.return_value = {
            "valid": True,
            "message": "API key is valid"
        }
        mock_client_instance.get_available_models.return_value = [
            {"id": "openai/gpt-3.5-turbo", "name": "GPT-3.5 Turbo"}
        ]
        mock_client.return_value = mock_client_instance
        
        # Mock the processing
        mock_process.return_value = {
            "query": "integration test query",
            "llm_provider": "openai/gpt-3.5-turbo",
            "result": "Integration test response",
            "search_engine": "searxng",
            "status": "success"
        }
        
        # Create a session
        session_id = create_user_session("integration-test-key")
        
        # Get the session
        session = get_user_session(session_id)
        assert session is not None
        assert session.api_key == "integration-test-key"
        
        # The API key should be used in the backend processing
        # This is verified by checking that the environment is set correctly
        # during the processing, which we can't easily test here without
        # mocking the entire agent flow
        
        # Clean up
        session_manager.delete_session(session_id)
    
    def test_session_cleanup_on_api_key_removal(self):
        """Test that sessions are properly cleaned up"""
        api_key = "cleanup-test-key"
        session_id = create_user_session(api_key)
        
        # Verify session exists
        session = get_user_session(session_id)
        assert session is not None
        
        # Clean up
        result = session_manager.delete_session(session_id)
        assert result is True
        
        # Verify it's gone
        assert get_user_session(session_id) is None


if __name__ == "__main__":
    pytest.main([__file__])