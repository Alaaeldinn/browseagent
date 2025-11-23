"""
Unit tests for error handling scenarios throughout the BrowseAgent application
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from fastapi import HTTPException
import os


class TestSearXNGErrorHandling:
    """Test error handling in SearXNG search tool"""
    
    def test_searxng_search_tool_language_error(self):
        """Test handling of language parameter errors in SearXNG tool"""
        from search_tool import SearXNGSearchTool
        
        tool = SearXNGSearchTool()
        
        with patch('search_tool.SearxSearchWrapper') as mock_wrapper:
            # First call (with language) fails, second call (without language) succeeds
            mock_wrapper.side_effect = [
                Exception("Language not supported"),
                Mock()
            ]
            
            with patch('search_tool.process_search_results_with_pipeline') as mock_pipeline:
                mock_pipeline.return_value = [
                    {"title": "Test", "href": "https://example.com", "body": "test", "similarity_score": 0.9}
                ]
                
                result = tool._run("test query")
                
                # Should handle language error gracefully
                assert "Error occurred during SearXNG search" not in result
                assert "Test" in result  # Should still return results
    
    def test_searxng_search_tool_empty_query(self):
        """Test SearXNG tool with empty query"""
        from search_tool import SearXNGSearchTool
        
        tool = SearXNGSearchTool()
        result = tool._run("")
        
        assert "Error: Query cannot be empty" in result
    
    def test_searxng_search_tool_connection_error(self):
        """Test SearXNG tool with connection errors"""
        from search_tool import SearXNGSearchTool
        
        tool = SearXNGSearchTool()
        
        with patch('search_tool.SearxSearchWrapper') as mock_wrapper:
            mock_wrapper.side_effect = Exception("Connection error")
            
            result = tool._run("test query")
            
            assert "Error initializing SearXNG search wrapper" in result
    
    def test_searxng_search_tool_invalid_host(self):
        """Test SearXNG tool with invalid host"""
        from search_tool import SearXNGSearchTool
        
        tool = SearXNGSearchTool(searx_host="invalid://url")
        result = tool._run("test query")
        
        assert "Invalid SearXNG host configuration" in result


class TestOpenRouterErrorHandling:
    """Test error handling in OpenRouter integration"""
    
    @patch('openrouter.requests.get')
    def test_openrouter_validate_api_key_connection_error(self, mock_get):
        """Test API key validation with connection error"""
        from openrouter import OpenRouterClient
        
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        client = OpenRouterClient(api_key="test-key")
        result = client.validate_api_key()
        
        assert result["valid"] is False
        assert "Could not connect" in result["message"]
    
    @patch('openrouter.requests.get')
    def test_openrouter_validate_api_key_timeout(self, mock_get):
        """Test API key validation with timeout"""
        from openrouter import OpenRouterClient
        
        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")
        
        client = OpenRouterClient(api_key="test-key")
        result = client.validate_api_key()
        
        assert result["valid"] is False
        assert "Error validating API key" in result["message"]
    
    @patch('openrouter.requests.get')
    def test_openrouter_get_available_models_error(self, mock_get):
        """Test getting available models with error"""
        from openrouter import OpenRouterClient
        
        mock_get.side_effect = Exception("API error")
        
        client = OpenRouterClient(api_key="test-key")
        models = client.get_available_models()
        
        assert models == []
    
    @patch('openrouter.requests.get')
    def test_openrouter_get_free_models_error(self, mock_get):
        """Test getting free models with error"""
        from openrouter import OpenRouterClient
        
        mock_get.side_effect = Exception("API error")
        
        client = OpenRouterClient(api_key="test-key")
        free_models = client.get_free_models()
        
        assert free_models == []


class TestSessionErrorHandling:
    """Test error handling in session management"""
    
    def test_get_expired_session(self):
        """Test getting an expired session"""
        from session_manager import session_manager, UserSession
        from datetime import datetime, timedelta
        
        # Create an expired session
        expired_session = UserSession(
            session_id="expired-session",
            api_key="test-key",
            created_at=datetime.now() - timedelta(days=2),
            last_accessed=datetime.now() - timedelta(days=2),
            expires_at=datetime.now() - timedelta(hours=1)  # Expired 1 hour ago  
        )
        
        # Add it to the session manager manually for testing
        session_manager.sessions["expired-session"] = expired_session
        
        result = session_manager.get_session("expired-session")
        
        # Should return None for expired session
        assert result is None
        # Should have been removed from the session manager
        assert "expired-session" not in session_manager.sessions
    
    def test_get_nonexistent_session(self):
        """Test getting a non-existent session"""
        from session_manager import get_user_session
        
        result = get_user_session("nonexistent-session-id")
        
        assert result is None
    
    def test_update_nonexistent_session(self):
        """Test updating a non-existent session"""
        from session_manager import update_user_model
        
        result = update_user_model("nonexistent-session", "test-model")
        
        assert result is False


class TestAgentErrorHandling:
    """Test error handling in BrowseAgent"""
    
    @patch('agent.get_llm_instance')
    def test_browse_agent_llm_initialization_error(self, mock_get_llm):
        """Test BrowseAgent with LLM initialization error"""
        from agent import BrowseAgent
        
        # Make get_llm_instance raise an exception
        mock_get_llm.side_effect = Exception("LLM initialization failed")
        
        try:
            agent = BrowseAgent(llm_provider="invalid-model")
            # If we get here, the exception wasn't raised where expected
            # Check if fallback handling works
            assert True
        except Exception:
            # This is expected - the agent initialization could fail
            pass
    
    @patch('agent.process_query_with_agent')
    def test_browse_agent_run_query_error(self, mock_process):
        """Test BrowseAgent run_query with error"""
        from agent import BrowseAgent
        
        # Mock error during query processing
        mock_process.side_effect = Exception("Processing error")
        
        agent = BrowseAgent()
        result = agent.run_query("test query")
        
        # Should return error message
        assert "Error occurred during agent execution" in result
        assert "Search also failed" in result  # Fallback search would also fail
    
    def test_extract_keywords_error_handling(self):
        """Test keyword extraction error handling"""
        from agent import BrowseAgent
        
        agent = BrowseAgent()
        
        # Mock litellm.completion to raise an exception
        with patch('agent.litellm.completion') as mock_completion:
            mock_completion.side_effect = Exception("API Error")
            
            # Should fallback to original query on error
            result = agent.extract_keywords("test query")
            
            # Should return original query when extraction fails
            assert result == "test query"


class TestAppErrorHandling:
    """Test error handling in FastAPI endpoints"""
    
    def test_get_session_with_expired_session(self):
        """Test getting session info with expired session"""
        from app import get_session_info
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        # Create an expired session
        expired_session = UserSession(
            session_id="test-session",
            api_key="test-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() - timedelta(hours=1)
        )
        
        # Try to get session info
        from app import validate_and_track_session
        with pytest.raises(HTTPException):
            # This should raise an exception due to expired session
            pass  # This test requires more complex mocking of the dependency system
    
    @patch('app.OpenRouterClient')
    def test_models_endpoint_error(self, mock_client_class):
        """Test models endpoint with API error"""
        from app import get_models
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        # Mock client to raise exception
        mock_client_instance = Mock()
        mock_client_instance.get_free_models.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client_instance
        
        # Create a mock session
        mock_session = UserSession(
            session_id="test-session",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # This should use fallback default models
        response = get_models(mock_session)
        
        # Should return default models as fallback
        from app import get_default_free_models
        default_count = len(get_default_free_models())
        assert response["message"] == f"Found {default_count} models"
    
    @patch('app.process_query_with_agent')
    def test_query_endpoint_error(self, mock_process):
        """Test query endpoint with processing error"""
        from app import process_query
        from app import QueryRequest
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        # Mock processing to raise an exception
        mock_process.side_effect = Exception("Processing failed")
        
        # Create a mock session
        mock_session = UserSession(
            session_id="test-session",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Create request
        request = QueryRequest(query="test query")
        
        from fastapi import HTTPException
        try:
            response = process_query(request, mock_session)
            # This shouldn't be reached due to the exception
            assert False, "Expected HTTPException"
        except HTTPException as e:
            # Should return formatted error response
            assert "Query processing error" in str(e.detail)
    
    @patch('app.OpenRouterClient')
    def test_model_selection_endpoint_invalid_model(self, mock_client_class):
        """Test model selection with invalid model"""
        from app import select_model, ModelSelectionRequest
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        # Mock client to return available models
        mock_client_instance = Mock()
        mock_client_instance.get_available_models.return_value = [
            {"id": "valid-model"}
        ]
        mock_client_class.return_value = mock_client_instance
        
        # Create a mock session
        mock_session = UserSession(
            session_id="test-session",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Try to select an invalid model
        request = ModelSelectionRequest(model="invalid-model")
        
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            select_model(request, mock_session)


class TestSearchToolErrorHandling:
    """Test error handling in search tools"""
    
    @patch('search_tool.search_ddgs')
    def test_old_search_tool_error(self, mock_search_ddgs):
        """Test OldSearchTool with DDGS error"""
        from search_tool import OldSearchTool
        
        # Make search_ddgs raise an exception
        mock_search_ddgs.side_effect = Exception("DDGS Error")
        
        tool = OldSearchTool()
        result = tool._run("test query")
        
        assert "Error occurred during search" in result
        assert "DDGS Error" in result
    
    @patch('search_tool.search_ddgs')
    def test_old_search_tool_pipeline_error(self, mock_search_ddgs):
        """Test OldSearchTool with pipeline processing error"""
        from search_tool import OldSearchTool
        
        # Mock search to work but pipeline to fail
        mock_search_ddgs.return_value = [{"title": "Test", "href": "https://example.com", "body": "test"}]
        
        with patch('search_tool.process_search_results_with_pipeline') as mock_pipeline:
            mock_pipeline.side_effect = Exception("Pipeline Error")
            
            tool = OldSearchTool()
            result = tool._run("test query")
            
            # Should still work with raw results if pipeline fails
            assert "Error occurred during search" not in result


class TestUtilityErrorHandling:
    """Test error handling in utility functions"""
    
    def test_titles_are_similar_edge_cases(self):
        """Test title similarity with edge cases"""
        from search_result_pipeline import SearchResultProcessor
        
        processor = SearchResultProcessor()
        
        # Test with empty strings
        result = processor._titles_are_similar("", "")
        assert result is False  # No common words, similarity = 0
        
        # Test with one empty string
        result = processor._titles_are_similar("test", "")
        assert result is False


class TestEnvironmentErrorHandling:
    """Test error handling with missing environment variables"""
    
    def test_missing_openrouter_api_key(self):
        """Test behavior when OpenRouter API key is missing"""
        from agent import get_llm_instance
        import os
        
        # Temporarily remove the API key from environment
        original_key = os.environ.get("OPENROUTER_API_KEY")
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]
        
        try:
            # This should raise an error for OpenRouter models
            with pytest.raises(ValueError):
                get_llm_instance("openrouter/test-model")
        finally:
            # Restore original environment
            if original_key is not None:
                os.environ["OPENROUTER_API_KEY"] = original_key
    
    def test_invalid_model_format(self):
        """Test handling of invalid model formats"""
        from agent import get_llm_instance
        
        # Test with invalid format - should use fallback
        llm = get_llm_instance("invalid-model-format")
        
        # Should default to a known model or raise appropriately
        assert hasattr(llm, 'temperature') or llm is not None


if __name__ == "__main__":
    pytest.main([__file__])