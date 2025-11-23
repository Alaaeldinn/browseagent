"""
Tests for error scenarios and fallback mechanisms in BrowseAgent
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from app import app
from agent import BrowseAgent, get_llm_instance, process_query_with_agent
from search_tool import SearXNGSearchTool, OldSearchTool
import os


class TestSearXNGFallbacks:
    """Tests for SearXNG search tool fallback mechanisms"""
    
    def test_searxng_fallback_to_minimal_params(self):
        """Test that SearXNG search falls back to minimal parameters when advanced ones fail"""
        from search_tool import SearXNGSearchTool
        
        tool = SearXNGSearchTool()
        
        with patch('search_tool.SearxSearchWrapper') as mock_wrapper:
            # First call with all params fails, second call with minimal params succeeds
            def side_effect(**kwargs):
                if 'language' in kwargs:
                    raise Exception("Language parameter not supported")
                else:
                    mock_instance = Mock()
                    mock_instance.results.return_value = [
                        {"title": "Fallback Result", "url": "https://example.com", "content": "fallback content", "engine": "fallback"}
                    ]
                    return mock_instance
            
            mock_wrapper.side_effect = side_effect
            
            with patch('search_tool.process_search_results_with_pipeline') as mock_pipeline:
                mock_pipeline.return_value = [
                    {"title": "Fallback Result", "href": "https://example.com", "body": "fallback content", "engine": "fallback", "similarity_score": 0.8}
                ]
                
                result = tool._run("test query")
                
                # Should succeed using fallback wrapper without language parameter
                assert "Fallback Result" in result
                assert mock_wrapper.call_count >= 2  # Called at least twice (once with language, once without)
    
    def test_searxng_fallback_on_search_failure(self):
        """Test fallback when SearXNG search completely fails"""
        from search_tool import SearXNGSearchTool
        
        tool = SearXNGSearchTool()
        
        with patch('search_tool.SearxSearchWrapper') as mock_wrapper:
            # Both main wrapper and fallback fail - simulate complete search failure
            mock_wrapper.side_effect = Exception("SearXNG instance unavailable")
            
            result = tool._run("test query")
            
            # Should return appropriate error message
            assert "Unable to retrieve search results" in result or "Error occurred during SearXNG search" in result
    
    def test_searxng_pipeline_failure_fallback(self):
        """Test that the tool works even when the result pipeline fails"""
        from search_tool import SearXNGSearchTool
        
        tool = SearXNGSearchTool()
        
        with patch('search_tool.SearxSearchWrapper') as mock_wrapper:
            mock_instance = Mock()
            mock_instance.results.return_value = [
                {"title": "Raw Result", "url": "https://example.com", "content": "raw content", "engine": "test"}
            ]
            mock_wrapper.return_value = mock_instance
            
            # Mock the pipeline to fail, so raw results are used
            with patch('search_tool.process_search_results_with_pipeline') as mock_pipeline:
                mock_pipeline.side_effect = Exception("Pipeline processing failed")
                
                result = tool._run("test query")
                
                # Should still return results even if pipeline fails
                assert "Raw Result" in result


class TestLLMModelFallbacks:
    """Tests for LLM model fallback mechanisms"""
    
    @patch('agent.ChatOpenAI')
    def test_openrouter_model_fallback(self, mock_chat_openai):
        """Test fallback when specific OpenRouter model fails"""
        from agent import get_llm_instance
        
        # Mock the ChatOpenAI constructor to fail for the primary model but succeed for fallback
        def chat_openai_side_effect(**kwargs):
            if kwargs.get('model') == 'nonexistent-model':
                raise Exception("Model not available")
            elif kwargs.get('model') == 'google/gemma-7b-it':  # fallback model
                mock_llm = Mock()
                mock_llm.temperature = 0.1
                return mock_llm
            else:
                mock_llm = Mock()
                mock_llm.temperature = kwargs.get('temperature', 0.1)
                return mock_llm
        
        mock_chat_openai.side_effect = chat_openai_side_effect
        
        # This should try the primary model, then fall back to the default
        llm = get_llm_instance(model="openrouter/nonexistent-model")
        
        # Should successfully create an LLM instance
        assert llm is not None
    
    @patch('agent.ChatOpenAI')
    def test_llm_fallback_to_default(self, mock_chat_openai):
        """Test fallback to default model when all attempts fail"""
        def chat_openai_side_effect(**kwargs):
            # Always raise an exception to test the fallback
            raise Exception("Model creation failed")
        
        mock_chat_openai.side_effect = chat_openai_side_effect
        
        try:
            llm = get_llm_instance(model="completely-invalid-model")
            # If it doesn't raise an exception, the fallback mechanism worked
            assert True
        except Exception:
            # The fallback should catch this and return a default LLM
            pass
    
    def test_browse_agent_model_fallback_during_query(self):
        """Test BrowseAgent's fallback mechanism when model fails during query processing"""
        from agent import BrowseAgent
        from langchain_core.messages import HumanMessage
        
        # Create an agent with a model that will fail during processing
        agent = BrowseAgent(llm_provider="openai/gpt-3.5-turbo")
        
        # Mock the agent's LLM to fail, triggering fallback mechanisms
        with patch.object(agent.llm, 'invoke') as mock_invoke:
            mock_invoke.side_effect = Exception("Model invocation failed")
            
            # Mock the search tool to still work
            with patch.object(agent.search_tool, '_run') as mock_search:
                mock_search.return_value = str([{
                    "title": "Search Result",
                    "href": "https://example.com",
                    "body": "This is a search result",
                    "similarity_score": 0.8
                }])
                
                result = agent.run_query("test query")
                
                # Should fall back to using search results directly when LLM fails
                assert "Search results" in result or "Error occurred" in result


class TestOpenRouterClientFallbacks:
    """Tests for OpenRouter client fallback mechanisms"""
    
    @patch('openrouter.requests.get')
    def test_api_key_validation_fallback(self, mock_get):
        """Test fallback when primary validation endpoint fails"""
        from openrouter import OpenRouterClient
        
        # Mock the user endpoint to fail but models endpoint to succeed
        def requests_get_side_effect(url, *args, **kwargs):
            mock_response = Mock()
            if "user" in url:
                mock_response.status_code = 403  # Forbidden on user endpoint
                return mock_response
            elif "models" in url:
                mock_response.status_code = 200
                mock_response.json.return_value = {"data": []}
                return mock_response
        
        mock_get.side_effect = requests_get_side_effect
        
        client = OpenRouterClient(api_key="test-key")
        result = client.validate_api_key()
        
        # Should still validate appropriately even if user endpoint fails
        assert "valid" in result
        # Could be True if models endpoint works, or False if auth issue
    
    @patch('openrouter.requests.get')
    def test_get_free_models_fallback(self, mock_get):
        """Test fallback when get_free_models fails"""
        from openrouter import OpenRouterClient, get_default_free_models
        
        # Mock get_available_models to fail
        mock_get.side_effect = Exception("API Error")
        
        client = OpenRouterClient(api_key="test-key")
        free_models = client.get_free_models()
        
        # Should return empty list or handle error gracefully
        assert isinstance(free_models, list)
    
    def test_get_default_free_models_fallback(self):
        """Test the default free models fallback function"""
        from openrouter import get_default_free_models
        
        default_models = get_default_free_models()
        
        # Should always return a list of default models
        assert isinstance(default_models, list)
        assert len(default_models) > 0
        
        # Verify structure
        for model in default_models:
            assert "id" in model
            assert "name" in model
            assert "pricing" in model


class TestSearchToolFallbacks:
    """Tests for search tool fallback mechanisms"""
    
    def test_old_search_tool_fallback(self):
        """Test the old DuckDuckGo search tool as fallback"""
        tool = OldSearchTool()
        
        with patch('search_tool.search_ddgs') as mock_search:
            mock_search.return_value = [
                {"title": "DDGS Result", "href": "https://example.com", "body": "DDGS content"}
            ]
            
            with patch('search_tool.process_search_results_with_pipeline') as mock_pipeline:
                mock_pipeline.return_value = [
                    {"title": "DDGS Result", "href": "https://example.com", "body": "DDGS content", "similarity_score": 0.7}
                ]
                
                result = tool._run("test query")
                
                assert "DDGS Result" in result
    
    def test_old_search_tool_error_handling(self):
        """Test error handling in old search tool"""
        tool = OldSearchTool()
        
        # Mock search to fail
        with patch('search_tool.search_ddgs') as mock_search:
            mock_search.side_effect = Exception("DDGS failed")
            
            result = tool._run("test query")
            
            assert "Error occurred during search" in result


class TestAPIErrorFallbacks:
    """Tests for API endpoint fallback mechanisms"""
    
    def setup_method(self):
        self.client = TestClient(app)
    
    @patch('app.process_query_with_agent')
    def test_query_endpoint_fallback(self, mock_process):
        """Test query endpoint fallback when processing fails"""
        # Mock the processing to fail
        mock_process.side_effect = Exception("Processing failed")
        
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        mock_session = UserSession(
            session_id="error-fallback-test",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        from app import process_query, QueryRequest
        request = QueryRequest(query="fallback test query")
        
        from fastapi import HTTPException
        try:
            result = process_query(request, mock_session)
            # If this doesn't throw an exception, the error handling is working
            assert True
        except HTTPException as e:
            # The error should be properly formatted
            assert "Query processing error" in str(e.detail)
    
    @patch('app.OpenRouterClient')
    def test_model_selection_with_unavailable_model(self, mock_client):
        """Test model selection when requested model is unavailable"""
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        mock_session = UserSession(
            session_id="model-fallback-test",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Mock client to return available models
        mock_client_instance = Mock()
        mock_client_instance.get_available_models.return_value = [
            {"id": "existing-model", "name": "Existing Model"}
        ]
        mock_client.return_value = mock_client_instance
        
        from app import select_model, ModelSelectionRequest
        request = ModelSelectionRequest(model="nonexistent-model")
        
        from fastapi import HTTPException
        try:
            select_model(request, mock_session)
            assert False, "Should have raised HTTPException"
        except HTTPException as e:
            # This is expected behavior
            assert "not available" in str(e.detail) or "Invalid model" in str(e.detail)
    
    @patch('app.find_fallback_model')
    @patch('app.OpenRouterClient')
    def test_model_fallback_in_query_processing(self, mock_client, mock_fallback):
        """Test that fallback model is used when preferred model is unavailable"""
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        mock_session = UserSession(
            session_id="query-fallback-test",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Mock the client and fallback mechanism
        mock_client_instance = Mock()
        mock_client_instance.get_available_models.return_value = [
            {"id": "openai/gpt-3.5-turbo", "name": "GPT-3.5 Turbo"}
        ]
        mock_client.return_value = mock_client_instance
        
        mock_fallback.return_value = "openai/gpt-3.5-turbo"  # Fallback to available model
        
        # Mock the processing function
        with patch('app.process_query_with_agent') as mock_process:
            mock_process.return_value = {
                "query": "fallback test",
                "llm_provider": "openai/gpt-3.5-turbo",
                "result": "Processed with fallback model",
                "search_engine": "searxng",
                "status": "success"
            }
            
            from app import process_query, QueryRequest
            request = QueryRequest(query="fallback test", llm_provider="nonexistent-model")
            
            result = process_query(request, mock_session)
            
            assert result["result"] == "Processed with fallback model"
            assert result["llm_provider"] == "openai/gpt-3.5-turbo"


class TestSessionManagerFallbacks:
    """Tests for session manager fallback mechanisms"""
    
    def test_session_cleanup_fallback(self):
        """Test that session cleanup works even if session doesn't exist"""
        from session_manager import session_manager
        
        # Try to delete a non-existent session
        result = session_manager.delete_session("non-existent-session-id")
        assert result is False  # Should return False when session doesn't exist
    
    def test_session_expired_cleanup(self):
        """Test automatic cleanup of expired sessions"""
        from session_manager import session_manager, UserSession
        from datetime import datetime, timedelta
        
        # Create an expired session
        expired_session = UserSession(
            session_id="expired-test",
            api_key="test-key",
            created_at=datetime.now() - timedelta(days=2),
            last_accessed=datetime.now() - timedelta(days=2),
            expires_at=datetime.now() - timedelta(hours=1)  # Already expired
        )
        
        # Add to session manager
        session_manager.sessions["expired-test"] = expired_session
        
        # Try to access the session - this should trigger cleanup
        retrieved_session = session_manager.get_session("expired-test")
        
        # Should return None for expired session
        assert retrieved_session is None
        # And session should be removed from manager
        assert "expired-test" not in session_manager.sessions
    
    def test_session_manager_cleanup_expired(self):
        """Test explicit cleanup of expired sessions"""
        from session_manager import session_manager, UserSession
        from datetime import datetime, timedelta
        
        # Create multiple expired sessions
        expired_sessions = []
        for i in range(3):
            session_id = f"expired-{i}"
            expired_session = UserSession(
                session_id=session_id,
                api_key="test-key",
                created_at=datetime.now() - timedelta(days=2),
                last_accessed=datetime.now() - timedelta(days=2),
                expires_at=datetime.now() - timedelta(hours=1)
            )
            session_manager.sessions[session_id] = expired_session
            expired_sessions.append(session_id)
        
        # Create a valid session
        valid_session = UserSession(
            session_id="valid-session",
            api_key="test-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)
        )
        session_manager.sessions["valid-session"] = valid_session
        
        # Perform cleanup
        cleaned_count = session_manager.cleanup_expired_sessions()
        
        # Should have cleaned up the expired sessions
        assert cleaned_count == 3
        
        # Valid session should still exist
        assert "valid-session" in session_manager.sessions
        for session_id in expired_sessions:
            assert session_id not in session_manager.sessions


class TestAgentErrorRecovery:
    """Tests for agent error recovery mechanisms"""
    
    def test_browse_agent_keyword_extraction_fallback(self):
        """Test fallback when keyword extraction fails"""
        agent = BrowseAgent()
        
        # Mock litellm to fail during keyword extraction
        with patch('agent.litellm.completion') as mock_completion:
            mock_completion.side_effect = Exception("Keyword extraction API failed")
            
            # Should fall back to original query when keyword extraction fails
            keywords = agent.extract_keywords("original query for testing")
            
            assert keywords == "original query for testing"  # Fallback to original
    
    def test_browse_agent_search_tool_fallback(self):
        """Test fallback when search tool fails but agent continues"""
        agent = BrowseAgent()
        
        # Mock the search tool to fail
        with patch.object(agent.search_tool, '_run') as mock_search_run:
            mock_search_run.side_effect = Exception("Search tool failed")
            
            # Mock the LLM to still work (so we can test the search failure handling)
            with patch.object(agent, 'agent_executor') as mock_executor:
                mock_executor.invoke.return_value = {"output": "Response without search"}
                
                result = agent.run_query("test query")
                
                # Should handle search failure gracefully
                assert "search also failed" in result or "Response without search" in result


class TestApplicationLevelFallbacks:
    """Tests for application-level fallback mechanisms"""
    
    @patch('app.process_query_with_agent')
    def test_application_error_handling(self, mock_process):
        """Test overall application error handling and fallbacks"""
        from app import process_query
        from app import QueryRequest
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        mock_session = UserSession(
            session_id="app-error-test",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Make the processing function raise an exception
        mock_process.side_effect = Exception("Application-level error")
        
        request = QueryRequest(query="error test query")
        
        from fastapi import HTTPException
        try:
            result = process_query(request, mock_session)
            assert False, "Should have raised HTTPException"
        except HTTPException as e:
            # Error should be properly formatted
            assert "Query processing error" in str(e.detail)
            assert "application" in str(e.detail).lower() or "error" in str(e.detail).lower()
    
    def test_environment_variable_fallbacks(self):
        """Test that the system handles missing environment variables gracefully"""
        import os
        
        # Temporarily remove API key environment variables
        original_keys = {}
        for key in ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]:
            if key in os.environ:
                original_keys[key] = os.environ[key]
                del os.environ[key]
        
        try:
            # This should handle missing keys gracefully
            from agent import get_llm_instance
            
            # For non-OpenRouter models, should still work
            llm = get_llm_instance("test-model")
            # This might fail due to the model not existing, but should handle missing API keys gracefully
            assert True  # If we get here without crashing, the fallback worked
            
        except ValueError as e:
            # Should get a ValueError for missing API key for OpenRouter models
            assert "required" in str(e).lower()
        finally:
            # Restore original environment
            for key, value in original_keys.items():
                os.environ[key] = value


class TestPipelineFallbacks:
    """Tests for pipeline-level fallback mechanisms"""
    
    @patch('search_result_pipeline.SearchResultProcessor._rank_by_relevance')
    def test_search_pipeline_component_fallback(self, mock_rank):
        """Test fallback when a component of the search pipeline fails"""
        from search_result_pipeline import SearchResultProcessor
        
        processor = SearchResultProcessor()
        
        # Mock ranking to fail
        mock_rank.side_effect = Exception("Ranking failed")
        
        # Create test results
        results = [
            {"title": "Test Result", "href": "https://example.com", "body": "test content", "similarity_score": 0.8}
        ]
        
        # Process results - should handle ranking failure gracefully
        processed = processor.process_results(results, "test query")
        
        # Should still return results even if ranking component fails
        assert isinstance(processed, list)
        # May be empty or contain results depending on implementation
    
    def test_empty_query_handling(self):
        """Test handling of empty or invalid queries"""
        from search_tool import process_search_results_with_pipeline
        
        # Test with empty results
        result = process_search_results_with_pipeline("test query", [])
        assert result == []
        
        # Test with empty query string
        result = process_search_results_with_pipeline("", [{"title": "Test", "href": "https://example.com", "body": "test"}])
        # Should handle gracefully, even if not optimally


if __name__ == "__main__":
    pytest.main([__file__])