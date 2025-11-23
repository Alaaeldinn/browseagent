"""
End-to-end integration tests for the complete BrowseAgent search flow
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from fastapi.testclient import TestClient
from app import app
from agent import process_query_with_agent
from session_manager import session_manager


class TestEndToEndFlow:
    """End-to-end tests for the complete BrowseAgent flow"""
    
    def test_complete_search_flow_with_mocked_components(self):
        """Test the complete flow with mocked external dependencies"""
        # Mock the search tool to return test results
        with patch('search_tool.SearXNGSearchTool._run') as mock_search_run:
            mock_search_run.return_value = str([
                {
                    "title": "Test Result",
                    "href": "https://example.com",
                    "body": "This is a test result for the query",
                    "engine": "test-engine",
                    "similarity_score": 0.95
                }
            ])
            
            # Mock the LLM to return a test response
            with patch('agent.get_llm_instance') as mock_get_llm:
                mock_llm = Mock()
                mock_llm.invoke.return_value = "This is the AI response to the test query"
                mock_get_llm.return_value = mock_llm
                
                # Test the complete flow
                result = process_query_with_agent(
                    query="test query",
                    llm_provider="openai/gpt-3.5-turbo",
                    use_searxng=True
                )
                
                # Verify the result structure
                assert result["query"] == "test query"
                assert result["llm_provider"] == "openai/gpt-3.5-turbo"
                assert result["status"] == "success"
                assert "AI response" in result["result"]
                
                # Verify search tool was called
                mock_search_run.assert_called_once()
    
    def test_complete_flow_with_different_models(self):
        """Test the complete flow with different model providers"""
        test_models = [
            "openai/gpt-3.5-turbo",
            "google/gemma-7b-it", 
            "huggingfaceh4/zephyr-7b-beta"
        ]
        
        for model in test_models:
            with patch('search_tool.SearXNGSearchTool._run') as mock_search_run:
                mock_search_run.return_value = str([
                    {"title": "Result", "href": "https://example.com", "body": "test", "similarity_score": 0.8}
                ])
                
                with patch('agent.get_llm_instance') as mock_get_llm:
                    mock_llm = Mock()
                    mock_llm.invoke.return_value = f"Response from {model}"
                    mock_get_llm.return_value = mock_llm
                    
                    result = process_query_with_agent(
                        query="model test query",
                        llm_provider=model
                    )
                    
                    assert result["llm_provider"] == model
                    assert result["status"] == "success"
    
    def test_flow_with_model_specific_configurations(self):
        """Test the complete flow with model-specific configurations"""
        from session_manager import ModelConfig
        
        config = ModelConfig(temperature=0.9, max_tokens=2000)
        
        with patch('search_tool.SearXNGSearchTool._run') as mock_search_run:
            mock_search_run.return_value = str([
                {"title": "Config Test", "href": "https://example.com", "body": "test", "similarity_score": 0.85}
            ])
            
            with patch('agent.get_llm_instance') as mock_get_llm:
                mock_llm = Mock()
                mock_llm.invoke.return_value = "Response with custom config"
                mock_get_llm.return_value = mock_llm
                
                result = process_query_with_agent(
                    query="config test query",
                    llm_provider="openai/gpt-4",
                    temperature=0.9,
                    max_tokens=2000
                )
                
                # Verify the configuration was passed through
                # Note: This checks that parameters were accepted, actual usage tested elsewhere
                assert result["status"] == "success"
                assert "custom config" in result["result"]


class TestAPIClientFlow:
    """End-to-end tests using FastAPI TestClient"""
    
    def setup_method(self):
        """Set up test client for each test"""
        self.client = TestClient(app)
    
    @patch('app.create_user_session')
    @patch('app.OpenRouterClient')
    def test_session_creation_and_query_flow(self, mock_client, mock_create_session):
        """Test the complete flow from session creation to query processing"""
        # Mock a valid API key validation
        mock_client_instance = Mock()
        mock_client_instance.validate_api_key.return_value = {
            "valid": True,
            "message": "API key is valid"
        }
        mock_client_instance.get_available_models.return_value = [
            {"id": "openai/gpt-3.5-turbo", "name": "GPT-3.5 Turbo"}
        ]
        mock_client.return_value = mock_client_instance
        
        # Mock session creation
        mock_create_session.return_value = "test-session-id-123"
        
        # Test session creation
        response = self.client.post("/session/create", json={
            "api_key": "test-api-key-123"
        })
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["valid"] is True
        assert "Session created successfully" in response_data["message"]
        assert "test-session-id-123" in response_data["message"]
        
        # Now test the query flow with the created session
        with patch('app.process_query_with_agent') as mock_process:
            mock_process.return_value = {
                "query": "test query",
                "llm_provider": "openai/gpt-3.5-turbo", 
                "result": "Test response from agent",
                "search_engine": "searxng",
                "status": "success"
            }
            
            response = self.client.post("/query", 
                json={"query": "test query", "llm_provider": "openai/gpt-3.5-turbo"},
                headers={"X-Session-ID": "test-session-id-123"}
            )
            
            assert response.status_code == 200
            query_result = response.json()
            assert query_result["query"] == "test query"
            assert query_result["result"] == "Test response from agent"
    
    @patch('app.OpenRouterClient')
    def test_models_endpoint_with_session(self, mock_client):
        """Test the models endpoint with a valid session"""
        # Mock client
        mock_client_instance = Mock()
        mock_client_instance.get_free_models.return_value = [
            {"id": "model1", "name": "Model 1", "pricing": {"prompt": "$0.00"}},
            {"id": "model2", "name": "Model 2", "pricing": {"prompt": "$0.00"}}
        ]
        mock_client.return_value = mock_client_instance
        
        # Mock the session validation dependency
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        mock_session = UserSession(
            session_id="test-session",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Since we can't easily mock FastAPI dependencies in TestClient,
        # we'll test the underlying function directly
        from app import get_models
        result = get_models(mock_session)
        
        assert result["message"] == "Found 2 models"
        assert len(result["models"]) == 2
    
    @patch('app.process_query_with_agent')
    def test_query_endpoint_session_flow(self, mock_process):
        """Test query endpoint with session-based flow"""
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        mock_session = UserSession(
            session_id="test-query-session",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
            selected_model="openai/gpt-3.5-turbo"
        )
        
        # Mock the processing function
        mock_process.return_value = {
            "query": "end-to-end test query",
            "llm_provider": "openai/gpt-3.5-turbo",
            "result": "End-to-end test response",
            "search_engine": "searxng", 
            "status": "success"
        }
        
        # Test the underlying function directly
        from app import process_query
        from app import QueryRequest
        
        request = QueryRequest(
            query="end-to-end test query",
            llm_provider="openai/gpt-3.5-turbo"
        )
        
        result = process_query(request, mock_session)
        
        assert result["query"] == "end-to-end test query"
        assert result["result"] == "End-to-end test response"
        assert result["status"] == "success"


class TestUserJourneyFlow:
    """Test complete user journey flows"""
    
    def test_new_user_journey(self):
        """Test the complete journey for a new user"""
        from session_manager import session_manager
        
        # Step 1: Create session with API key
        with patch('app.OpenRouterClient') as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.validate_api_key.return_value = {
                "valid": True,
                "message": "API key is valid"
            }
            # Mock available models
            mock_client_instance.get_available_models.return_value = [
                {"id": "openai/gpt-3.5-turbo"},
                {"id": "google/gemma-7b-it"}
            ]
            mock_client_instance.get_free_models.return_value = [
                {"id": "google/gemma-7b-it"}
            ]
            mock_client.return_value = mock_client_instance
            
            from app import create_session, APIKeyValidationRequest
            request = APIKeyValidationRequest(api_key="test-user-api-key")
            
            session_response = create_session(request)
            assert session_response.valid is True
            
            # Extract session ID from the message
            import re
            session_id_match = re.search(r'Session ID: (\w+)', session_response.message)
            assert session_id_match, "Session ID not found in response"
            session_id = session_id_match.group(1)
            
            # Step 2: Get available models
            from session_manager import get_user_session
            user_session = get_user_session(session_id)
            assert user_session is not None
            
            from app import get_models
            models_response = get_models(user_session)
            assert models_response["message"].startswith("Found")
            
            # Step 3: Select a model
            from app import select_model, ModelSelectionRequest
            model_request = ModelSelectionRequest(model="google/gemma-7b-it")
            model_response = select_model(model_request, user_session)
            assert model_response["success"] is True
            
            # Step 4: Process a query
            with patch('app.process_query_with_agent') as mock_process:
                mock_process.return_value = {
                    "query": "Hello world query",
                    "llm_provider": "google/gemma-7b-it",
                    "result": "Hello world response",
                    "search_engine": "searxng",
                    "status": "success"
                }
                
                from app import process_query, QueryRequest
                query_request = QueryRequest(query="Hello world query", llm_provider="google/gemma-7b-it")
                query_response = process_query(query_request, user_session)
                
                assert query_response["result"] == "Hello world response"
            
            # Step 5: Clean up session
            session_manager.delete_session(session_id)
    
    def test_model_switching_flow(self):
        """Test the flow of switching between different models"""
        from session_manager import session_manager
        
        # Create a session
        session_id = "test-switch-session"
        api_key = "test-api-key"
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        # Add session directly for testing
        test_session = UserSession(
            session_id=session_id,
            api_key=api_key,
            created_at=datetime.now(),
            last_accessed=datetime.now(), 
            expires_at=datetime.now() + timedelta(hours=1),
            selected_model="openai/gpt-3.5-turbo"
        )
        session_manager.sessions[session_id] = test_session
        
        # Verify initial model
        session = session_manager.get_session(session_id)
        assert session.selected_model == "openai/gpt-3.5-turbo"
        
        # Switch to a different model
        success = session_manager.update_session_model(session_id, "google/gemma-7b-it")
        assert success is True
        
        # Verify the model changed
        updated_session = session_manager.get_session(session_id)
        assert updated_session.selected_model == "google/gemma-7b-it"
        
        # Clean up
        session_manager.delete_session(session_id)


class TestSearXNGIntegrationFlow:
    """Test the SearXNG integration end-to-end"""
    
    def test_searxng_query_optimization_flow(self):
        """Test how queries are optimized for different types of searches"""
        from search_tool import get_query_category, get_optimized_engines, SearXNGSearchTool
        
        # Test scientific query optimization
        sci_query = "research papers on quantum computing"
        category = get_query_category(sci_query)
        assert category == "scientific_research"
        
        engines, categories = get_optimized_engines(sci_query)
        # For scientific research, should use academic engines
        assert "science" in categories or any(cat in ["arxiv", "semantic_scholar"] for cat in engines if engines)
        
        # Test technical query optimization
        tech_query = "how to fix python TypeError"
        category = get_query_category(tech_query)
        assert category == "technical_search"
        
        engines, categories = get_optimized_engines(tech_query)
        # For technical queries, should use dev platforms
        assert "it" in categories or any(cat in ["github", "stackoverflow"] for cat in engines if engines)
        
        # Test the complete flow through the tool
        with patch('search_tool.SearxSearchWrapper') as mock_wrapper:
            mock_instance = Mock()
            mock_instance.results.return_value = [
                {"title": "Quantum Computing Research", "url": "https://arxiv.org", "content": "research content", "engine": "arxiv"}
            ]
            mock_wrapper.return_value = mock_instance
            
            with patch('search_tool.process_search_results_with_pipeline') as mock_pipeline:
                mock_pipeline.return_value = [
                    {"title": "Quantum Computing Research", "href": "https://arxiv.org", "body": "research content", "engine": "arxiv", "similarity_score": 0.95}
                ]
                
                tool = SearXNGSearchTool()
                result = tool._run(sci_query)
                
                # Should use optimized engines for the query type
                assert "Quantum Computing Research" in result


class TestErrorRecoveryFlow:
    """Test error recovery in end-to-end flows"""
    
    @patch('search_tool.SearxSearchWrapper')
    def test_searxng_fallback_flow(self, mock_wrapper):
        """Test SearXNG tool fallback when primary search fails"""
        from search_tool import SearXNGSearchTool
        
        # Mock the wrapper to fail initially but work on fallback
        mock_instance = Mock()
        mock_instance.results.side_effect = [
            Exception("Primary search failed"),  # First call fails
            [{"title": "Fallback result", "url": "https://example.com", "content": "fallback content", "engine": "fallback"}]  # Second call succeeds
        ]
        mock_wrapper.return_value = mock_instance
        
        with patch('search_tool.process_search_results_with_pipeline') as mock_pipeline:
            mock_pipeline.return_value = [
                {"title": "Fallback result", "href": "https://example.com", "body": "fallback content", "engine": "fallback", "similarity_score": 0.8}
            ]
            
            tool = SearXNGSearchTool(k=3)  # Should try fallback with 3 results max
            result = tool._run("test query")
            
            # Should succeed with fallback results
            assert "Fallback result" in result
            assert mock_instance.results.call_count >= 1
    
    @patch('agent.get_llm_instance')
    def test_llm_fallback_flow(self, mock_get_llm):
        """Test LLM fallback when primary model fails"""
        from agent import BrowseAgent
        
        # Mock the primary LLM to fail, then success on fallback
        def llm_side_effect(model, **kwargs):
            if "gpt-4" in model:
                raise Exception("Model not available")
            else:
                mock_llm = Mock()
                mock_llm.invoke.return_value = "Response from fallback model"
                return mock_llm
        
        mock_get_llm.side_effect = llm_side_effect
        
        agent = BrowseAgent(llm_provider="openai/gpt-4")  # This will fail
        result = agent.run_query("test query")  # This should trigger fallback
        
        # Should use fallback model and succeed
        assert "fallback model" in result or "direct search" in result


if __name__ == "__main__":
    pytest.main([__file__])