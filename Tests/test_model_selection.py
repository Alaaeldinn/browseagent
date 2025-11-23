"""
Unit tests for model selection functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from session_manager import session_manager, create_user_session, get_user_session, update_user_model
from app import find_fallback_model


class TestModelSelection:
    """Unit tests for model selection functionality"""
    
    def test_session_model_storage(self):
        """Test storing and retrieving selected model in session"""
        api_key = "test-api-key"
        session_id = create_user_session(api_key, selected_model="openai/gpt-4")
        
        session = get_user_session(session_id)
        assert session is not None
        assert session.selected_model == "openai/gpt-4"
        
        # Update the model
        success = update_user_model(session_id, "anthropic/claude-3")
        assert success is True
        
        updated_session = get_user_session(session_id)
        assert updated_session.selected_model == "anthropic/claude-3"
        
        # Clean up
        session_manager.delete_session(session_id)
        
    def test_model_config_storage_in_session(self):
        """Test storing and retrieving model configurations"""
        from session_manager import ModelConfig
        
        api_key = "test-api-key"
        session_id = create_user_session(api_key)
        session = get_user_session(session_id)
        
        # Create and store a model config
        config = ModelConfig(temperature=0.8, max_tokens=2000)
        session.set_model_config("openai/gpt-4", config)
        
        # Retrieve the config
        retrieved_config = session.get_model_config("openai/gpt-4")
        assert retrieved_config.temperature == 0.8
        assert retrieved_config.max_tokens == 2000
        
        # Test default config for non-existent model
        default_config = session.get_model_config("non-existent-model")
        assert default_config.temperature == 0.7  # Default from class
        
        # Clean up
        session_manager.delete_session(session_id)
        

class TestModelEndpoints:
    """Tests for model selection endpoints in app.py"""
    
    @patch('app.OpenRouterClient')
    def test_get_available_models_endpoint(self, mock_client_class):
        """Test the /models endpoint for getting available models"""
        from app import get_models
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        # Create a mock session
        mock_session = UserSession(
            session_id="test-session",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Mock the OpenRouterClient
        mock_client_instance = Mock()
        mock_client_instance.get_free_models.return_value = [
            {"id": "model1", "name": "Model 1", "pricing": {"prompt": "$0.00"}},
            {"id": "model2", "name": "Model 2", "pricing": {"prompt": "$0.00"}}
        ]
        mock_client_class.return_value = mock_client_instance
        
        response = get_models(mock_session)
        
        assert response["message"] == "Found 2 models"
        assert len(response["models"]) == 2
        assert response["models"][0]["id"] == "model1"
    
    @patch('app.OpenRouterClient')
    def test_get_available_models_fallback(self, mock_client_class):
        """Test the /models endpoint when get_free_models fails"""
        from app import get_models
        from session_manager import UserSession
        from datetime import datetime, timedelta
        from openrouter import get_default_free_models
        
        # Create a mock session
        mock_session = UserSession(
            session_id="test-session",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Mock the OpenRouterClient to fail on get_free_models
        mock_client_instance = Mock()
        mock_client_instance.get_free_models.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client_instance
        
        response = get_models(mock_session)
        
        # Should return default models as fallback
        default_models = get_default_free_models()
        assert response["message"] == f"Found {len(default_models)} models"
        assert len(response["models"]) == len(default_models)
        
    def test_model_selection_endpoint(self):
        """Test the /model/select endpoint for updating selected model"""
        from app import select_model
        from app import ModelSelectionRequest
        import app  # Import the app module to access global functions
        
        # Create a mock session
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        api_key = "test-api-key"
        session_id = create_user_session(api_key)
        mock_session = get_user_session(session_id)
        
        # Create a request object
        request = ModelSelectionRequest(model="anthropic/claude-3")
        
        response = select_model(request, mock_session)
        
        assert response["success"] is True
        assert "anthropic/claude-3" in response["message"]
        
        # Verify the model was actually updated
        updated_session = get_user_session(session_id)
        assert updated_session.selected_model == "anthropic/claude-3"
        
        # Clean up
        session_manager.delete_session(session_id)
    
    @patch('app.OpenRouterClient')
    @patch('app.update_user_model')
    def test_model_selection_with_validation(self, mock_update_model, mock_client_class):
        """Test model selection with validation against available models"""
        from app import select_model
        from app import ModelSelectionRequest
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        # Create a mock session
        mock_session = UserSession(
            session_id="test-session",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Mock the OpenRouterClient to return available models
        mock_client_instance = Mock()
        mock_client_instance.get_available_models.return_value = [
            {"id": "model1", "name": "Model 1"},
            {"id": "valid-model", "name": "Valid Model"},
        ]
        mock_client_instance.get_free_models.return_value = [
            {"id": "valid-model", "name": "Valid Model"}
        ]
        mock_client_class.return_value = mock_client_instance
        
        # Mock update function to return success
        mock_update_model.return_value = True
        
        # Test with a valid model
        request = ModelSelectionRequest(model="valid-model")
        response = select_model(request, mock_session)
        
        assert response["success"] is True
        assert "valid-model" in response["message"]
        
        # Test with an invalid model
        invalid_request = ModelSelectionRequest(model="invalid-model")
        from fastapi import HTTPException
        try:
            select_model(invalid_request, mock_session)
            assert False, "Expected HTTPException for invalid model"
        except HTTPException as e:
            assert "not available" in str(e.detail)


class TestModelFallbacks:
    """Tests for model fallback functionality"""
    
    @patch('app.OpenRouterClient')
    def test_find_fallback_model_success(self, mock_client_class):
        """Test find_fallback_model function with success scenarios"""
        # Mock the OpenRouterClient
        mock_client_instance = Mock()
        mock_client_instance.get_available_models.return_value = [
            {"id": "openai/gpt-4"}, 
            {"id": "anthropic/claude-3"},
            {"id": "google/gemma-7b-it"}
        ]
        mock_client_instance.get_free_models.return_value = [
            {"id": "google/gemma-7b-it"}
        ]
        mock_client_class.return_value = mock_client_instance
        
        # Test when preferred model is available
        fallback = find_fallback_model(mock_client_instance, "openai/gpt-4")
        assert fallback == "openai/gpt-4"
        
        # Test when preferred model is not available but free models are
        fallback = find_fallback_model(mock_client_instance, "nonexistent-model")
        assert fallback == "google/gemma-7b-it"
        
        # Test when no preferred model is specified
        fallback = find_fallback_model(mock_client_instance)
        assert fallback in ["openai/gpt-4", "anthropic/claude-3", "google/gemma-7b-it"]
    
    @patch('app.OpenRouterClient')
    def test_find_fallback_model_with_default_fallbacks(self, mock_client_class):
        """Test find_fallback_model with default fallback list"""
        from app import DEFAULT_FALLBACK_MODELS
        
        # Mock the OpenRouterClient to have some of the default fallbacks
        mock_client_instance = Mock()
        mock_client_instance.get_available_models.return_value = [
            {"id": "non-default-model"},
            {"id": "openai/gpt-3.5-turbo"},
            {"id": "another-model"}
        ]
        mock_client_instance.get_free_models.return_value = []
        mock_client_class.return_value = mock_client_instance
        
        # Should find the first available default fallback
        fallback = find_fallback_model(mock_client_instance)
        assert fallback == "openai/gpt-3.5-turbo"
    
    @patch('app.OpenRouterClient')
    def test_find_fallback_model_exception_handling(self, mock_client_class):
        """Test find_fallback_model handles exceptions gracefully"""
        # Mock the OpenRouterClient to raise an exception
        mock_client_instance = Mock()
        mock_client_instance.get_available_models.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client_instance
        
        # Should return default fallback on error
        fallback = find_fallback_model(mock_client_instance)
        assert fallback == "openai/gpt-3.5-turbo"  # The ultimate fallback
    
    @patch('app.OpenRouterClient')
    def test_find_fallback_model_no_available_models(self, mock_client_class):
        """Test find_fallback_model when no models are available"""
        # Mock the OpenRouterClient to return empty lists
        mock_client_instance = Mock()
        mock_client_instance.get_available_models.return_value = []
        mock_client_instance.get_free_models.return_value = []
        mock_client_class.return_value = mock_client_instance
        
        # Should return default fallback when no models available
        fallback = find_fallback_model(mock_client_instance)
        assert fallback == "openai/gpt-3.5-turbo"  # The ultimate fallback


class TestModelConfigurationEndpoints:
    """Tests for model configuration endpoints"""
    
    def test_get_model_config_endpoint(self):
        """Test getting model configuration"""
        from app import get_model_config
        from session_manager import UserSession, ModelConfig
        from datetime import datetime, timedelta
        
        # Create a mock session with a model config
        mock_session = UserSession(
            session_id="test-session",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Add a model config to the session
        config = ModelConfig(temperature=0.8, max_tokens=1500)
        mock_session.set_model_config("test-model", config)
        
        response = get_model_config("test-model", mock_session)
        
        assert response["model"] == "test-model"
        assert response["config"]["temperature"] == 0.8
        assert response["config"]["max_tokens"] == 1500
        assert "retrieved successfully" in response["message"]
    
    def test_set_model_config_endpoint(self):
        """Test setting model configuration"""
        from app import set_model_config, ModelConfigRequest
        from session_manager import UserSession
        from datetime import datetime, timedelta
        
        # Create a mock session
        mock_session = UserSession(
            session_id="test-session",
            api_key="test-api-key",
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Create a config request
        request = ModelConfigRequest(
            model="test-model",
            temperature=0.9,
            max_tokens=2000,
            top_p=0.8
        )
        
        response = set_model_config(request, mock_session)
        
        assert response["model"] == "test-model"
        assert response["config"]["temperature"] == 0.9
        assert response["config"]["max_tokens"] == 2000
        assert response["config"]["top_p"] == 0.8
        assert "updated successfully" in response["message"]
        
        # Verify the config was stored in the session
        stored_config = mock_session.get_model_config("test-model")
        assert stored_config.temperature == 0.9
        assert stored_config.max_tokens == 2000


# Test utility functions
def test_query_category_classification():
    """Test the query category classification in search_tool"""
    from search_tool import get_query_category
    
    # Test scientific research category
    scientific_queries = [
        "research on quantum computing",
        "academic paper about AI",
        "study on climate change",
        "scientific article about vaccines"
    ]
    for query in scientific_queries:
        assert get_query_category(query) == "scientific_research"
    
    # Test news category
    news_queries = [
        "latest news about politics",
        "current events today",
        "breaking news",
        "recent updates on economy"
    ]
    for query in news_queries:
        assert get_query_category(query) == "news_search"
    
    # Test technical category
    tech_queries = [
        "how to fix Python TypeError",
        "github repository for React",
        "programming tutorial",
        "API documentation"
    ]
    for query in tech_queries:
        assert get_query_category(query) == "technical_search"
    
    # Test multimedia category
    multimedia_queries = [
        "images of Eiffel Tower",
        "video tutorial for cooking",
        "photographs of nature"
    ]
    for query in multimedia_queries:
        assert get_query_category(query) == "multimedia_search"
    
    # Test general category
    general_queries = [
        "what is the weather today",
        "recipe for chocolate cake",
        "how to tie a tie"
    ]
    for query in general_queries:
        assert get_query_category(query) == "general_research"


def test_optimized_engines_selection():
    """Test the optimized engines selection based on query"""
    from search_tool import get_optimized_engines
    
    # Test scientific research
    engines, categories = get_optimized_engines("research on quantum computing")
    assert "arxiv" in engines or "semantic_scholar" in engines
    assert "science" in categories
    
    # Test technical query
    engines, categories = get_optimized_engines("how to fix Python error")
    assert "github" in engines or "stackoverflow" in engines
    assert "it" in categories
    
    # Test news query
    engines, categories = get_optimized_engines("latest news about politics")
    assert "bing_news" in engines or "reddit" in engines
    assert "news" in categories


if __name__ == "__main__":
    pytest.main([__file__])