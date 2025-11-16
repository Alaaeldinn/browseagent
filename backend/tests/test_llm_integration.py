"""
Tests for LLM Integration and Configuration

This module tests the LLM configuration system, including:
1. LLMManager functionality
2. Model configuration and validation
3. API key management
4. Model switching
"""

import pytest
import os
from dotenv import load_dotenv
from llm_config import LLMManager, LLMConfig, LLMProvider, get_available_models_info

# Load environment variables
load_dotenv()


class TestLLMManager:
    """Test cases for LLMManager class"""
    
    def test_llm_manager_initialization(self):
        """Test that LLMManager initializes correctly"""
        manager = LLMManager()
        assert isinstance(manager, LLMManager)
        assert isinstance(manager.configs, dict)
        assert manager.default_model is not None
    
    def test_add_and_get_config(self):
        """Test adding and retrieving model configurations"""
        manager = LLMManager()
        
        # Create a test configuration
        config = LLMConfig(
            model_name="test-model",
            provider=LLMProvider.OPENAI,
            api_key="test-key",
            temperature=0.2
        )
        
        # Add the configuration
        manager.add_config(config)
        
        # Retrieve the configuration
        retrieved_config = manager.get_config("test-model")
        assert retrieved_config is not None
        assert retrieved_config.model_name == "test-model"
        assert retrieved_config.provider == LLMProvider.OPENAI
        assert retrieved_config.api_key == "test-key"
        assert retrieved_config.temperature == 0.2
    
    def test_list_available_models(self):
        """Test listing available models"""
        manager = LLMManager()
        models = manager.list_available_models()
        
        assert isinstance(models, list)
        # Should have at least the default models if API keys are set
        if os.getenv("OPENAI_API_KEY"):
            assert len(models) > 0
    
    def test_list_models_by_provider(self):
        """Test filtering models by provider"""
        manager = LLMManager()
        
        # Test with OpenAI provider
        openai_models = manager.list_models_by_provider(LLMProvider.OPENAI)
        assert isinstance(openai_models, list)
        
        # Test with Anthropic provider
        anthropic_models = manager.list_models_by_provider(LLMProvider.ANTHROPIC)
        assert isinstance(anthropic_models, list)
    
    def test_set_default_model(self):
        """Test setting default model"""
        manager = LLMManager()
        
        # Get current default
        original_default = manager.get_default_model()
        
        # Set a new default (if available)
        models = manager.list_available_models()
        if len(models) > 1:
            new_default = models[1]
            success = manager.set_default_model(new_default)
            assert success is True
            assert manager.get_default_model() == new_default
            
            # Reset to original
            manager.set_default_model(original_default)
    
    def test_validate_config(self):
        """Test configuration validation"""
        manager = LLMManager()
        
        # Test with valid model
        if manager.list_available_models():
            valid_model = manager.list_available_models()[0]
            assert manager.validate_config(valid_model) is True
        
        # Test with invalid model
        assert manager.validate_config("nonexistent-model") is False
    
    def test_get_litellm_params(self):
        """Test getting LiteLLM parameters"""
        manager = LLMManager()
        
        if manager.list_available_models():
            model_name = manager.list_available_models()[0]
            params = manager.get_litellm_params(model_name)
            
            assert isinstance(params, dict)
            assert "model" in params
            assert params["model"] == model_name
            assert "temperature" in params
    
    def test_model_without_api_key(self):
        """Test model configuration without API key"""
        manager = LLMManager()
        
        # Create a config without API key
        config = LLMConfig(
            model_name="no-api-key-model",
            provider=LLMProvider.OPENAI,
            api_key=None
        )
        
        manager.add_config(config)
        
        # Should fail validation
        assert manager.validate_config("no-api-key-model") is False


class TestBrowseAgentLLMIntegration:
    """Test cases for BrowseAgent LLM integration"""
    
    def test_agent_initialization_with_llm_manager(self):
        """Test that BrowseAgent initializes with LLMManager"""
        from agent import BrowseAgent
        
        # Create a mock LLM manager
        llm_manager = LLMManager()
        
        # Initialize agent
        agent = BrowseAgent(llm_manager=llm_manager)
        
        assert agent.llm_manager is llm_manager
        assert agent.model_name is not None
    
    def test_agent_model_switching(self):
        """Test switching models in the agent"""
        from agent import BrowseAgent
        
        llm_manager = LLMManager()
        agent = BrowseAgent(llm_manager=llm_manager)
        
        # Get available models
        models = agent.get_available_models()
        
        if len(models) > 1:
            original_model = agent.model_name
            new_model = models[1]
            
            # Switch model
            agent.set_model(new_model)
            assert agent.model_name == new_model
            
            # Switch back
            agent.set_model(original_model)
            assert agent.model_name == original_model
    
    def test_agent_model_info(self):
        """Test getting model information"""
        from agent import BrowseAgent
        
        llm_manager = LLMManager()
        agent = BrowseAgent(llm_manager=llm_manager)
        
        model_info = agent.get_model_info()
        
        assert isinstance(model_info, dict)
        assert "model_name" in model_info
        assert "provider" in model_info
        assert "is_valid" in model_info
        assert model_info["model_name"] == agent.model_name
    
    def test_agent_model_testing(self):
        """Test model functionality"""
        from agent import BrowseAgent
        
        llm_manager = LLMManager()
        agent = BrowseAgent(llm_manager=llm_manager)
        
        # Test current model
        test_result = agent.test_current_model("Hello")
        assert isinstance(test_result, dict)
        assert "success" in test_result
        assert "model" in test_result
        assert test_result["model"] == agent.model_name


class TestAPIEndpoints:
    """Test cases for API endpoints related to LLM integration"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        return client
    
    def test_get_models_endpoint(self, client):
        """Test /models endpoint"""
        response = client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
    
    def test_get_models_info_endpoint(self, client):
        """Test /models/info endpoint"""
        response = client.get("/models/info")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, dict)
    
    def test_get_current_model_info_endpoint(self, client):
        """Test /models/current endpoint"""
        response = client.get("/models/current")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, dict)
        assert "model_name" in data
    
    def test_get_models_by_provider_endpoint(self, client):
        """Test /models/providers endpoint"""
        response = client.get("/models/providers/openai")
        assert response.status_code == 200
        
        data = response.json()
        assert "provider" in data
        assert "models" in data
        assert isinstance(data["models"], list)
    
    def test_test_model_endpoint(self, client):
        """Test /models/test endpoint"""
        # Test with current model
        response = client.post("/models/test", json={"test_query": "Hello"})
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "model" in data
    
    def test_test_all_models_endpoint(self, client):
        """Test /models/test/all endpoint"""
        response = client.post("/models/test/all")
        assert response.status_code == 200
        
        data = response.json()
        assert "test_query" in data
        assert "results" in data
        assert isinstance(data["results"], dict)


if __name__ == "__main__":
    pytest.main([__file__])
