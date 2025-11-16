"""
LLM Configuration and Integration Module

This module handles the configuration and management of different LLM providers
using LiteLLM, providing a unified interface for multiple LLM backends.
"""

import os
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import litellm
from pydantic import BaseModel, Field
from enum import Enum

# Load environment variables
load_dotenv()


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    GOOGLE = "google"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class LLMConfig(BaseModel):
    """Configuration for a specific LLM model"""
    model_name: str = Field(..., description="Name of the model")
    provider: LLMProvider = Field(..., description="LLM provider")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Base URL for the provider")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens for the model")
    temperature: float = Field(0.1, description="Temperature for response generation")
    top_p: Optional[float] = Field(None, description="Top-p sampling parameter")
    frequency_penalty: Optional[float] = Field(None, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(None, description="Presence penalty")


class LLMManager:
    """
    Manager for handling multiple LLM providers and configurations.
    
    This class provides a unified interface for interacting with different
    LLM providers through LiteLLM, with proper API key management and
    configuration validation.
    """
    
    def __init__(self):
        """Initialize the LLM manager with default configurations."""
        self.configs: Dict[str, LLMConfig] = {}
        self.default_model: str = "gpt-3.5-turbo"
        self._load_default_configs()
    
    def _load_default_configs(self):
        """Load default configurations from environment variables."""
        # OpenAI configuration
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.add_config(LLMConfig(
                model_name="gpt-3.5-turbo",
                provider=LLMProvider.OPENAI,
                api_key=openai_api_key
            ))
            
            self.add_config(LLMConfig(
                model_name="gpt-4",
                provider=LLMProvider.OPENAI,
                api_key=openai_api_key
            ))
            
            self.add_config(LLMConfig(
                model_name="gpt-4-turbo",
                provider=LLMProvider.OPENAI,
                api_key=openai_api_key
            ))
        
        # Anthropic configuration
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            self.add_config(LLMConfig(
                model_name="claude-3-sonnet-20240229",
                provider=LLMProvider.ANTHROPIC,
                api_key=anthropic_api_key
            ))
            
            self.add_config(LLMConfig(
                model_name="claude-3-opus-20240229",
                provider=LLMProvider.ANTHROPIC,
                api_key=anthropic_api_key
            ))
        
        # Google configuration
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            self.add_config(LLMConfig(
                model_name="gemini-pro",
                provider=LLMProvider.GOOGLE,
                api_key=google_api_key
            ))
        
        # Cohere configuration
        cohere_api_key = os.getenv("COHERE_API_KEY")
        if cohere_api_key:
            self.add_config(LLMConfig(
                model_name="command",
                provider=LLMProvider.COHERE,
                api_key=cohere_api_key
            ))
        
        # Azure configuration
        azure_api_key = os.getenv("AZURE_API_KEY")
        azure_api_base = os.getenv("AZURE_API_BASE")
        azure_api_version = os.getenv("AZURE_API_VERSION")
        
        if azure_api_key and azure_api_base:
            self.add_config(LLMConfig(
                model_name="gpt-35-turbo",
                provider=LLMProvider.AZURE,
                api_key=azure_api_key,
                base_url=f"{azure_api_base}/openai/deployments/{os.getenv('AZURE_DEPLOYMENT_NAME', 'gpt-35-turbo')}"
            ))
    
    def add_config(self, config: LLMConfig):
        """
        Add a new LLM configuration.
        
        Args:
            config: LLMConfig object containing the model configuration
        """
        self.configs[config.model_name] = config
    
    def get_config(self, model_name: str) -> Optional[LLMConfig]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            LLMConfig object if found, None otherwise
        """
        return self.configs.get(model_name)
    
    def list_available_models(self) -> List[str]:
        """
        Get list of all available model names.
        
        Returns:
            List of model names
        """
        return list(self.configs.keys())
    
    def list_models_by_provider(self, provider: LLMProvider) -> List[str]:
        """
        Get list of models from a specific provider.
        
        Args:
            provider: LLM provider to filter by
            
        Returns:
            List of model names from the specified provider
        """
        return [
            model_name for model_name, config in self.configs.items()
            if config.provider == provider
        ]
    
    def set_default_model(self, model_name: str) -> bool:
        """
        Set the default model to use.
        
        Args:
            model_name: Name of the model to set as default
            
        Returns:
            True if successful, False if model not found
        """
        if model_name in self.configs:
            self.default_model = model_name
            return True
        return False
    
    def get_default_model(self) -> str:
        """
        Get the current default model.
        
        Returns:
            Name of the default model
        """
        return self.default_model
    
    def validate_config(self, model_name: str) -> bool:
        """
        Validate if a model configuration is complete.
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        config = self.get_config(model_name)
        if not config:
            return False
        
        # Check if required API key is present
        if config.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC, 
                              LLMProvider.GOOGLE, LLMProvider.COHERE, 
                              LLMProvider.AZURE]:
            if not config.api_key:
                return False
        
        return True
    
    def get_litellm_params(self, model_name: str) -> Dict[str, Any]:
        """
        Get parameters for LiteLLM from a model configuration.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of parameters for LiteLLM
        """
        config = self.get_config(model_name)
        if not config:
            raise ValueError(f"Model {model_name} not found")
        
        params = {
            "model": model_name,
            "temperature": config.temperature,
        }
        
        # Add optional parameters if they're set
        if config.max_tokens:
            params["max_tokens"] = config.max_tokens
        if config.top_p:
            params["top_p"] = config.top_p
        if config.frequency_penalty:
            params["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty:
            params["presence_penalty"] = config.presence_penalty
        
        # Add provider-specific parameters
        if config.provider == LLMProvider.AZURE and config.base_url:
            params["api_base"] = config.base_url
            params["api_version"] = os.getenv("AZURE_API_VERSION", "2023-12-01-preview")
        
        return params
    
    def test_model(self, model_name: str, test_query: str = "Hello, how are you?") -> Dict[str, Any]:
        """
        Test a model with a sample query.
        
        Args:
            model_name: Name of the model to test
            test_query: Test query to send to the model
            
        Returns:
            Dictionary containing test results
        """
        try:
            if not self.validate_config(model_name):
                return {
                    "success": False,
                    "error": f"Invalid configuration for model {model_name}"
                }
            
            params = self.get_litellm_params(model_name)
            
            # Make the API call
            response = litellm.completion(
                messages=[{"role": "user", "content": test_query}],
                **params
            )
            
            return {
                "success": True,
                "model": model_name,
                "response": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "model": model_name,
                "error": str(e)
            }
    
    def test_all_models(self, test_query: str = "Hello, how are you?") -> Dict[str, Any]:
        """
        Test all available models with a sample query.
        
        Args:
            test_query: Test query to send to the models
            
        Returns:
            Dictionary containing test results for all models
        """
        results = {}
        for model_name in self.configs:
            results[model_name] = self.test_model(model_name, test_query)
        return results


# Global LLM manager instance
llm_manager = LLMManager()


def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance."""
    return llm_manager


def get_available_models_info() -> Dict[str, Dict[str, Any]]:
    """
    Get detailed information about all available models.
    
    Returns:
        Dictionary with model information
    """
    models_info = {}
    manager = get_llm_manager()
    
    for model_name, config in manager.configs.items():
        models_info[model_name] = {
            "provider": config.provider.value,
            "is_valid": manager.validate_config(model_name),
            "is_default": model_name == manager.default_model,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
    
    return models_info
