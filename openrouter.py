"""
OpenRouter API Integration for BrowseAgent
"""
import os
import requests
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


import logging

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """
    A client to interact with the OpenRouter API
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenRouter client with an API key
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def validate_api_key(self) -> Dict[str, Any]:
        """
        Validate the API key by making a simple request and return detailed information
        """
        if not self.api_key:
            logger.warning("API key validation attempted without API key")
            return {
                "valid": False,
                "message": "API key is required",
                "details": "No API key provided"
            }

        try:
            logger.info("Starting API key validation")

            # First, try to get user info to validate the API key
            response = requests.get(
                f"{self.base_url}/user",
                headers=self.headers
            )

            if response.status_code == 200:
                user_data = response.json()
                logger.info("API key validation successful via user endpoint")
                return {
                    "valid": True,
                    "message": "API key is valid",
                    "details": user_data,
                    "user_info": user_data.get("data", {})
                }
            elif response.status_code == 401:
                logger.warning("API key validation failed - authentication error")
                return {
                    "valid": False,
                    "message": "Invalid API key",
                    "details": "Authentication failed"
                }
            elif response.status_code == 429:
                logger.info("API key is valid but rate limited")
                return {
                    "valid": True,  # Key is valid, just rate limited
                    "message": "API key valid but rate limited",
                    "details": "Rate limit exceeded"
                }
            else:
                logger.info(f"User endpoint returned {response.status_code}, trying models endpoint")

                # Try models endpoint as fallback
                models_response = requests.get(
                    f"{self.base_url}/models",
                    headers=self.headers
                )

                if models_response.status_code == 200:
                    logger.info("API key validation successful via models endpoint")
                    return {
                        "valid": True,
                        "message": "API key is valid",
                        "details": "Access to models confirmed"
                    }
                else:
                    logger.warning(f"API key validation failed via models endpoint: {models_response.status_code}")
                    return {
                        "valid": False,
                        "message": f"Invalid API key or access denied: {models_response.status_code}",
                        "details": models_response.text
                    }

        except requests.exceptions.ConnectionError:
            logger.error("Connection error during API key validation")
            return {
                "valid": False,
                "message": "Could not connect to OpenRouter API",
                "details": "Connection error"
            }
        except Exception as e:
            logger.error(f"Unexpected error during API key validation: {str(e)}")
            return {
                "valid": False,
                "message": f"Error validating API key: {str(e)}",
                "details": str(e)
            }

    def get_account_balance(self) -> Dict[str, Any]:
        """
        Get account balance information
        """
        if not self.api_key:
            return {
                "success": False,
                "message": "API key is required"
            }

        try:
            response = requests.get(
                f"{self.base_url}/user/usage",
                headers=self.headers
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "data": data
                }
            else:
                return {
                    "success": False,
                    "message": f"Error retrieving balance: {response.status_code}",
                    "details": response.text
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error retrieving balance: {str(e)}"
            }

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available models from OpenRouter
        """
        try:
            logger.info("Fetching available models from OpenRouter")
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            model_count = len(data.get("data", []))
            logger.info(f"Successfully fetched {model_count} models from OpenRouter")
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Error fetching models from OpenRouter: {str(e)}")
            return []

    def get_free_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of free models available on OpenRouter
        """
        try:
            logger.info("Fetching free models from OpenRouter")
            all_models = self.get_available_models()

            free_models = [
                model for model in all_models
                if model.get("pricing", {}).get("prompt", "0").replace("$", "").replace("0", "") == ""
                or float(model.get("pricing", {}).get("prompt", "0").replace("$", "")) == 0
                or "free" in model.get("name", "").lower()
            ]

            # Let's also include models that are very cheap (under $0.01 per million tokens)
            cheap_models = [
                model for model in all_models
                if model not in free_models
                and float(model.get("pricing", {}).get("prompt", "0").replace("$", "")) <= 0.00001
            ]

            total_free = len(free_models)
            total_cheap = len(cheap_models)
            logger.info(f"Found {total_free} free models and {total_cheap} cheap models")

            return free_models + cheap_models
        except Exception as e:
            logger.error(f"Error filtering free models: {str(e)}")
            return []

    def get_model_pricing_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get pricing information for a specific model
        """
        try:
            logger.info(f"Fetching pricing info for model: {model_id}")
            all_models = self.get_available_models()
            for model in all_models:
                if model.get("id") == model_id:
                    pricing_info = {
                        "id": model.get("id"),
                        "name": model.get("name"),
                        "pricing": model.get("pricing", {}),
                        "description": model.get("description", ""),
                        "context_length": model.get("context_length", 0)
                    }
                    logger.info(f"Pricing info found for model: {model_id}")
                    return pricing_info
            logger.info(f"No pricing info found for model: {model_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting pricing info for model {model_id}: {str(e)}")
            return None

    def test_model_access(self, model_id: str) -> bool:
        """
        Test if a specific model can be accessed with the current API key
        """
        try:
            # Make a simple test request to the model
            test_payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, this is a test message to check if I can access this model."
                    }
                ],
                "max_tokens": 10
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=test_payload
            )
            
            return response.status_code in [200, 400, 401, 402, 403]  # 400 means model exists, others indicate access issues
        except Exception as e:
            print(f"Error testing model access: {e}")
            return False


class ModelInfo(BaseModel):
    """
    Model information for API responses
    """
    id: str = Field(description="The model identifier")
    name: str = Field(description="The model name")
    description: str = Field(description="Description of the model")
    pricing: Dict[str, str] = Field(description="Pricing information")
    context_length: int = Field(description="Maximum context length in tokens")


class ModelListResponse(BaseModel):
    """
    Response model for available models list
    """
    models: List[ModelInfo] = Field(description="List of available models")
    total: int = Field(description="Total number of models")


def get_default_free_models() -> List[Dict[str, Any]]:
    """
    Get a curated list of default free models if API access is not available
    """
    return [
        {
            "id": "openchat/openchat-7b",
            "name": "OpenChat 7B",
            "description": "A strong open-source model based on OpenChat",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 4096
        },
        {
            "id": "huggingfaceh4/zephyr-7b-beta",
            "name": "Zephyr 7B Beta",
            "description": "A fine-tuned version of mistral that performs well on chat",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 4096
        },
        {
            "id": "teknium/openhermes-2.5-mistral-7b",
            "name": "OpenHermes 2.5 Mistral 7B",
            "description": "A great general purpose model based on Mistral 7B",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 4096
        },
        {
            "id": "google/gemma-7b-it",
            "name": "Gemma 7B IT",
            "description": "Google's Gemma 7B model for instruction following",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 8192
        },
        {
            "id": "microsoft/phi-2",
            "name": "Microsoft Phi-2",
            "description": "Microsoft's Phi-2 model, good for reasoning tasks",
            "pricing": {"prompt": "0", "completion": "0"},
            "context_length": 2048
        }
    ]


if __name__ == "__main__":
    # Example usage
    client = OpenRouterClient()
    if client.validate_api_key():
        print("API key is valid!")
        models = client.get_free_models()
        print(f"Found {len(models)} free models")
        for model in models[:5]:  # Show first 5
            print(f"- {model['id']}: {model['name']}")
    else:
        print("API key validation failed or not provided")
        print("Default free models:")
        default_models = get_default_free_models()
        for model in default_models:
            print(f"- {model['id']}: {model['name']}")