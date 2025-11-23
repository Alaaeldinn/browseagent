from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import time
import logging
import secrets

from agent import BrowseAgent, process_query_with_agent
from openrouter import OpenRouterClient, get_default_free_models
from session_manager import session_manager, create_user_session, get_user_session, update_user_model, delete_user_session

# Set up comprehensive logging
import logging
import sys
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("browseagent.log", maxBytes=10000000, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="BrowseAgent API",
    description="An AI research agent that searches, browses, and synthesizes information from the web",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting storage (in production, you'd use Redis or a database)
request_counts = {}
RATE_LIMIT = 100  # requests per hour per API key
RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds

# Add a middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log the incoming request
    logger.info(f"Incoming request: {request.method} {request.url}")

    response = await call_next(request)

    # Log the response
    process_time = time.time() - start_time
    logger.info(f"Request completed: {response.status_code} in {process_time:.2f}s")

    return response

# Define request/response models
class QueryRequest(BaseModel):
    query: str
    llm_provider: Optional[str] = "openai/gpt-3.5-turbo"
    searx_host: Optional[str] = "https://searx.space"
    use_searxng: Optional[bool] = True

class QueryResponse(BaseModel):
    query: str
    llm_provider: str
    result: str
    success: bool
    error: Optional[str] = None

class APIKeyValidationRequest(BaseModel):
    api_key: str

class APIKeyValidationResponse(BaseModel):
    valid: bool
    message: str

class ModelListResponse(BaseModel):
    models: list
    message: str

# Add a new dependency to get session ID from header/cookie
def get_session_id(session_id: str = Header(None, alias="X-Session-ID")):
    """
    Get session ID from header
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required in X-Session-ID header")
    return session_id

def get_session_with_validation(session_id: str = Depends(get_session_id)):
    """
    Get and validate user session
    """
    try:
        session = get_user_session(session_id)
        if not session:
            raise HTTPException(status_code=401, detail={
                "error": "Invalid or expired session",
                "message": "Your session has expired or is invalid. Please create a new session."
            })

        return session
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error validating session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "error": "Session validation error",
            "message": "An error occurred while validating your session"
        })

def check_rate_limit_for_session(session_id: str):
    """
    Check if the session has exceeded the rate limit
    """
    current_time = int(time.time())

    # Clean up old entries
    current_window_start = current_time - RATE_LIMIT_WINDOW
    global request_counts
    request_counts = {k: v for k, v in request_counts.items()
                      if v['timestamp'] > current_window_start}

    # Check if session has made too many requests
    if session_id in request_counts:
        if request_counts[session_id]['count'] >= RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {RATE_LIMIT} requests per hour.",
                    "retry_after": RATE_LIMIT_WINDOW
                }
            )
        else:
            # Increment request count
            request_counts[session_id]['count'] += 1
    else:
        # Initialize request count for this session
        request_counts[session_id] = {
            'count': 1,
            'timestamp': current_time
        }

def validate_and_track_session(session = Depends(get_session_with_validation)):
    """
    Validate session and track usage for rate limiting
    """
    session_id = session.session_id
    api_key = session.api_key

    # Log the API usage
    logger.info(f"Session validated for request: {session_id[:8]}...")

    # Check rate limit
    check_rate_limit_for_session(session_id)

    # Increment request count in session
    session_manager.increment_request_count(session_id)

    return session

# Session creation endpoint
@app.post("/session/create", response_model=APIKeyValidationResponse)
async def create_session(request: APIKeyValidationRequest):
    """
    Create a new session with the provided API key
    """
    try:
        client = OpenRouterClient(api_key=request.api_key)
        validation_result = client.validate_api_key()

        if validation_result["valid"]:
            # Get available models to find a suitable default
            available_models = client.get_available_models()
            available_model_ids = [model["id"] for model in available_models]

            # Find a suitable default model
            selected_model = "openai/gpt-3.5-turbo"  # Default fallback
            for model in DEFAULT_FALLBACK_MODELS:
                if model in available_model_ids:
                    selected_model = model
                    break

            # Create a new session with the selected model
            session_id = create_user_session(request.api_key, selected_model)

            return APIKeyValidationResponse(
                valid=True,
                message=f"Session created successfully. Session ID: {session_id}. Default model: {selected_model}"
            )
        else:
            logger.warning(f"Invalid API key provided for session creation: {str(validation_result.get('details', 'Unknown error'))}")
            return APIKeyValidationResponse(
                valid=False,
                message=validation_result["message"]
            )
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        return APIKeyValidationResponse(
            valid=False,
            message=f"Error creating session: {str(e)}"
        )

def find_fallback_model(client, preferred_model: str = None) -> str:
    """
    Find an appropriate fallback model based on availability
    """
    try:
        # Get available models from OpenRouter
        available_models = client.get_available_models()
        available_model_ids = [model["id"] for model in available_models]

        # If a preferred model is provided and available, use it
        if preferred_model and preferred_model in available_model_ids:
            return preferred_model

        # First, try free models
        free_models = client.get_free_models()
        free_model_ids = [model["id"] for model in free_models]

        for model in free_model_ids:
            if model in available_model_ids:
                return model

        # If no free models are available, try the default fallbacks
        for model in DEFAULT_FALLBACK_MODELS:
            if model in available_model_ids:
                return model

        # If nothing else works, return the first available model (or default)
        if available_model_ids:
            return available_model_ids[0]

        # Fallback to default if no models are available
        return "openai/gpt-3.5-turbo"
    except Exception as e:
        logger.error(f"Error finding fallback model: {str(e)}")
        return "openai/gpt-3.5-turbo"  # Ultimate fallback

# API endpoint for processing queries with the agent
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, session = Depends(validate_and_track_session)):
    """
    Process a query with the BrowseAgent using session info
    """
    try:
        # Get API key and other settings from session
        api_key = session.api_key
        selected_model = session.selected_model
        searx_host = session.searx_host
        use_searxng = session.use_searxng

        # Use the request model if provided, otherwise use session model
        llm_provider = request.llm_provider if request.llm_provider != "openai/gpt-3.5-turbo" else selected_model

        # Validate the model exists, find fallback if not
        client = OpenRouterClient(api_key=api_key)
        if llm_provider not in [model["id"] for model in client.get_available_models()]:
            logger.warning(f"Requested model {llm_provider} not available, finding fallback")
            llm_provider = find_fallback_model(client, llm_provider)

        # Temporarily set the API key in environment for the request
        original_api_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = api_key

        # Get model-specific configuration from session
        model_config = session.get_model_config(llm_provider)

        # Process the query with the agent
        result = process_query_with_agent(
            query=request.query,
            llm_provider=llm_provider,
            searx_host=searx_host,
            use_searxng=use_searxng,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens
        )

        # Restore original API key
        if original_api_key is not None:
            os.environ["OPENROUTER_API_KEY"] = original_api_key
        elif "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]

        return QueryResponse(
            query=result["query"],
            llm_provider=llm_provider,
            result=result["result"],
            success=True
        )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        # Clean up environment in case of error
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Query processing error",
                "message": f"An error occurred while processing your request: {str(e)}"
            }
        )

# API key validation endpoint
@app.post("/validate-api-key", response_model=APIKeyValidationResponse)
async def validate_api_key(request: APIKeyValidationRequest):
    """
    Validate an OpenRouter API key
    """
    try:
        client = OpenRouterClient(api_key=request.api_key)
        validation_result = client.validate_api_key()

        if validation_result["valid"]:
            return APIKeyValidationResponse(
                valid=True,
                message=validation_result["message"]
            )
        else:
            return APIKeyValidationResponse(
                valid=False,
                message=validation_result["message"]
            )
    except Exception as e:
        return APIKeyValidationResponse(
            valid=False,
            message=f"Error validating API key: {str(e)}"
        )

# Endpoint to get account information
@app.get("/account")
async def get_account_info(session = Depends(validate_and_track_session)):
    """
    Get account information including usage and balance
    """
    try:
        api_key = session.api_key
        client = OpenRouterClient(api_key=api_key)
        account_info = client.get_account_balance()

        return account_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching account info: {str(e)}")

# Get available models endpoint
@app.get("/models", response_model=ModelListResponse)
async def get_models(session = Depends(validate_and_track_session)):
    """
    Get list of available models from OpenRouter
    """
    try:
        # Get API key from session
        api_key = session.api_key

        # Temporarily set the API key in environment for the request
        original_api_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = api_key

        client = OpenRouterClient(api_key=api_key)
        models = client.get_free_models()

        # If we couldn't fetch models (maybe API key doesn't have access to model list),
        # return default free models as fallback
        if not models:
            models = get_default_free_models()

        # Restore original API key
        if original_api_key is not None:
            os.environ["OPENROUTER_API_KEY"] = original_api_key
        elif "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]

        return ModelListResponse(
            models=models,
            message=f"Found {len(models)} models"
        )
    except Exception as e:
        # Clean up environment in case of error
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")

# Endpoint to update user's selected model
class ModelSelectionRequest(BaseModel):
    model: str

@app.post("/model/select")
async def select_model(request: ModelSelectionRequest, session = Depends(validate_and_track_session)):
    """
    Update the user's selected model preference
    """
    try:
        # Validate the model exists
        client = OpenRouterClient(api_key=session.api_key)
        models = client.get_available_models()
        available_model_ids = [model["id"] for model in models]

        if request.model not in available_model_ids:
            # Check if it's in the free models as well
            free_models = client.get_free_models()
            free_model_ids = [model["id"] for model in free_models]
            if request.model not in free_model_ids:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid model",
                        "message": f"Model {request.model} is not available",
                        "available_models": available_model_ids[:10]  # Show first 10 models as example
                    }
                )

        success = update_user_model(session.session_id, request.model)
        if success:
            # Update session model preference
            session.selected_model = request.model
            return {"message": f"Model updated to {request.model}", "success": True}
        else:
            raise HTTPException(status_code=400, detail="Failed to update model preference")
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error updating model for session {session.session_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Model update error",
                "message": f"An error occurred while updating the model: {str(e)}"
            }
        )

# Define default free models as fallbacks
DEFAULT_FALLBACK_MODELS = [
    "openai/gpt-3.5-turbo",
    "google/gemma-7b-it",
    "mistralai/mistral-7b-instruct",
    "openchat/openchat-7b",
    "huggingfaceh4/zephyr-7b-beta"
]

# Model configuration request/response models
class ModelConfigRequest(BaseModel):
    model: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    top_p: Optional[float] = 0.9
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0

class ModelConfigResponse(BaseModel):
    model: str
    config: Dict[str, Any]
    message: str

# Endpoint to get/set model-specific configurations
@app.get("/model/config", response_model=ModelConfigResponse)
async def get_model_config(model: str, session = Depends(validate_and_track_session)):
    """
    Get configuration for a specific model
    """
    try:
        config = session.get_model_config(model)
        return ModelConfigResponse(
            model=model,
            config=config.to_dict(),
            message=f"Configuration for {model} retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Error getting model config for {model}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Model config retrieval error",
                "message": f"An error occurred while retrieving model config: {str(e)}"
            }
        )

@app.post("/model/config", response_model=ModelConfigResponse)
async def set_model_config(request: ModelConfigRequest, session = Depends(validate_and_track_session)):
    """
    Set configuration for a specific model
    """
    try:
        from session_manager import ModelConfig

        # Create a ModelConfig instance from the request
        config = ModelConfig(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty
        )

        # Store the configuration in the session
        session.set_model_config(request.model, config)

        return ModelConfigResponse(
            model=request.model,
            config=config.to_dict(),
            message=f"Configuration for {request.model} updated successfully"
        )
    except Exception as e:
        logger.error(f"Error setting model config for {request.model}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Model config update error",
                "message": f"An error occurred while updating model config: {str(e)}"
            }
        )

# Endpoint to delete/end a session
@app.delete("/session")
async def end_session(session = Depends(validate_and_track_session)):
    """
    End the current session
    """
    try:
        session_id = session.session_id
        success = delete_user_session(session_id)

        if success:
            return {"message": "Session ended successfully", "success": True}
        else:
            raise HTTPException(status_code=400, detail="Failed to end session")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ending session: {str(e)}")

# Endpoint to get user's session info
@app.get("/session/info")
async def get_session_info(session = Depends(validate_and_track_session)):
    """
    Get information about the current session
    """
    try:
        # Return session info without exposing the API key directly
        return {
            "session_id": session.session_id,
            "selected_model": session.selected_model,
            "searx_host": session.searx_host,
            "use_searxng": session.use_searxng,
            "request_count": session.request_count,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "is_active": session.is_active
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching session info: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running
    """
    return {"status": "healthy", "message": "BrowseAgent API is running"}

# Simple GET endpoint for testing
@app.get("/")
async def root():
    """
    Root endpoint for basic API information
    """
    return {
        "message": "Welcome to BrowseAgent API",
        "description": "An AI research agent that searches, browses, and synthesizes information from the web",
        "endpoints": [
            {"method": "GET", "path": "/health", "description": "Health check"},
            {"method": "POST", "path": "/query", "description": "Process a query with the agent"}
        ]
    }

# Additional endpoint to get available LLM providers
@app.get("/providers")
async def get_providers():
    """
    Get a list of supported LLM providers
    """
    providers = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "anthropic/claude-3-opus",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-haiku",
        "google/gemini-pro",
        "mistral/mistral-large-latest"
    ]
    return {"providers": providers}
