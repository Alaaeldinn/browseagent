from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import time
import logging

from agent import BrowseAgent, process_query_with_agent
from openrouter import OpenRouterClient, get_default_free_models

# Set up logging
logging.basicConfig(level=logging.INFO)
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

def get_api_key(api_key: str = Header(None, alias="X-API-Key")):
    """
    Get API key from header and validate it
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="API key is required in X-API-Key header")

    # Validate the API key format (basic validation)
    if len(api_key) < 20:  # OpenRouter API keys are typically longer
        raise HTTPException(status_code=400, detail="Invalid API key format")

    return api_key

def check_rate_limit(api_key: str):
    """
    Check if the API key has exceeded the rate limit
    """
    current_time = int(time.time())

    # Clean up old entries
    current_window_start = current_time - RATE_LIMIT_WINDOW
    global request_counts
    request_counts = {k: v for k, v in request_counts.items()
                      if v['timestamp'] > current_window_start}

    # Check if API key has made too many requests
    if api_key in request_counts:
        if request_counts[api_key]['count'] >= RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {RATE_LIMIT} requests per hour."
            )
        else:
            # Increment request count
            request_counts[api_key]['count'] += 1
    else:
        # Initialize request count for this API key
        request_counts[api_key] = {
            'count': 1,
            'timestamp': current_time
        }

def validate_and_track_api_key(api_key: str = Depends(get_api_key)):
    """
    Validate API key and track usage for rate limiting
    """
    # Log the API usage
    logger.info(f"API key validated for request: {api_key[:8]}...")

    # Check rate limit
    check_rate_limit(api_key)

    return api_key

# API endpoint for processing queries with the agent
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, x_api_key: str = Depends(validate_and_track_api_key)):
    """
    Process a query with the BrowseAgent
    """
    try:
        # Temporarily set the API key in environment for the request
        original_api_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = x_api_key

        # Process the query with the agent
        result = process_query_with_agent(
            query=request.query,
            llm_provider=request.llm_provider,
            searx_host=request.searx_host,
            use_searxng=request.use_searxng
        )

        # Restore original API key
        if original_api_key is not None:
            os.environ["OPENROUTER_API_KEY"] = original_api_key
        elif "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]

        return QueryResponse(
            query=result["query"],
            llm_provider=result["llm_provider"],
            result=result["result"],
            success=True
        )
    except Exception as e:
        # Clean up environment in case of error
        if "OPENROUTER_API_KEY" in os.environ:
            del os.environ["OPENROUTER_API_KEY"]
        raise HTTPException(status_code=500, detail=str(e))

# API key validation endpoint
@app.post("/validate-api-key", response_model=APIKeyValidationResponse)
async def validate_api_key(request: APIKeyValidationRequest):
    """
    Validate an OpenRouter API key
    """
    try:
        client = OpenRouterClient(api_key=request.api_key)
        is_valid = client.validate_api_key()

        if is_valid:
            return APIKeyValidationResponse(
                valid=True,
                message="API key is valid and has access to OpenRouter"
            )
        else:
            return APIKeyValidationResponse(
                valid=False,
                message="Invalid API key or no access to OpenRouter"
            )
    except Exception as e:
        return APIKeyValidationResponse(
            valid=False,
            message=f"Error validating API key: {str(e)}"
        )

# Get available models endpoint
@app.get("/models", response_model=ModelListResponse)
async def get_models(x_api_key: str = Depends(validate_and_track_api_key)):
    """
    Get list of available models from OpenRouter
    """
    try:
        # Temporarily set the API key in environment for the request
        original_api_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = x_api_key

        client = OpenRouterClient(api_key=x_api_key)
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
