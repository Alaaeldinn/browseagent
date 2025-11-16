from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from agent import BrowseAgent
from llm_config import LLMManager, LLMProvider, get_available_models_info

# Load environment variables
load_dotenv()

app = FastAPI(title="BrowseAgent API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the LLM manager and agent
try:
    llm_manager = LLMManager()
    agent = BrowseAgent(llm_manager=llm_manager)
    agent_available = True
except Exception as e:
    agent_available = False
    agent_error = str(e)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    model: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    model_used: str
    intermediate_steps: List[Dict[str, Any]]

class ModelTestRequest(BaseModel):
    model: Optional[str] = None
    test_query: str = "Hello, how are you?"

class ModelTestResponse(BaseModel):
    success: bool
    model: str
    response: Optional[str] = None
    error: Optional[str] = None
    tokens_used: Optional[int] = None

@app.get("/")
async def root():
    return {"message": "BrowseAgent API is running"}

@app.get("/health")
async def health_check():
    if agent_available:
        return {"status": "healthy", "agent": "available"}
    else:
        return {"status": "degraded", "agent": "unavailable", "error": agent_error}

@app.get("/models")
async def get_available_models():
    """
    Get list of available LLM models
    """
    if not agent_available:
        raise HTTPException(status_code=503, detail="Agent not available")
    
    return {"models": agent.get_available_models()}

@app.get("/models/providers")
async def get_models_by_provider(provider: LLMProvider):
    """
    Get list of models from a specific provider
    
    Args:
        provider: LLM provider to filter by (e.g., "openai", "anthropic")
    """
    if not agent_available:
        raise HTTPException(status_code=503, detail="Agent not available")
    
    try:
        models = agent.get_models_by_provider(provider)
        return {"provider": provider, "models": models}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models/info")
async def get_models_info():
    """
    Get detailed information about all available models
    """
    if not agent_available:
        raise HTTPException(status_code=503, detail="Agent not available")
    
    try:
        return get_available_models_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/current")
async def get_current_model_info():
    """
    Get information about the currently active model
    """
    if not agent_available:
        raise HTTPException(status_code=503, detail="Agent not available")
    
    try:
        model_info = agent.get_model_info()
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/test")
async def test_model(request: ModelTestRequest):
    """
    Test a specific model with a sample query
    """
    if not agent_available:
        raise HTTPException(status_code=503, detail="Agent not available")
    
    try:
        model_to_test = request.model or agent.model_name
        
        # Temporarily set the model if different from current
        if model_to_test != agent.model_name:
            agent.set_model(model_to_test)
        
        # Test the model
        result = agent.test_current_model(request.test_query)
        
        return ModelTestResponse(
            success=result["success"],
            model=result["model"],
            response=result.get("response"),
            error=result.get("error"),
            tokens_used=result.get("tokens_used")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/test/all")
async def test_all_models(test_query: str = "Hello, how are you?"):
    """
    Test all available models with a sample query
    """
    if not agent_available:
        raise HTTPException(status_code=503, detail="Agent not available")
    
    try:
        results = agent.test_all_models(test_query)
        return {"test_query": test_query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query through the BrowseAgent system
    """
    if not agent_available:
        raise HTTPException(status_code=503, detail="Agent not available")
    
    try:
        # Set model if specified
        if request.model:
            agent.set_model(request.model)
        
        # Process the query
        result = agent.run(request.query)
        
        # Extract sources from intermediate steps
        sources = []
        for step in result.get("intermediate_steps", []):
            if hasattr(step, 'observation'):
                # This is a tool execution step
                try:
                    # Parse the observation to extract sources
                    obs = str(step.observation)
                    if "Found" in obs and "search results" in obs:
                        # Extract links from the observation
                        lines = obs.split('\n')
                        for line in lines:
                            if "Link:" in line:
                                link = line.split("Link: ")[1].strip()
                                sources.append({"link": link})
                except:
                    pass
        
        return QueryResponse(
            response=result.get("response", ""),
            sources=sources,
            model_used=result.get("model_used", request.model or agent.model_name),
            intermediate_steps=result.get("intermediate_steps", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/default")
async def set_default_model(model_name: str):
    """
    Set the default model to use
    """
    if not agent_available:
        raise HTTPException(status_code=503, detail="Agent not available")
    
    try:
        if model_name not in agent.get_available_models():
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        success = agent.llm_manager.set_default_model(model_name)
        if success:
            return {"message": f"Default model set to {model_name}", "model": model_name}
        else:
            raise HTTPException(status_code=400, detail=f"Failed to set default model to {model_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
