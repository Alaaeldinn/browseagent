from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os

from agent import BrowseAgent, process_query_with_agent

# Create FastAPI app instance
app = FastAPI(
    title="BrowseAgent API",
    description="An AI research agent that searches, browses, and synthesizes information from the web",
    version="1.0.0"
)

# Define request/response models
class QueryRequest(BaseModel):
    query: str
    llm_provider: Optional[str] = "openai/gpt-3.5-turbo"

class QueryResponse(BaseModel):
    query: str
    llm_provider: str
    result: str
    success: bool
    error: Optional[str] = None

# API endpoint for processing queries with the agent
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a query with the BrowseAgent
    """
    try:
        # Process the query with the agent
        result = process_query_with_agent(
            query=request.query,
            llm_provider=request.llm_provider
        )
        
        return QueryResponse(
            query=result["query"],
            llm_provider=result["llm_provider"],
            result=result["result"],
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
