from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from agent import BrowseAgent

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

# Initialize the agent
try:
    agent = BrowseAgent()
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
            model_used=request.model or agent.model_name,
            intermediate_steps=result.get("intermediate_steps", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
