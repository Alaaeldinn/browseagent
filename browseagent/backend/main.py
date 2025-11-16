from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

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

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    model: str = "gpt-3.5-turbo"

class QueryResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    model_used: str

@app.get("/")
async def root():
    return {"message": "BrowseAgent API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query through the BrowseAgent system
    """
    try:
        # TODO: Implement actual agent logic here
        # For now, return a placeholder response
        return QueryResponse(
            response=f"Processing query: {request.query}",
            sources=[],
            model_used=request.model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
