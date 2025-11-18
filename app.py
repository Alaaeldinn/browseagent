from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

app = FastAPI(title="BrowseAgent", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    llm_provider: str = "openai/gpt-3.5-turbo"

@app.get("/")
def read_root():
    return {"message": "Welcome to BrowseAgent - AI Research Agent"}

@app.post("/search")
async def search_endpoint(request: QueryRequest):
    """
    Main endpoint to process user queries through the AI agent
    """
    # This will be implemented in later phases
    return {"query": request.query, "llm_provider": request.llm_provider, "status": "processing"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
