from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="BrowseAgent API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    llm_provider: str = "openai"
    api_token: str

class QueryResponse(BaseModel):
    response: str
    sources: list

@app.get("/")
async def root():
    return {"message": "BrowseAgent API is running"}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    # This will be implemented in later phases
    return {
        "response": "Query processing will be implemented in Phase 3",
        "sources": []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
