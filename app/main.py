from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.agent import agent, ResearchResponse

app = FastAPI(title="BrowseAgent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str

@app.post("/api/chat", response_model=ResearchResponse)
async def chat(request: ChatRequest):
    try:
        response = agent.run(request.prompt)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import os
# Get the project root directory (one level up from app)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
static_dir = os.path.join(project_root, "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
