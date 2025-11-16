"""
BrowseAgent Frontend - Simple Web Interface

This module implements a basic web interface for BrowseAgent using FastAPI.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import aiohttp
import os
from typing import Dict, List, Any

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
APP_TITLE = "BrowseAgent"
APP_DESCRIPTION = "AI-Powered Research Assistant"

# Create FastAPI app
app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION)

# Templates setup
templates = Jinja2Templates(directory=".")

# Color scheme with warmer tones
COLORS = {
    "primary": "#8B4513",  # Saddle brown
    "secondary": "#D2691E",  # Chocolate
    "accent": "#FF8C00",  # Dark orange
    "background": "#FFF8DC",  # Cornsilk
    "card": "#FFFAF0",  # Floral white
    "text": "#2F4F4F",  # Dark slate gray
    "text_light": "#696969",  # Dim gray
    "border": "#DEB887",  # Burlywood
    "success": "#32CD32",  # Lime green
    "error": "#DC143C",  # Crimson
}

async def call_api(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API call to backend"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url) as response:
                    return await response.json()
            elif method == "POST":
                async with session.post(url, json=data) as response:
                    return await response.json()
    except Exception as e:
        return {"error": f"API call failed: {str(e)}"}

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    """Render the main page"""
    # Get available models
    models_response = await call_api("/models")
    models = models_response.get("models", [])
    
    context = {
        "request": request,
        "title": APP_TITLE,
        "description": APP_DESCRIPTION,
        "models": models,
        "colors": COLORS
    }
    
    return templates.TemplateResponse("index.html", context)

@app.post("/query")
async def process_query(request: Request):
    """Process a query through the backend API"""
    try:
        form_data = await request.form()
        query = form_data.get("query", "")
        model = form_data.get("model", "")
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query is required")
        
        # Call the backend API
        response = await call_api("/query", "POST", {"query": query, "model": model})
        
        if "error" in response:
            raise HTTPException(status_code=500, detail=response["error"])
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
