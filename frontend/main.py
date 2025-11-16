"""
BrowseAgent Frontend - FastHTML Application

This module implements the user interface for BrowseAgent, providing:
1. User query input interface
2. Results display with warmer color scheme
3. Agent interaction flow
4. LLM selection interface
5. Loading states and error handling
"""

from fasthtml.common import *
from fasthtml.jupyter import *
import asyncio
import json
import aiohttp
from typing import Dict, List, Any, Optional
import os

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
APP_TITLE = "BrowseAgent"
APP_DESCRIPTION = "AI-Powered Research Assistant"

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
    "loading": "#FFA500",  # Orange
}

# Create FastHTML app
app = FastHTML(hdrs=(
    Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"),
    Style(f"""
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: {COLORS['background']};
            color: {COLORS['text']};
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            background-color: {COLORS['primary']};
            color: white;
            padding: 1.5rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }}
        
        .header-content {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
        }}
        
        .subtitle {{
            font-size: 1.1rem;
            opacity: 0.9;
            margin-top: 0.5rem;
        }}
        
        .query-section {{
            background-color: {COLORS['card']};
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border: 1px solid {COLORS['border']};
        }}
        
        .form-group {{
            margin-bottom: 1.5rem;
        }}
        
        label {{
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
            color: {COLORS['text']};
        }}
        
        input[type="text"],
        textarea,
        select {{
            width: 100%;
            padding: 12px 16px;
            border: 2px solid {COLORS['border']};
            border-radius: 8px;
            font-size: 1rem;
            transition: all 0.3s ease;
            background-color: white;
        }}
        
        input[type="text"]:focus,
        textarea:focus,
        select:focus {{
            outline: none;
            border-color: {COLORS['accent']};
            box-shadow: 0 0 0 3px rgba(255, 140, 0, 0.1);
        }}
        
        textarea {{
            resize: vertical;
            min-height: 120px;
        }}
        
        .button-group {{
            display: flex;
            gap: 1rem;
            align-items: center;
        }}
        
        button {{
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .btn-primary {{
            background-color: {COLORS['accent']};
            color: white;
        }}
        
        .btn-primary:hover {{
            background-color: #FF7F00;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 140, 0, 0.3);
        }}
        
        .btn-primary:disabled {{
            background-color: {COLORS['text_light']};
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }}
        
        .btn-secondary {{
            background-color: {COLORS['secondary']};
            color: white;
        }}
        
        .btn-secondary:hover {{
            background-color: #CD853F;
        }}
        
        .results-section {{
            background-color: {COLORS['card']};
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            border: 1px solid {COLORS['border']};
            display: none;
        }}
        
        .results-section.show {{
            display: block;
            animation: fadeIn 0.5s ease;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .result-card {{
            background-color: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid {COLORS['border']};
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }}
        
        .result-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid {COLORS['border']};
        }}
        
        .result-title {{
            font-size: 1.25rem;
            font-weight: 600;
            color: {COLORS['primary']};
        }}
        
        .result-meta {{
            display: flex;
            gap: 1rem;
            font-size: 0.875rem;
            color: {COLORS['text_light']};
        }}
        
        .result-content {{
            line-height: 1.7;
        }}
        
        .sources-section {{
            margin-top: 2rem;
        }}
        
        .sources-title {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: {COLORS['primary']};
        }}
        
        .source-item {{
            background-color: white;
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            border: 1px solid {COLORS['border']};
            transition: all 0.3s ease;
        }}
        
        .source-item:hover {{
            border-color: {COLORS['accent']};
            box-shadow: 0 2px 8px rgba(255, 140, 0, 0.1);
        }}
        
        .source-title {{
            font-weight: 600;
            color: {COLORS['primary']};
            margin-bottom: 0.25rem;
        }}
        
        .source-link {{
            color: {COLORS['accent']};
            text-decoration: none;
            font-size: 0.875rem;
        }}
        
        .source-link:hover {{
            text-decoration: underline;
        }}
        
        .loading-spinner {{
            display: none;
            text-align: center;
            padding: 2rem;
        }}
        
        .spinner {{
            border: 4px solid {COLORS['border']};
            border-top: 4px solid {COLORS['loading']};
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .loading-text {{
            color: {COLORS['text_light']};
            font-size: 1.1rem;
        }}
        
        .error-message {{
            background-color: #FEE;
            color: {COLORS['error']};
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border: 1px solid {COLORS['error']};
            display: none;
        }}
        
        .error-message.show {{
            display: block;
        }}
        
        .success-message {{
            background-color: #E8F5E9;
            color: {COLORS['success']};
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            border: 1px solid {COLORS['success']};
            display: none;
        }}
        
        .success-message.show {{
            display: block;
        }}
        
        .model-info {{
            background-color: white;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
            color: {COLORS['text_light']};
            border: 1px solid {COLORS['border']};
        }}
        
        .model-badge {{
            background-color: {COLORS['accent']};
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .intermediate-steps {{
            margin-top: 1.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid {COLORS['border']};
        }}
        
        .step {{
            background-color: white;
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-left: 4px solid {COLORS['accent']};
        }}
        
        .step-title {{
            font-weight: 600;
            color: {COLORS['primary']};
            margin-bottom: 0.5rem;
        }}
        
        .step-content {{
            font-size: 0.875rem;
            color: {COLORS['text_light']};
        }}
        
        @media (max-width: 768px) {{
            .header-content {{
                flex-direction: column;
                text-align: center;
                gap: 1rem;
            }}
            
            h1 {{
                font-size: 2rem;
            }}
            
            .button-group {{
                flex-direction: column;
            }}
            
            button {{
                width: 100%;
                justify-content: center;
            }}
        }}
    """)
))

# Store session data
session_data = {}

def get_session_data(session_id: str) -> Dict[str, Any]:
    """Get or create session data"""
    if session_id not in session_data:
        session_data[session_id] = {
            "query": "",
            "results": None,
            "loading": False,
            "error": None,
            "model": None
        }
    return session_data[session_id]

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

def render_query_form(session_id: str) -> Div:
    """Render the query input form"""
    session = get_session_data(session_id)
    
    return Div(
        Div(
            Label("Your Research Query", cls="form-group"),
            Textarea(
                name="query",
                placeholder="Enter your research question here...",
                value=session["query"],
                id="query-input"
            ),
            cls="form-group"
        ),
        Div(
            Label("Select Model", cls="form-group"),
            Select(
                name="model",
                id="model-select",
                cls="form-group"
            ),
            cls="form-group"
        ),
        Div(
            Button(
                "Research",
                cls="btn-primary",
                id="submit-btn",
                type="button",
                onclick=f"submitQuery('{session_id}')"
            ),
            Button(
                "Clear",
                cls="btn-secondary",
                onclick=f"clearForm('{session_id}')"
            ),
            Div(
                cls="model-info",
                id="model-info"
            ),
            cls="button-group"
        ),
        cls="query-section"
    )

def render_results(session_id: str) -> Div:
    """Render the results section"""
    session = get_session_data(session_id)
    
    if not session["results"]:
        return Div()
    
    results = session["results"]
    
    return Div(
        Div(
            Div(
                cls="loading-spinner",
                id=f"loading-{session_id}",
                Div(cls="spinner"),
                Div("Processing your query...", cls="loading-text")
            ),
            Div(
                cls="error-message",
                id=f"error-{session_id}"
            ),
            Div(
                cls="success-message",
                id=f"success-{session_id}"
            ),
            Div(
                cls="result-card",
                Div(
                    Div(
                        Div(results.get("response", ""), cls="result-content"),
                        cls="result-body"
                    ),
                    cls="result-card-inner"
                ),
                Div(
                    Div(
                        Div("Sources", cls="sources-title"),
                        *[Div(
                            Div(
                                Div(source.get("title", f"Source {i+1}"), cls="source-title"),
                                A(
                                    source.get("link", ""),
                                    href=source.get("link", ""),
                                    target="_blank",
                                    cls="source-link"
                                )
                            ) for i, source in enumerate(results.get("sources", []))
                        ], cls="sources-list"
                    ) if results.get("sources") else Div(),
                    cls="sources-section"
                ),
                Div(
                    Div(
                        Div("Processing Steps", cls="intermediate-title"),
                        *[Div(
                            Div(
                                Div(f"Step {i+1}", cls="step-title"),
                                Div(str(step), cls="step-content")
                            ) for i, step in enumerate(results.get("intermediate_steps", []))
                        ], cls="steps-list"
                    ) if results.get("intermediate_steps") else Div(),
                    cls="intermediate-steps"
                ),
                cls="results-section show",
                id=f"results-{session_id}"
            )
        )
    )

@app.get("/")
async def homepage():
    """Render the main page"""
    session_id = "default"
    
    # Get available models
    models_response = await call_api("/models")
    models = models_response.get("models", [])
    
    # Set default model if available
    if models:
        get_session_data(session_id)["model"] = models[0]
    
    return Title(APP_TITLE), Div(
        Header(
            Div(
                H1(APP_TITLE),
                P(APP_DESCRIPTION, cls="subtitle"),
                cls="header-content"
            )
        ),
        Main(
            render_query_form(session_id),
            render_results(session_id),
            cls="container"
        ),
        Script("""
            async function submitQuery(sessionId) {
                const query = document.getElementById('query-input').value;
                const model = document.getElementById('model-select').value;
                const submitBtn = document.getElementById('submit-btn');
                const loadingSpinner = document.getElementById(`loading-${sessionId}`);
                const errorMessage = document.getElementById(`error-${sessionId}`);
                const successMessage = document.getElementById(`success-${sessionId}`);
                const resultsSection = document.getElementById(`results-${sessionId}`);
                
                if (!query.trim()) {
                    alert('Please enter a query');
                    return;
                }
                
                // Show loading state
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Researching...';
                loadingSpinner.style.display = 'block';
                errorMessage.classList.remove('show');
                successMessage.classList.remove('show');
                resultsSection.classList.remove('show');
                
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            query: query,
                            model: model
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        // Store results
                        sessionStorage.setItem(`results-${sessionId}`, JSON.stringify(data));
                        sessionStorage.setItem(`query-${sessionId}`, query);
                        sessionStorage.setItem(`model-${sessionId}`, model);
                        
                        // Show success message
                        successMessage.textContent = 'Research completed successfully!';
                        successMessage.classList.add('show');
                        
                        // Update results
                        location.reload();
                    } else {
                        throw new Error(data.detail || 'Request failed');
                    }
                } catch (error) {
                    errorMessage.textContent = error.message;
                    errorMessage.classList.add('show');
                } finally {
                    // Hide loading state
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-search"></i> Research';
                    loadingSpinner.style.display = 'none';
                }
            }
            
            async function loadModels() {
                try {
                    const response = await fetch('/models');
                    const data = await response.json();
                    
                    const modelSelect = document.getElementById('model-select');
                    modelSelect.innerHTML = '';
                    
                    data.models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.textContent = model;
                        modelSelect.appendChild(option);
                    });
                    
                    // Set default model
                    const defaultModel = sessionStorage.getItem('model-default');
                    if (defaultModel && data.models.includes(defaultModel)) {
                        modelSelect.value = defaultModel;
                    }
                } catch (error) {
                    console.error('Failed to load models:', error);
                }
            }
            
            function clearForm(sessionId) {
                document.getElementById('query-input').value = '';
                document.getElementById('model-select').value = '';
                sessionStorage.removeItem(`query-${sessionId}`);
                sessionStorage.removeItem(`results-${sessionId}`);
                sessionStorage.removeItem(`model-${sessionId}`);
                
                const resultsSection = document.getElementById(`results-${sessionId}`);
                resultsSection.classList.remove('show');
            }
            
            // Initialize on page load
            document.addEventListener('DOMContentLoaded', function() {
                const sessionId = 'default';
                
                // Load saved data
                const savedQuery = sessionStorage.getItem(`query-${sessionId}`);
                const savedResults = sessionStorage.getItem(`results-${sessionId}`);
                const savedModel = sessionStorage.getItem(`model-${sessionId}`);
                
                if (savedQuery) {
                    document.getElementById('query-input').value = savedQuery;
                }
                
                if (savedModel) {
                    document.getElementById('model-select').value = savedModel;
                }
                
                if (savedResults) {
                    // Results will be rendered by the server
                }
                
                // Load models
                loadModels();
            });
        """),
        cls="min-h-screen"
    )

@app.post("/query")
async def process_query(query: str, model: str = None):
    """Process a query through the backend API"""
    try:
        # Call the backend API
        response = await call_api("/query", "POST", {"query": query, "model": model})
        
        if "error" in response:
            raise Exception(response["error"])
        
        return response
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
