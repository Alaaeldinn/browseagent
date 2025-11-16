"""
BrowseAgent Frontend - Simple FastHTML Application

This module implements a clean user interface for BrowseAgent.
"""

from fasthtml.common import *
import asyncio
import aiohttp
import os

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
APP_TITLE = "BrowseAgent"
APP_DESCRIPTION = "AI-Powered Research Assistant"

# Create FastHTML app
app = FastHTML()

async def call_api(endpoint: str, method: str = "GET", data: dict = None) -> dict:
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

def render_header():
    """Render the page header"""
    return Header(
        Div(
            H1(APP_TITLE),
            P(APP_DESCRIPTION, cls="subtitle"),
            cls="header-content"
        ),
        cls="header"
    )

def render_query_form():
    """Render the query input form"""
    return Div(
        Div(
            Label("Your Research Query", cls="form-group"),
            Textarea(
                name="query",
                placeholder="Enter your research question here...",
                id="query-input",
                cls="query-input"
            ),
            cls="form-group"
        ),
        Div(
            Label("Select Model", cls="form-group"),
            Select(
                name="model",
                id="model-select",
                cls="model-select"
            ),
            cls="form-group"
        ),
        Div(
            Button(
                "Research",
                id="submit-btn",
                cls="btn-primary",
                onclick="submitQuery()"
            ),
            Button(
                "Clear",
                cls="btn-secondary",
                onclick="clearForm()"
            ),
            cls="button-group"
        ),
        cls="query-section"
    )

def render_results():
    """Render the results section"""
    return Div(
        Div(
            Div(
                Div("Processing your query...", cls="loading-text"),
                cls="loading-spinner",
                id="loading-spinner"
            ),
            Div(
                cls="error-message",
                id="error-message"
            ),
            Div(
                cls="success-message",
                id="success-message"
            ),
            Div(
                Div(
                    Div("Research Results", cls="result-title"),
                    Div(cls="result-content", id="result-content"),
                    cls="result-card"
                ),
                Div(
                    Div("Sources", cls="sources-title"),
                    Div(cls="sources-list", id="sources-list"),
                    cls="sources-section"
                ),
                Div(
                    Div("Processing Steps", cls="steps-title"),
                    Div(cls="steps-list", id="steps-list"),
                    cls="steps-section"
                ),
                cls="results-section",
                id="results-section"
            )
        ),
        cls="results-container"
    )

@app.get("/")
async def homepage():
    """Render the main page"""
    # Get available models
    models_response = await call_api("/models")
    models = models_response.get("models", [])
    
    return Title(APP_TITLE), Div(
        render_header(),
        Main(
            render_query_form(),
            render_results(),
            cls="container"
        ),
        Style("""
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #FFF8DC;
                color: #2F4F4F;
                line-height: 1.6;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                background-color: #8B4513;
                color: white;
                padding: 2rem 0;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            
            .header-content h1 {
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
            }
            
            .subtitle {
                font-size: 1.1rem;
                opacity: 0.9;
            }
            
            .query-section {
                background-color: #FFFAF0;
                border-radius: 12px;
                padding: 2rem;
                margin-bottom: 2rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.05);
                border: 1px solid #DEB887;
            }
            
            .form-group {
                margin-bottom: 1.5rem;
            }
            
            label {
                display: block;
                margin-bottom: 0.5rem;
                font-weight: 600;
                color: #2F4F4F;
            }
            
            .query-input, .model-select {
                width: 100%;
                padding: 12px 16px;
                border: 2px solid #DEB887;
                border-radius: 8px;
                font-size: 1rem;
                transition: all 0.3s ease;
                background-color: white;
            }
            
            .query-input:focus, .model-select:focus {
                outline: none;
                border-color: #FF8C00;
                box-shadow: 0 0 0 3px rgba(255, 140, 0, 0.1);
            }
            
            .button-group {
                display: flex;
                gap: 1rem;
            }
            
            button {
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .btn-primary {
                background-color: #FF8C00;
                color: white;
            }
            
            .btn-primary:hover {
                background-color: #FF7F00;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(255, 140, 0, 0.3);
            }
            
            .btn-primary:disabled {
                background-color: #696969;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            
            .btn-secondary {
                background-color: #D2691E;
                color: white;
            }
            
            .btn-secondary:hover {
                background-color: #CD853F;
            }
            
            .results-container {
                display: none;
            }
            
            .results-section {
                background-color: #FFFAF0;
                border-radius: 12px;
                padding: 2rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.05);
                border: 1px solid #DEB887;
            }
            
            .result-card {
                background-color: white;
                border-radius: 8px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                border: 1px solid #DEB887;
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            }
            
            .result-title {
                font-size: 1.25rem;
                font-weight: 600;
                color: #8B4513;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid #DEB887;
            }
            
            .result-content {
                line-height: 1.7;
            }
            
            .sources-title, .steps-title {
                font-size: 1.1rem;
                font-weight: 600;
                margin-bottom: 1rem;
                color: #8B4513;
            }
            
            .source-item {
                background-color: white;
                border-radius: 6px;
                padding: 1rem;
                margin-bottom: 0.75rem;
                border: 1px solid #DEB887;
                transition: all 0.3s ease;
            }
            
            .source-item:hover {
                border-color: #FF8C00;
                box-shadow: 0 2px 8px rgba(255, 140, 0, 0.1);
            }
            
            .source-title {
                font-weight: 600;
                color: #8B4513;
                margin-bottom: 0.25rem;
            }
            
            .source-link {
                color: #FF8C00;
                text-decoration: none;
                font-size: 0.875rem;
            }
            
            .source-link:hover {
                text-decoration: underline;
            }
            
            .step {
                background-color: white;
                border-radius: 6px;
                padding: 1rem;
                margin-bottom: 0.75rem;
                border-left: 4px solid #FF8C00;
            }
            
            .step-title {
                font-weight: 600;
                color: #8B4513;
                margin-bottom: 0.5rem;
            }
            
            .step-content {
                font-size: 0.875rem;
                color: #696969;
            }
            
            .loading-spinner {
                display: none;
                text-align: center;
                padding: 2rem;
            }
            
            .loading-text {
                color: #696969;
                font-size: 1.1rem;
            }
            
            .error-message, .success-message {
                padding: 1rem;
                border-radius: 8px;
                margin-bottom: 1rem;
                display: none;
            }
            
            .error-message {
                background-color: #FEE;
                color: #DC143C;
                border: 1px solid #DC143C;
            }
            
            .success-message {
                background-color: #E8F5E9;
                color: #32CD32;
                border: 1px solid #32CD32;
            }
            
            @media (max-width: 768px) {
                .button-group {
                    flex-direction: column;
                }
                
                button {
                    width: 100%;
                }
            }
        """),
        Script("""
            async function submitQuery() {
                const query = document.getElementById('query-input').value;
                const model = document.getElementById('model-select').value;
                const submitBtn = document.getElementById('submit-btn');
                const loadingSpinner = document.getElementById('loading-spinner');
                const errorMessage = document.getElementById('error-message');
                const successMessage = document.getElementById('success-message');
                const resultsSection = document.getElementById('results-section');
                
                if (!query.trim()) {
                    alert('Please enter a query');
                    return;
                }
                
                // Show loading state
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Researching...';
                loadingSpinner.style.display = 'block';
                errorMessage.style.display = 'none';
                successMessage.style.display = 'none';
                resultsSection.style.display = 'none';
                
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
                        // Display results
                        displayResults(data);
                        
                        // Show success message
                        successMessage.textContent = 'Research completed successfully!';
                        successMessage.style.display = 'block';
                        
                        // Show results section
                        resultsSection.style.display = 'block';
                    } else {
                        throw new Error(data.detail || 'Request failed');
                    }
                } catch (error) {
                    errorMessage.textContent = error.message;
                    errorMessage.style.display = 'block';
                } finally {
                    // Hide loading state
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = '<i class="fas fa-search"></i> Research';
                    loadingSpinner.style.display = 'none';
                }
            }
            
            function displayResults(data) {
                // Display response
                document.getElementById('result-content').innerHTML = data.response || 'No response received.';
                
                // Display sources
                const sourcesList = document.getElementById('sources-list');
                sourcesList.innerHTML = '';
                
                if (data.sources && data.sources.length > 0) {
                    data.sources.forEach(source => {
                        const sourceItem = document.createElement('div');
                        sourceItem.className = 'source-item';
                        sourceItem.innerHTML = `
                            <div class="source-title">${source.title || 'Source'}</div>
                            <a href="${source.link || '#'}" target="_blank" class="source-link">View Source</a>
                        `;
                        sourcesList.appendChild(sourceItem);
                    });
                } else {
                    sourcesList.innerHTML = '<p>No sources found.</p>';
                }
                
                // Display processing steps
                const stepsList = document.getElementById('steps-list');
                stepsList.innerHTML = '';
                
                if (data.intermediate_steps && data.intermediate_steps.length > 0) {
                    data.intermediate_steps.forEach((step, index) => {
                        const stepItem = document.createElement('div');
                        stepItem.className = 'step';
                        stepItem.innerHTML = `
                            <div class="step-title">Step ${index + 1}</div>
                            <div class="step-content">${step || 'No step details'}</div>
                        `;
                        stepsList.appendChild(stepItem);
                    });
                } else {
                    stepsList.innerHTML = '<p>No processing steps recorded.</p>';
                }
            }
            
            function clearForm() {
                document.getElementById('query-input').value = '';
                document.getElementById('model-select').value = '';
                document.getElementById('results-section').style.display = 'none';
                document.getElementById('error-message').style.display = 'none';
                document.getElementById('success-message').style.display = 'none';
            }
            
            // Load models on page load
            document.addEventListener('DOMContentLoaded', async function() {
                try {
                    const response = await fetch('/models');
                    const data = await response.json();
                    
                    const modelSelect = document.getElementById('model-select');
                    modelSelect.innerHTML = '';
                    
                    if (data.models && data.models.length > 0) {
                        data.models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model;
                            option.textContent = model;
                            modelSelect.appendChild(option);
                        });
                    } else {
                        modelSelect.innerHTML = '<option value="">No models available</option>';
                    }
                } catch (error) {
                    console.error('Failed to load models:', error);
                    document.getElementById('model-select').innerHTML = '<option value="">Error loading models</option>';
                }
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
