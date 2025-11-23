from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import SearxSearchWrapper
from sentence_transformers import SentenceTransformer, util
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BrowseAgent API",
    description="AI-powered search assistant with OpenRouter and SearXNG integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class QueryRequest(BaseModel):
    query: str
    api_key: str
    model: str = "openrouter/gpt-3.5-turbo"

class ValidateApiKeyRequest(BaseModel):
    api_key: str
    model: str = "openrouter/gpt-3.5-turbo"

class StreamResponse(BaseModel):
    event_type: str  # 'start', 'token', 'end', 'error'
    content: str
    metadata: Optional[Dict[str, Any]] = None

# Global model for semantic similarity
semantic_model = None

def get_semantic_model():
    global semantic_model
    if semantic_model is None:
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    return semantic_model

@tool("web_search")
def web_search(query: str) -> str:
    """Search the web for current information using SearXNG."""
    try:
        # Use a public SearXNG instance
        searx = SearxSearchWrapper(
            searx_host="https://searx.space",
            engines=["google", "bing", "duckduckgo"]  # Multiple engines for better results
        )
        
        # Get raw results
        raw_results = searx.results(query, num_results=10)  # Get more results for better filtering
        
        # Apply semantic similarity
        processed_results = apply_semantic_similarity(query, raw_results)
        
        # Return top 5 results
        top_results = processed_results[:5]
        
        # Format results
        formatted_results = []
        for result in top_results:
            formatted_results.append({
                "title": result.get("title", ""),
                "url": result.get("url", result.get("href", "")),
                "content": result.get("content", result.get("body", "")),
                "score": result.get("similarity_score", 0)
            })
        
        return str(formatted_results)
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return f"Error performing search: {str(e)}"

def apply_semantic_similarity(query: str, results: List[Dict]) -> List[Dict]:
    """Apply semantic similarity to rank search results."""
    if not results:
        return []
    
    model = get_semantic_model()
    
    # Encode the query
    query_embedding = model.encode([query])
    
    # Prepare result texts for encoding
    result_texts = []
    for result in results:
        content = result.get('content', '') or result.get('body', '') or result.get('title', '')
        result_texts.append(content)
    
    # Encode all result texts
    result_embeddings = model.encode(result_texts)
    
    # Calculate cosine similarity between query and results
    similarities = util.cos_sim(query_embedding, result_embeddings)[0]
    
    # Add similarity scores to results
    for i, result in enumerate(results):
        result['similarity_score'] = similarities[i].item()
    
    # Sort results by similarity score in descending order
    sorted_results = sorted(results, key=lambda x: x.get('similarity_score', 0), reverse=True)
    
    return sorted_results

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Process query with AI agent using OpenRouter and search tools."""
    try:
        # Validate API key by making a simple request
        if not request.api_key or len(request.api_key) < 10:
            raise HTTPException(status_code=400, detail="Invalid API key")
        
        # Set the API key for this request
        os.environ["OPENROUTER_API_KEY"] = request.api_key
        
        # Use the model name as provided - OpenRouter models can have various formats
        model_name = request.model
        
        # Create ChatOpenAI instance with OpenRouter
        llm = ChatOpenAI(
            model=model_name,
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=request.api_key,
            default_headers={
                "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:3000"),
                "X-Title": os.getenv("APP_NAME", "BrowseAgent")
            },
            timeout=30
        )
        
        # Create the agent with search tool
        tools = [web_search]
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that can search the web for information. Use the web_search tool to find current information. Always provide specific, factual answers based on search results."),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Execute the agent
        result = agent_executor.invoke({"input": request.query})
        
        # Ensure result is properly formatted
        response_content = result.get("output", str(result)) if isinstance(result, dict) else str(result)
        
        return {
            "query": request.query,
            "result": response_content,
            "model": request.model,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/validate-api-key")
async def validate_api_key(request: ValidateApiKeyRequest):
    """Validate OpenRouter API key."""
    try:
        # Test the API key with a simple model call
        os.environ["OPENROUTER_API_KEY"] = request.api_key

        llm = ChatOpenAI(
            model="openai/gpt-3.5-turbo",
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=request.api_key,
            default_headers={
                "HTTP-Referer": os.getenv("SITE_URL", "http://localhost:3000"),
                "X-Title": os.getenv("APP_NAME", "BrowseAgent")
            },
            request_timeout=10
        )

        # Try a simple completion to validate the key
        test_msg = [HumanMessage(content="Hello")]
        llm.invoke(test_msg)

        return {"valid": True, "message": "API key is valid"}

    except Exception as e:
        logger.error(f"API key validation error: {str(e)}")
        return {"valid": False, "message": f"Invalid API key: {str(e)}"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "BrowseAgent API is running"}

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to BrowseAgent API",
        "description": "AI-powered search assistant with OpenRouter and SearXNG integration",
        "endpoints": [
            {"method": "POST", "path": "/query", "description": "Process a query with search capabilities"},
            {"method": "POST", "path": "/validate-api-key", "description": "Validate an API key"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)