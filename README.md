# BrowseAgent

BrowseAgent is an AI research agent (Level 2) designed to search, browse, and synthesize information from the web.

It helps you turn any query into structured insights using LLM reasoning and automated search tools.

## Features

- Query processing with keyword extraction
- Tool calling functionality using LangChain
- Web search via DuckDuckGo with semantic result ranking
- Support for multiple LLM providers via LiteLLM
- Simple, practical UI with intentional color scheme
- FastAPI backend with FastHTML frontend

## Requirements

- Python 3.8+
- API keys for LLM providers (optional, for full functionality)

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd browseagent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional):
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   # etc. for other providers you plan to use
   ```

## Usage

1. Start the backend server:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

2. In a separate terminal, start the frontend:
   ```bash
   python frontend.py
   ```

3. Access the application in your browser at `http://localhost:5000`

## API Endpoints

- `POST /query` - Process a query with the BrowseAgent
- `GET /health` - Health check endpoint
- `GET /providers` - List available LLM providers

## Architecture

BrowseAgent uses the following components:
- Backend: FastAPI server with LangChain agent
- Frontend: FastHTML with Tailwind CSS
- Search: DuckDuckGo Search with semantic ranking using Sentence Transformers
- LLM: Multiple providers supported via LiteLLM