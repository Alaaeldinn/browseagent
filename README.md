# BrowseAgent - AI Research Assistant

## Overview

BrowseAgent is a I research assistant that combines ai agent  with real-time web search capabilities (SearXNG) and semantic filtering to provide comprehensive, accurate, and up-to-date research answers.


**This project was 100% vibe coded**

## Features

- **Real-time Web Search**: Integrated with SearXNG for up-to-date information
- **Ollama Integration**: Leverages local LLMs for processing (llama3.2)
- **Semantic Filtering**: Uses Model2Vec to rank search results by relevance
- **Keyword Optimization**: AI-generated search keywords for better results
- **API Access**: RESTful API endpoint for programmatic access
- **MCP Protocol Support**: Model Context Protocol server for AI integration
- **Comprehensive Answers**: Generates detailed responses from multiple sources

## Architecture

BrowseAgent follows a modular architecture with:

- **FastAPI Backend**: Robust API server with CORS support
- **Research Agent**: Core AI research functionality with semantic processing
- **Search Tools**: SearXNG integration for web search
- **Model2Vec Integration**: Fast semantic similarity matching
- **MCP Server**: Model Context Protocol wrapper for broader AI integration

## Installation

### Prerequisites

- Python 3.10
- Ollama installed and running
- SearXNG instance running

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd browseagent
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama model** (if not already installed):
   ```bash
   ollama pull llama3.2
   ```

5. **Set up SearXNG** (if not already running):
   - Follow SearXNG installation guide to set up your instance on port 4000
   - Ensure the search engine is accessible at http://localhost:4000



## Future Tasks

- [ ] **Optimize Search Accuracy, Template, and Latency**
  - [ ] Implement advanced query reformulation techniques
  - [ ] Optimize semantic similarity algorithms
  - [ ] Create standardized answer templates
  - [ ] Reduce overall response latency through caching and optimization
  - [ ] Fine-tune Model2Vec for domain-specific searches

- [ ] **Add Agentic Features**
  - [ ] Implement multi-step reasoning capabilities
  - [ ] Add memory and context persistence
  - [ ] Create research task planning functionality
  - [ ] Implement verification and fact-checking steps
  - [ ] Add source credibility assessment

- [ ] **Add Deep Agent**
  - [ ] Build a multi-agent system for complex research tasks
  - [ ] Implement specialized agents for different domains
  - [ ] Create agent collaboration and task delegation
  - [ ] Add autonomous research workflows
  - [ ] Implement agent learning from feedback

- [ ] **Add Research Focus**
  - [ ] Create specialized research modes (academic, news, technical)
  - [ ] Implement citation generation and management
  - [ ] Add research paper parsing capabilities
#