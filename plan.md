# BrowseAgent Development Plan (LangChain Agent Level 2)

## Project Overview
BrowseAgent is an AI-powered research agent built as a LangChain Agent Level 2 (Tool Calling) that helps users conduct research efficiently by:
1. Taking user queries
2. Using LLMs to extract and optimize keywords
3. Using LangChain's agent framework to call custom search tools
4. Applying semantic search to filter and rank results (without vector database storage)
5. Synthesizing results into structured insights

## Technology Stack
- **Frontend**: FastHTML
- **Backend**: FastAPI
- **AI Framework**: LangChain (Agent Level 2 - Tool Calling)
- **LLM Integration**: LiteLLM for multiple LLM support
- **Search Tool**: Custom LangChain tool based on scrape.py (DDGS)
- **Semantic Search**: In-memory similarity matching using embeddings
- **Testing**: Unit tests for business logic

## Development Phases

### Phase 1: Project Setup & LangChain Foundation
- [ ] Initialize project structure
- [ ] Set up virtual environment and dependencies (FastAPI, FastHTML, LangChain, LiteLLM, sentence-transformers)
- [ ] Configure FastAPI backend
- [ ] Set up FastHTML frontend
- [ ] Implement basic routing
- [ ] Set up Git with descriptive commit messages
- [ ] Create LangChain agent initialization

### Phase 2: Custom Search Tool Development
- [ ] Refactor scrape.py into a proper LangChain tool
  - Create `DDGSSearchTool` class inheriting from `BaseTool`
  - Implement `_run` and `_arun` methods
  - Add tool description and parameters
- [ ] Integrate custom search tool with LangChain agent
- [ ] Test tool functionality independently

### Phase 3: Semantic Search Implementation
- [ ] Create internal semantic search function (langchain)
  - Initialize embedding model (e.g., sentence-transformers)
  - Implement query embedding generation
  - Create function to embed search results
  - Implement similarity scoring and ranking
  - Return top 5 most relevant results
- [ ] Integrate semantic search into the agent workflow
- [ ] Test semantic search accuracy

### Phase 4: LangChain Agent Implementation
- [ ] Set up LangChain Agent with tool calling capabilities
- [ ] Create agent with custom search tool
- [ ] Implement prompt engineering for research tasks
- [ ] Add memory/context management
- [ ] Test agent with various queries

### Phase 5: LLM Integration & Configuration
- [ ] Set up LiteLLM integration for multiple LLMs
- [ ] Create API key management system
- [ ] Configure agent to use different LLMs
- [ ] Implement LLM selection logic
- [ ] Test agent with different LLM backends

### Phase 6: Frontend Implementation
- [ ] Create user query input interface
- [ ] Design results display with warmer color scheme
- [ ] Implement agent interaction flow
- [ ] Add LLM selection interface
- [ ] Style with browser/AI browser inspiration
- [ ] Add loading states and error handling

### Phase 7: Testing & Quality Assurance
- [ ] Write unit tests for business logic
- [ ] Test agent tool calling functionality
- [ ] Test search tool integration
- [ ] Test semantic search accuracy
- [ ] Add integration tests for complete flow
- [ ] Performance testing

### Phase 8: Deployment & Polish
- [ ] Set up environment configurations
- [ ] Implement proper logging
- [ ] Add documentation
- [ ] Final UI polish
- [ ] Deployment preparation

## Key Components to Develop

### LangChain Agent Components
1. **Custom Search Tool**
   ```python
   class DDGSSearchTool(BaseTool):
       name = "ddgs_search"
       description = "Useful for searching the web for current information"
       
       def _run(self, query: str) -> str:
           # Implementation using DDGS
           pass
       
       def _arun(self, query: str) -> str:
           # Async implementation
           pass
   ```

2. **Semantic Search Function**
   ```python
   class SemanticSearch:
       def __init__(self, embedding_model="all-MiniLM-L6-v2"):
           self.embedding_model = SentenceTransformer(embedding_model)
       
       def rank_results(self, query: str, search_results: List[Dict]) -> List[Dict]:
           # Generate query embedding
           # Embed search results
           # Calculate similarities
           # Return top 5 results
           pass
   ```

3. **Agent Configuration**
   ```python
   tools = [DDGSSearchTool()]
   semantic_search = SemanticSearch()
   agent = create_openai_tools_agent(
       llm=llm,
       tools=tools,
       prompt=prompt_template
   )
   ```

4. **Agent Executor**
   ```python
   agent_executor = AgentExecutor(
       agent=agent,
       tools=tools,
       verbose=True,
       max_iterations=5
   )
   ```

### Backend Components
1. **API Endpoints**
   - `/query` - Process user queries through agent
   - `/llms` - List available LLMs
   - `/health` - Health check endpoint

2. **Agent Service**
   - Initialize and configure LangChain agent
   - Handle query processing
   - Manage tool execution
   - Apply semantic search to results

### Frontend Components
1. **Query Interface**
   - Text input for user queries
   - LLM selection dropdown
   - Submit button

2. **Results Display**
   - Show agent reasoning process
   - Display search results
   - Display semantic search filtering
   - Display final synthesized response
   - Loading indicators

## Open Questions
1. Which embedding model should we use for semantic search?
2. Should we implement caching for embeddings to improve performance?
3. How should we handle the semantic search threshold for relevance?
4. Which specific LLMs should we prioritize for integration?
5. Should we implement user authentication for API key management?
6. What's the target deployment environment?

## Success Metrics
- Functional LangChain agent with tool calling
- Custom search tool properly integrated
- Semantic search functionality working without vector database
- Support for multiple LLMs via LiteLLM
- Comprehensive test coverage
- Clean, intuitive UI with warmer tones
- Proper error handling and logging
