import os
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
import torch

load_dotenv()

# ============================================================================
# SEARXNG SEARCH TOOL
# ============================================================================

@tool
def searxng_search(query: str) -> List[Dict[str, str]]:
    """
    Search the web using SearXNG. Returns a list of search results with title, URL, and content.
    
    Args:
        query: The search query string
        
    Returns:
        List of dictionaries containing 'title', 'url', and 'content' for each result
    """
    base_url = "http://localhost:4000/search"  # Replace with your SearxNG instance URL
    
    params = {
        "q": query,
        "format": "json"
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            # Extract metadata and body content from each result
            formatted_results = []
            for result in results[:20]:  # Get up to 20 results for semantic filtering
                formatted_results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'content': result.get('content', '')  # This is the snippet/body
                })
            
            print(f"✓ Fetched {len(formatted_results)} results from SearXNG")
            return formatted_results
        else:
            print(f"✗ SearXNG returned status {response.status_code}")
            return []
            
    except Exception as e:
        print(f"✗ SearXNG search failed: {str(e)}")
        return []


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SearchResult(BaseModel):
    title: str
    link: str  # Using 'link' instead of 'url' for consistency
    snippet: str  # Using 'snippet' instead of 'content' for API compatibility

class ResearchResponse(BaseModel):
    keywords: List[str]
    results: List[SearchResult]
    answer: str


# ============================================================================
# RESEARCH AGENT
# ============================================================================

class ResearchAgent:
    def __init__(self):
        # Initialize LLM with OpenRouter configuration (keep as is)
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
            model=os.getenv("OPENROUTER_MODEL"),
            temperature=0
        )
        
        # Initialize sentence transformer for semantic similarity
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create the agent with search tool
        self.tools = [searxng_search]
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
    
    def _create_agent(self):
        """Create a tool-calling agent for research"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant. When the user asks a question:
1. Generate optimal search keywords or statements to find relevant information
2. Use the searxng_search tool to search for information
3. You will receive search results to work with

Be concise and focused in your searches."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        return create_tool_calling_agent(self.llm, self.tools, prompt)
    
    def generate_keywords(self, query: str) -> List[str]:
        """Generate search keywords using the LLM"""
        prompt = ChatPromptTemplate.from_template(
            "Generate 3-5 optimal search keywords or short phrases for this research topic: {topic}\n"
            "Return them as a JSON array of strings."
        )
        
        from langchain_core.output_parsers import JsonOutputParser
        chain = prompt | self.llm | JsonOutputParser()
        keywords = chain.invoke({"topic": query})
        print(f"\n📝 Generated keywords: {keywords}")
        return keywords
    
    def semantic_search(self, query: str, results: List[Dict[str, str]], top_k: int = 5) -> List[Dict[str, str]]:
        """
        Perform semantic similarity search on results using sentence-transformers.
        Returns top_k most similar results to the original query.
        """
        if not results:
            return []
        
        # Embed the original query
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        
        # Embed all result contents (metadata + body)
        # Combine title and content for better semantic matching
        result_texts = [f"{r['title']} {r['content']}" for r in results]
        result_embeddings = self.embedding_model.encode(result_texts, convert_to_tensor=True)
        
        # Calculate cosine similarity
        cosine_scores = util.cos_sim(query_embedding, result_embeddings)[0]
        
        # Get top-k results
        top_indices = torch.topk(cosine_scores, k=min(top_k, len(results))).indices
        
        top_results = [results[i] for i in top_indices]
        print(f"\n🔍 Semantic search: Selected top {len(top_results)} most relevant results")
        return top_results
    
    def generate_answer(self, query: str, context_results: List[Dict[str, str]]) -> str:
        """Generate an answer based on the search results context"""
        # Format context from search results
        context_str = "\n\n".join([
            f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
            for r in context_results
        ])
        
        prompt = ChatPromptTemplate.from_template(
            """You are a research assistant. Answer the user's question based ONLY on the following search results:

Context:
{context}

Question: {question}

Provide a comprehensive answer based on the context. If the context doesn't contain enough information, say so."""
        )
        
        from langchain_core.output_parsers import StrOutputParser
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "question": query})
        return answer
    
    def run(self, user_query: str) -> ResearchResponse:
        """
        Main research flow:
        1. Generate keywords
        2. Search using tool calling
        3. Semantic similarity filtering
        4. Generate answer from context
        """
        print(f"\n{'='*70}")
        print(f"🔬 Research Query: {user_query}")
        print(f"{'='*70}")
        
        # Step 1: Generate keywords
        keywords = self.generate_keywords(user_query)
        
        # Step 2: Perform search with generated keywords
        search_query = " ".join(keywords[:3])  # Use top 3 keywords
        print(f"\n🔎 Searching for: {search_query}")
        
        search_results = searxng_search(search_query)
        
        # Step 3: Semantic similarity search
        top_results = self.semantic_search(user_query, search_results, top_k=5)
        
        # Step 4: Generate answer from context
        if top_results:
            print(f"\n💬 Generating answer from {len(top_results)} results...")
            answer = self.generate_answer(user_query, top_results)
        else:
            answer = "I'm sorry, I couldn't find relevant information to answer your question."
        
        # Format response
        formatted_results = [
            SearchResult(
                title=r['title'],
                link=r['url'],
                snippet=r['content']
            )
            for r in top_results
        ]
        
        return ResearchResponse(
            keywords=keywords,
            results=formatted_results,
            answer=answer
        )


# ============================================================================
# GLOBAL AGENT INSTANCE
# ============================================================================

agent = ResearchAgent()
