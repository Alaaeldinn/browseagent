import os
import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import torch

load_dotenv()

class SearchResult(BaseModel):
    title: str
    link: str
    snippet: str

class ResearchResponse(BaseModel):
    keywords: List[str]
    results: List[SearchResult]
    answer: str

class ResearchAgent:
    def __init__(self):
        # We use ChatOpenAI because OpenRouter is OpenAI-compatible
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL"),
            model=os.getenv("OPENROUTER_MODEL"),
            temperature=0
        )
        
        # Initialize SearxSearchWrapper with local or remote instance
        searx_host = os.getenv("SEARXNG_URL", "http://localhost:8888")
        self.searx = SearxSearchWrapper(searx_host=searx_host)
        
        # Initialize sentence transformer model for semantic search
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def search_searxng(self, query: str, num_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search using SearxSearchWrapper with multi-engine meta-search.
        Returns list of search results with title, link, and snippet.
        """
        try:
            # Use .results() method to get structured data
            # This aggregates results from multiple search engines
            results = self.searx.results(
                query,
                num_results=num_results,
                engines=['google', 'bing', 'duckduckgo', 'wikipedia']  # Multi-engine search
            )
            
            print(f"✓ Successfully fetched {len(results)} results from SearXNG")
            return results
            
        except Exception as e:
            print(f"✗ SearXNG search failed: {str(e)}")
            return []

    def generate_keywords(self, query: str) -> List[str]:
        prompt = ChatPromptTemplate.from_template(
            "Suggest 3-5 search keywords for the following research topic: {topic}. Return them as a JSON list of strings."
        )
        chain = prompt | self.llm | JsonOutputParser()
        return chain.invoke({"topic": query})

    def search(self, keywords: List[str], original_query: str) -> List[Dict[str, Any]]:
        # Search for the most relevant keyword combination
        query = " ".join(keywords[:3])
        # Fetch more results to filter down
        results = self.search_searxng(query, num_results=20)
        
        if not results:
            return []

        # Semantic filtering
        # 1. Embed the original query
        query_embedding = self.embedding_model.encode(original_query, convert_to_tensor=True)
        
        # 2. Embed the search results (snippets)
        snippets = [r.get("snippet", "") for r in results]
        snippet_embeddings = self.embedding_model.encode(snippets, convert_to_tensor=True)
        
        # 3. Calculate cosine similarity
        cosine_scores = util.cos_sim(query_embedding, snippet_embeddings)[0]
        
        # 4. Get top 5 results
        top_results_indices = torch.topk(cosine_scores, k=min(5, len(results))).indices
        
        top_results = [results[i] for i in top_results_indices]
        return top_results

    def generate_answer(self, query: str, context: List[Dict[str, Any]]) -> str:
        context_str = "\n\n".join([f"Title: {r.get('title')}\nLink: {r.get('link')}\nSnippet: {r.get('snippet')}" for r in context])
        
        prompt = ChatPromptTemplate.from_template(
            """You are a research assistant. Answer the user's question based ONLY on the following context:

            Context:
            {context}

            Question: {question}

            Answer:"""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context_str, "question": query})

    def run(self, query: str) -> ResearchResponse:
        keywords = self.generate_keywords(query)
        raw_results = self.search(keywords, query)
        
        # Format results
        formatted_results = []
        for r in raw_results:
            formatted_results.append(SearchResult(
                title=r.get("title", ""),
                link=r.get("link", ""),
                snippet=r.get("snippet", "")
            ))
            
        answer = self.generate_answer(query, raw_results)
        
        return ResearchResponse(
            keywords=keywords,
            results=formatted_results,
            answer=answer
        )

agent = ResearchAgent()
