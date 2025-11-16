from langchain.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig
from typing import Optional, List, Dict, Any
import asyncio
from ddgs import DDGS
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


class DDGSSearchTool(BaseTool):
    """
    A custom LangChain tool for performing web searches using DuckDuckGo.
    
    This tool allows the LangChain agent to search the web for current information
    and return structured results.
    """
    name = "ddgs_search"
    description = "Useful for searching the web for current information. Returns a list of search results with titles, links, and snippets."
    
    def _run(
        self, 
        query: str, 
        max_results: int = 10,
        region: str = "wt-wt",
        safesearch: str = "off",
        timelimit: str = "y",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform a web search using DuckDuckGo.
        
        Args:
            query: The search query to execute
            max_results: Maximum number of results to return (default: 10)
            region: Search region (default: "wt-wt" for worldwide)
            safesearch: Safe search level (default: "off")
            timelimit: Time limit for search results (default: "y" for past year)
            run_manager: Callback manager for tool execution
            
        Returns:
            List of dictionaries containing search results with title, link, and body
        """
        try:
            results = DDGS().text(
                query,
                max_results=max_results,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit
            )
            
            # Convert results to a structured format
            formatted_results = []
            for i, result in enumerate(results):
                if result:
                    formatted_results.append({
                        "id": i,
                        "title": result.get("title", ""),
                        "link": result.get("href", ""),
                        "body": result.get("body", ""),
                        "source": result.get("source", ""),
                        "date": result.get("date", "")
                    })
            
            return formatted_results
            
        except Exception as e:
            if run_manager:
                run_manager.on_error(e)
            return [{"error": f"Search failed: {str(e)}"}]

    async def _arun(
        self, 
        query: str, 
        max_results: int = 10,
        region: str = "wt-wt",
        safesearch: str = "off",
        timelimit: str = "y",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """
        Async version of the search tool.
        """
        try:
            # Run the synchronous version in a thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                self._run, 
                query, 
                max_results, 
                region, 
                safesearch, 
                timelimit,
                run_manager
            )
            return results
        except Exception as e:
            if run_manager:
                run_manager.on_error(e)
            return [{"error": f"Async search failed: {str(e)}"}]


class SemanticSearchTool:
    """
    A utility class for performing semantic search on search results.
    
    This class uses sentence transformers to compute embeddings and find
    the most relevant results based on semantic similarity.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic search tool.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def rank_results(self, query: str, search_results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rank search results based on semantic similarity to the query.
        
        Args:
            query: The original search query
            search_results: List of search results from DDGS
            top_k: Number of top results to return
            
        Returns:
            List of top-k most relevant search results
        """
        if not search_results:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query)
            
            # Generate embeddings for search results
            result_texts = []
            for result in search_results:
                # Combine title and body for better context
                text = f"{result.get('title', '')} {result.get('body', '')}"
                result_texts.append(text)
            
            result_embeddings = self.model.encode(result_texts)
            
            # Calculate cosine similarities
            similarities = []
            for i, result_embedding in enumerate(result_embeddings):
                # Cosine similarity: dot product of normalized vectors
                similarity = np.dot(query_embedding, result_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(result_embedding)
                )
                similarities.append((i, similarity))
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k results
            top_results = []
            for idx, _ in similarities[:top_k]:
                result = search_results[idx].copy()
                result['similarity_score'] = float(similarities[idx][1])
                top_results.append(result)
            
            return top_results
            
        except Exception as e:
            # Return original results if semantic search fails
            for i, result in enumerate(search_results[:top_k]):
                result['similarity_score'] = 0.0
            return search_results[:top_k]
