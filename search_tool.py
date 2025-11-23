from typing import List, Dict, Any, Optional
from sentence_transformers import util
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_community.utilities import SearxSearchWrapper
import re
import logging


def semantic_search_similarity(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Perform semantic search similarity using sentence transformers
    """
    if not results:
        return []

    from sentence_transformers import SentenceTransformer

    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode the query
    query_embedding = model.encode([query])

    # Prepare result texts for encoding
    result_texts = [result.get('body', '') or result.get('snippet', '') or result.get('title', '') for result in results]

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


def process_search_results_with_pipeline(query: str, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process search results using the comprehensive pipeline
    """
    from search_result_pipeline import SearchResultProcessor

    processor = SearchResultProcessor()
    processed_results = processor.process_results(raw_results, query)
    formatted_output = processor.format_for_llm(processed_results, max_results=5)

    # Convert back to the format expected by the agent
    final_results = []
    for result in processed_results[:5]:  # Take top 5
        final_results.append({
            "title": result.title,
            "href": result.url,
            "body": result.clean_content(),
            "engine": result.engine,
            "similarity_score": result.similarity_score,
            "source_type": result.source_type
        })

    return final_results


def top_5(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return the top 5 results based on similarity scores
    """
    return results[:5]


def get_query_category(query: str) -> str:
    """
    Determine the category of the query to optimize search engine selection
    """
    query_lower = query.lower()

    # Define keywords for different categories
    science_keywords = ["research", "study", "academic", "science", "article", "paper", "scholarly", "thesis", "dissertation", "scientific", "journal", "pubmed", "arxiv"]
    news_keywords = ["news", "current", "latest", "today", "breaking", "recent", "update", "report"]
    tech_keywords = ["code", "github", "programming", "software", "developer", "api", "bug", "tutorial", "framework", "library", "technology", "algorithm"]
    multimedia_keywords = ["image", "photo", "video", "picture", "visual", "gallery", "photograph", "movie", "film", "youtube"]

    # Check for category-specific keywords
    if any(keyword in query_lower for keyword in science_keywords):
        return "scientific_research"
    elif any(keyword in query_lower for keyword in news_keywords):
        return "news_search"
    elif any(keyword in query_lower for keyword in tech_keywords):
        return "technical_search"
    elif any(keyword in query_lower for keyword in multimedia_keywords):
        return "multimedia_search"
    else:
        return "general_research"


def get_optimized_engines(query: str) -> tuple:
    """
    Return optimized engines and categories based on query type
    Returns (engines, categories) tuple
    """
    category = get_query_category(query)

    # Define optimal configurations for different categories
    engine_configurations = {
        "general_research": (["duckduckgo", "bing"], ["general"]),
        "scientific_research": (["arxiv", "semantic_scholar"], ["science"]),
        "news_search": (["bing_news", "reddit"], ["news"]),
        "technical_search": (["github", "stackoverflow"], ["it"]),
        "multimedia_search": (["duckduckgo_images"], ["images"])
    }

    return engine_configurations.get(category, (None, ["general"]))


class SearXNGInput(BaseModel):
    query: str = Field(description="The search query to find information about")


class SearXNGSearchTool(BaseTool):
    name: str = "searxng_search_tool"
    description: str = "Useful for searching the web for current information on any topic using SearXNG, an open-source privacy-respecting metasearch engine"
    args_schema: type = SearXNGInput

    # SearXNG host - defaults to a public instance, but can be configured
    searx_host: str = Field(default="https://search.us.projectsegfau.lt", description="The SearXNG instance URL")
    k: int = Field(default=5, description="Number of results to return")
    engines: Optional[List[str]] = Field(default=None, description="Specific search engines to use")
    categories: Optional[List[str]] = Field(default=None, description="Categories to search in")
    language: str = Field(default="en", description="Language code for search")

    def __init__(
        self,
        searx_host: str = "https://search.us.projectsegfau.lt",
        k: int = 5,
        engines: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        language: str = "en",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.searx_host = searx_host
        self.k = k
        self.engines = engines
        self.categories = categories
        self.language = language

    def _run(self, query: str) -> str:
        """
        Run the SearXNG search tool with the given query
        """
        try:
            # Validate inputs
            if not query or not query.strip():
                return "Error: Query cannot be empty"

            # Get optimized engines and categories based on the query
            try:
                optimized_engines, optimized_categories = get_optimized_engines(query)
            except Exception as e:
                logging.error(f"Error getting optimized engines for query: {str(e)}")
                optimized_engines, optimized_categories = (None, ["general"])  # Fallback

            # Use optimized configuration if original configuration is not specified
            engines_to_use = self.engines if self.engines else optimized_engines
            categories_to_use = self.categories if self.categories else optimized_categories

            # Validate SearXNG host
            if not self.searx_host or not self.searx_host.startswith(('http://', 'https://')):
                return "Error: Invalid SearXNG host configuration"

            # Initialize SearXNG search wrapper
            # Note: language parameter might not be supported in all versions
            # We'll handle it conditionally
            searx_params = {
                "searx_host": self.searx_host,
                "k": self.k,
                "engines": engines_to_use,
                "categories": categories_to_use
            }

            # Only add language if it's supported (some versions don't support it)
            searx_search = None
            try:
                searx_search = SearxSearchWrapper(**searx_params, language=self.language)
            except Exception:
                # If language parameter is not supported, create without it
                try:
                    searx_search = SearxSearchWrapper(**searx_params)
                except Exception as e:
                    return f"Error initializing SearXNG search wrapper: {str(e)}"

            # Get raw results from SearXNG
            try:
                raw_results = searx_search.results(query, num_results=self.k)
            except Exception as e:
                logging.error(f"Error getting results from SearXNG: {str(e)}")
                # Try with minimal parameters as fallback
                minimal_params = {"searx_host": self.searx_host, "k": self.k}
                try:
                    searx_search_fallback = SearxSearchWrapper(**minimal_params)
                    raw_results = searx_search_fallback.results(query, num_results=min(3, self.k))
                except Exception:
                    return "Error: Unable to retrieve search results from SearXNG instance"

            # Process results through the comprehensive pipeline
            try:
                processed_results = process_search_results_with_pipeline(query, raw_results)
            except Exception as e:
                logging.error(f"Error in search result pipeline: {str(e)}")
                # Fallback to basic processing
                processed_results = raw_results  # Use raw results if pipeline fails

            # Get top results (already processed by the pipeline if successful)
            top_results = processed_results

            # Format results for the LLM
            formatted_results = []
            for result in top_results:
                formatted_result = {
                    "title": result.get("title", ""),
                    "href": result.get("url", "") or result.get("href", ""),
                    "body": result.get("content", "") or result.get("body", "") or result.get("snippet", ""),
                    "engine": result.get("engine", "unknown"),
                    "similarity_score": round(result.get("similarity_score", 0), 4),
                    "source_type": result.get("source_type", "unknown")
                }
                formatted_results.append(formatted_result)

            return str(formatted_results)

        except Exception as e:
            logging.error(f"Unexpected error in SearXNG search tool: {str(e)}")
            return f"Error occurred during SearXNG search: {str(e)}"

    def _arun(self, query: str):
        """
        Async version of the search tool
        """
        raise NotImplementedError("SearXNGSearchTool does not support async")


# Backward compatibility - keeping the old DDGS-based tool as an alternative
from ddgs import DDGS

def search_ddgs(query: str, max_results: int = 200) -> List[Dict[str, Any]]:
    """
    Perform a search using DuckDuckGo Search (for fallback purposes)
    """
    results = DDGS().text(
        query,
        max_results=max_results,
        region="wt-wt",
        safesearch="off",
        timelimit='y'
    )
    return results


class OldSearchTool(BaseTool):
    name: str = "search_tool"
    description: str = "Useful for searching the web for current information on any topic (DuckDuckGo)"
    args_schema: type = SearXNGInput

    def _run(self, query: str) -> str:
        """
        Run the old search tool with the given query (for fallback)
        """
        try:
            # Validate inputs
            if not query or not query.strip():
                return "Error: Query cannot be empty"

            # Step 1: Perform initial search
            try:
                initial_results = search_ddgs(query, max_results=200)
            except Exception as e:
                logging.error(f"Error in DDGS search: {str(e)}")
                return f"Error occurred during DuckDuckGo search: {str(e)}"

            # Step 2: Process results through the comprehensive pipeline
            try:
                final_results = process_search_results_with_pipeline(query, initial_results)
            except Exception as e:
                logging.error(f"Error in search result pipeline: {str(e)}")
                # Fallback to basic processing
                final_results = initial_results  # Use raw results if pipeline fails

            # Format results for the LLM
            formatted_results = []
            for result in final_results:
                formatted_result = {
                    "title": result.get("title", ""),
                    "href": result.get("href", ""),
                    "body": result.get("body", ""),
                    "similarity_score": round(result.get("similarity_score", 0), 4)
                }
                formatted_results.append(formatted_result)

            return str(formatted_results)

        except Exception as e:
            logging.error(f"Unexpected error in old search tool: {str(e)}")
            return f"Error occurred during search: {str(e)}"

    def _arun(self, query: str):
        """
        Async version of the search tool
        """
        raise NotImplementedError("SearchTool does not support async")
