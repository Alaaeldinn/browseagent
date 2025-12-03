import logging
import os
import sys
from typing import List, Dict
import importlib.util
from mcp.server.fastmcp import FastMCP

# Configure logging
name = "browseagent-mcp-server"
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(name)

# Create MCP server instance - FastMCP doesn't accept logger in constructor
mcp = FastMCP(name)

# Import the research agent functionality by adding parent directory to path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import agent


@mcp.tool()
def research_query(query: str, max_results: int = 5) -> Dict:
    """
    Execute a research query using the BrowseAgent AI research system.

    Args:
        query: The research question or topic to investigate
        max_results: Maximum number of results to return (default: 5)

    Returns:
        A dictionary containing the research results with keywords, sources, and answer
    """
    logger.info(f"Research query called: {query}")
    try:
        # Run the research using the existing agent
        response = agent.run(query)

        # Convert to dictionary format for MCP
        result = {
            "keywords": response.keywords,
            "results": [
                {
                    "title": result.title,
                    "link": result.link,
                    "snippet": result.snippet
                }
                for result in response.results
            ],
            "answer": response.answer
        }

        logger.info(f"Research completed, found {len(result['results'])} results")
        return result
    except Exception as e:
        logger.error(f"Error in research_query: {str(e)}")
        return {
            "keywords": [],
            "results": [],
            "answer": f"Error conducting research: {str(e)}"
        }


@mcp.tool()
def generate_keywords(query: str) -> List[str]:
    """
    Generate search keywords for a research topic using the AI research agent.

    Args:
        query: The research topic to generate keywords for

    Returns:
        List of generated keywords
    """
    logger.info(f"Generating keywords for: {query}")
    try:
        # Use the agent's keyword generation functionality
        keywords = agent.generate_keywords(query)
        logger.info(f"Generated keywords: {keywords}")
        return keywords
    except Exception as e:
        logger.error(f"Error in generate_keywords: {str(e)}")
        return [query]  # Fallback to the original query


@mcp.tool()
def semantic_search(query: str, search_results: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Perform semantic similarity search on provided results using the AI research agent's capabilities.

    Args:
        query: The original query to match against
        search_results: List of search results to filter
        top_k: Number of top results to return (default: 5)

    Returns:
        List of top K most relevant results based on semantic similarity
    """
    logger.info(f"Performing semantic search for: {query}")
    try:
        # Use the agent's semantic search functionality
        results = agent.semantic_search(query, search_results, top_k)
        logger.info(f"Semantic search completed, selected {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error in semantic_search: {str(e)}")
        return search_results[:top_k]  # Fallback to first K results


if __name__ == "__main__":
    logger.info("Starting BrowseAgent MCP Server...")
    try:
        # Run the MCP server using stdio transport
        mcp.run(transport="stdio")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("Server terminated")