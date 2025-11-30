"""
Tools module for BrowseAgent
Contains utility functions and specialized tools
"""

import os
import requests
from typing import List, Dict
from langchain_core.tools import tool


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