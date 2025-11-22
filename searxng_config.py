"""
SearXNG Configuration Guide for BrowseAgent

This file provides configuration examples for SearXNG engines and settings
to optimize search results for the BrowseAgent application.
"""

# Engine categories available in SearXNG
ENGINE_CATEGORIES = [
    "general",      # General web search (Google, DuckDuckGo, etc.)
    "science",      # Scientific databases (arXiv, PubMed, etc.)
    "images",       # Image search engines
    "videos",       # Video search engines (YouTube, etc.)
    "news",         # News sources
    "social media", # Social media search
    "it",           # IT related searches
    "files",        # File search
    "map",          # Maps and location services
]

# Recommended engine configurations for different use cases
ENGINE_CONFIGURATIONS = {
    "general_research": {
        "categories": ["general"],
        "engines": ["google", "duckduckgo", "bing", "qwant"],
        "timeout": 5.0,
        "k": 10
    },
    "scientific_research": {
        "categories": ["science"],
        "engines": ["arxiv", "google_scholar", "pubmed", "semantic_scholar"],
        "timeout": 10.0,
        "k": 10
    },
    "news_search": {
        "categories": ["news"],
        "engines": ["google_news", "bing_news", "reddit", "yahoo_news"],
        "timeout": 5.0,
        "k": 10
    },
    "technical_search": {
        "categories": ["it"],
        "engines": ["github", "gitlab", "stackoverflow", "archlinux"],
        "timeout": 5.0,
        "k": 10
    },
    "multimedia_search": {
        "categories": ["images", "videos"],
        "engines": ["duckduckgo_images", "bing_images", "youtube", "invidious"],
        "timeout": 5.0,
        "k": 10
    }
}

# Default configuration for BrowseAgent
BROWSEAGENT_DEFAULT_CONFIG = {
    "searx_host": "http://localhost:8080",  # Default local instance
    "categories": ["general", "it", "science"],  # Multi-category search
    "engines": None,  # Use all available engines in categories
    "timeout": 5.0,
    "k": 10,
    "language": "en",
    "safesearch": 0  # No safe search filtering
}

# Configuration for production use (with public instances)
PUBLIC_INSTANCE_CONFIG = {
    "searx_host": "https://searx.example.com",  # Replace with actual instance
    "categories": ["general"],
    "engines": ["duckduckgo", "bing"],  # More reliable engines
    "timeout": 8.0,
    "k": 5,
    "language": "en",
    "safesearch": 0
}

# Function to get optimized engine settings based on query type
def get_optimized_config(query: str) -> dict:
    """
    Return an optimized configuration based on the query content
    """
    query_lower = query.lower()
    
    # Map query keywords to appropriate configurations
    if any(keyword in query_lower for keyword in ["research", "study", "academic", "science", "article", "paper", "scholarly"]):
        return ENGINE_CONFIGURATIONS["scientific_research"]
    elif any(keyword in query_lower for keyword in ["news", "current", "latest", "today", "breaking"]):
        return ENGINE_CONFIGURATIONS["news_search"]
    elif any(keyword in query_lower for keyword in ["code", "github", "programming", "software", "developer", "api"]):
        return ENGINE_CONFIGURATIONS["technical_search"]
    elif any(keyword in query_lower for keyword in ["image", "photo", "video", "picture", "visual"]):
        return ENGINE_CONFIGURATIONS["multimedia_search"]
    else:
        # Default to general research
        return ENGINE_CONFIGURATIONS["general_research"]

# Example usage in BrowseAgent
def apply_searxng_config(tool, query: str):
    """
    Apply optimized configuration to a SearXNG tool based on query
    """
    config = get_optimized_config(query)
    
    # Apply configuration to the tool (assuming the tool allows runtime configuration)
    if hasattr(tool, 'searx_host'):
        # Update configuration attributes
        for key, value in config.items():
            if hasattr(tool, key):
                setattr(tool, key, value)
    
    return tool

if __name__ == "__main__":
    # Example: How to use the configurations
    print("SearXNG Configuration Guide for BrowseAgent")
    print("\nAvailable engine configurations:")
    for name, config in ENGINE_CONFIGURATIONS.items():
        print(f"  - {name}: {config}")
    
    print(f"\nDefault BrowseAgent config: {BROWSEAGENT_DEFAULT_CONFIG}")
    
    # Example of getting optimized config
    sample_queries = [
        "What is the latest research on quantum computing?",
        "Show me the latest news about AI",
        "How to fix a Python TypeError",
        "Find images of the Eiffel Tower"
    ]
    
    print(f"\nOptimized configurations for sample queries:")
    for query in sample_queries:
        config = get_optimized_config(query)
        print(f"  Query: '{query}' -> Config: {config}")