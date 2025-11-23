"""
Search Result Processing Pipeline for BrowseAgent

This module provides a comprehensive pipeline to process search results,
including ranking, filtering, deduplication, and formatting for LLM consumption.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import re
import logging
from urllib.parse import urlparse


@dataclass
class SearchResult:
    """Represents a single search result"""
    title: str
    url: str
    content: str
    engine: str = ""
    similarity_score: float = 0.0
    published_date: Optional[str] = None
    source_type: Optional[str] = None  # news, blog, academic, etc.
    
    def domain(self) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(self.url).netloc
        except:
            return ""
    
    def clean_content(self) -> str:
        """Return cleaned content with HTML tags removed"""
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', self.content)
        # Remove extra whitespace
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()


class SearchResultProcessor:
    """Processes search results through multiple stages: filtering, ranking, and formatting"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_results(self, results: List[Dict[str, Any]], query: str) -> List[SearchResult]:
        """
        Process raw search results through the complete pipeline
        """
        # Convert raw results to SearchResult objects
        search_results = self._convert_to_search_results(results)
        
        # Apply processing pipeline
        processed_results = self._filter_results(search_results)
        processed_results = self._deduplicate_results(processed_results)
        processed_results = self._rank_by_relevance(processed_results)
        processed_results = self._categorize_results(processed_results)

        return processed_results
    
    def _convert_to_search_results(self, raw_results: List[Dict[str, Any]]) -> List[SearchResult]:
        """Convert raw search results to SearchResult objects"""
        results = []
        for raw_result in raw_results:
            result = SearchResult(
                title=raw_result.get("title", ""),
                url=raw_result.get("href", raw_result.get("url", "")),
                content=raw_result.get("body", raw_result.get("content", "")),
                engine=raw_result.get("engine", ""),
                similarity_score=raw_result.get("similarity_score", 0.0)
            )
            results.append(result)
        return results
    
    def _filter_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Filter out low quality results based on various criteria"""
        filtered_results = []
        
        for result in results:
            # Filter out results with empty titles or URLs
            if not result.title.strip():
                continue
            if not result.url.strip():
                continue
            
            # Filter out results with very short content (likely low quality)
            if len(result.content.strip()) < 50:
                continue
            
            # Filter out results from known problematic domains
            domain = result.domain().lower()
            blocked_domains = [
                'example.com', 'test.com', 'invalid.com',
                'localhost', '127.0.0.1'
            ]
            if domain in blocked_domains:
                continue
            
            # Filter out results that don't have meaningful content
            content = result.content.lower()
            if any(keyword in content for keyword in [
                'page not found', '404 error', 'not found', 
                'access denied', 'forbidden'
            ]):
                continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL and title similarity"""
        seen_urls = set()
        seen_titles = set()
        unique_results = []
        
        for result in results:
            url = result.url.lower().strip()
            title = result.title.lower().strip()
            
            # Check for duplicate URLs
            if url in seen_urls:
                continue
            
            # Check for very similar titles
            is_duplicate = False
            for seen_title in seen_titles:
                if self._titles_are_similar(title, seen_title):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_urls.add(url)
                seen_titles.add(title)
                unique_results.append(result)
        
        return unique_results
    
    def _titles_are_similar(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """Check if two titles are similar using simple string similarity"""
        # Simple Jaccard similarity on words
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def _rank_by_relevance(self, results: List[SearchResult]) -> List[SearchResult]:
        """Rank results by relevance, considering multiple factors"""
        def relevance_score(result: SearchResult) -> float:
            score = result.similarity_score  # Base score from semantic search
            
            # Boost score for high-authority domains
            domain = result.domain().lower()
            high_authority_domains = [
                'wikipedia.org', 'github.com', 'stackoverflow.com',
                'arxiv.org', 'researchgate.net', 'edu', 'gov', 'org'
            ]
            if any(auth_domain in domain for auth_domain in high_authority_domains):
                score += 0.1
            
            # Boost score for results with publication dates
            if result.published_date:
                score += 0.05
            
            # Penalty for very long URLs (often parameter-heavy)
            if len(result.url) > 200:
                score -= 0.05
            
            return score
        
        # Sort by relevance score (descending)
        sorted_results = sorted(results, key=relevance_score, reverse=True)
        return sorted_results
    
    def _categorize_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Categorize results by source type"""
        categorized_results = []
        
        for result in results:
            domain = result.domain().lower()
            
            if any(site in domain for site in ['arxiv.org', 'researchgate.net', 'springer.com']):
                result.source_type = 'academic'
            elif any(site in domain for site in ['github.com', 'stackoverflow.com', 'gitlab.com']):
                result.source_type = 'technical'
            elif any(site in domain for site in ['news', 'cnn.com', 'bbc.com', 'reuters.com']):
                result.source_type = 'news'
            elif any(site in domain for site in ['blog', 'wordpress.com', 'medium.com']):
                result.source_type = 'blog'
            elif any(site in domain for site in ['wikipedia.org', 'encyclopedia']):
                result.source_type = 'reference'
            else:
                result.source_type = 'general'
            
            categorized_results.append(result)
        
        return categorized_results
    
    def format_for_llm(self, results: List[SearchResult], max_results: int = 5) -> str:
        """
        Format results for LLM consumption
        """
        if not results:
            return "No search results found."
        
        formatted_results = []
        for i, result in enumerate(results[:max_results]):
            formatted_result = (
                f"Result {i+1}:\n"
                f"Title: {result.title}\n"
                f"URL: {result.url}\n"
                f"Source Type: {result.source_type}\n"
                f"Domain: {result.domain()}\n"
                f"Content: {result.clean_content()[:500]}...\n"  # Truncate long content
                f"Relevance Score: {result.similarity_score:.3f}\n"
                f"Search Engine: {result.engine}\n"
                f"---\n"
            )
            formatted_results.append(formatted_result)
        
        return "".join(formatted_results)


# For this implementation, we'll use a simpler approach
def apply_pipeline(data, *functions):
    """Apply a series of functions to data in sequence"""
    result = data
    for func in functions:
        result = func(result)
    return result


# Example usage and testing
if __name__ == "__main__":
    # Test the processor
    processor = SearchResultProcessor()
    
    # Sample raw results
    raw_results = [
        {
            "title": "Introduction to Machine Learning",
            "href": "https://example.com/ml-intro",
            "body": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
            "engine": "google",
            "similarity_score": 0.85
        },
        {
            "title": "Deep Learning Explained",
            "href": "https://another.com/deep-learning",
            "body": "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
            "engine": "bing",
            "similarity_score": 0.73
        }
    ]
    
    # Process the results
    processed = processor.process_results(raw_results, "machine learning")
    formatted = processor.format_for_llm(processed)
    
    print("Processed Results:")
    print(formatted)
    
    print(f"\nFound {len(processed)} unique results after processing")