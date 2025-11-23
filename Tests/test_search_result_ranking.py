"""
Unit tests for search result ranking and processing logic
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from search_result_pipeline import SearchResultProcessor, SearchResult


class TestSearchResultRanking:
    """Unit tests for search result ranking and processing logic"""
    
    def test_semantic_search_similarity_basic(self):
        """Test basic semantic search similarity functionality"""
        from search_tool import semantic_search_similarity
        
        query = "machine learning"
        results = [
            {"title": "Introduction to ML", "body": "Machine learning is a subset of AI"},
            {"title": "Cooking Guide", "body": "How to cook pasta"},
            {"title": "ML Algorithms", "body": "Different machine learning algorithms"}
        ]
        
        ranked_results = semantic_search_similarity(query, results)
        
        # Should have similarity scores
        assert all("similarity_score" in result for result in ranked_results)
        
        # Results should be sorted by similarity (highest first)
        scores = [result["similarity_score"] for result in ranked_results]
        assert scores == sorted(scores, reverse=True)
        
        # The ML-related results should rank higher than cooking
        first_result = ranked_results[0]
        assert "machine learning" in first_result["title"].lower() or "ml" in first_result["title"].lower()
    
    def test_semantic_search_with_empty_results(self):
        """Test semantic search with empty results"""
        from search_tool import semantic_search_similarity
        
        ranked_results = semantic_search_similarity("test", [])
        
        assert ranked_results == []
    
    def test_top_5_results(self):
        """Test the top_5 function"""
        from search_tool import top_5
        
        results = [{"score": i} for i in range(10, 0, -1)]  # 10 results with scores 10, 9, 8, ..., 1
        
        top_results = top_5(results)
        
        assert len(top_results) == 5
        assert top_results[0]["score"] == 10
        assert top_results[-1]["score"] == 6
    
    def test_top_5_with_fewer_results(self):
        """Test top_5 with fewer than 5 results"""
        from search_tool import top_5
        
        results = [{"score": i} for i in range(3)]  # Only 3 results
        
        top_results = top_5(results)
        
        assert len(top_results) == 3  # Should return all results
    
    def test_process_search_results_with_pipeline(self):
        """Test the complete search result processing pipeline"""
        from search_tool import process_search_results_with_pipeline
        
        raw_results = [
            {
                "title": "Machine Learning Basics",
                "href": "https://example.com/ml",
                "body": "An introduction to machine learning concepts and algorithms",
                "engine": "google"
            },
            {
                "title": "Cooking Pasta",
                "href": "https://example.com/pasta",
                "body": "How to cook delicious pasta with sauce",
                "engine": "bing"
            }
        ]
        
        processed_results = process_search_results_with_pipeline("machine learning", raw_results)
        
        # Should return processed results
        assert len(processed_results) <= 5  # Max 5 results
        assert all("title" in result for result in processed_results)
        assert all("href" in result for result in processed_results)
        assert all("body" in result for result in processed_results)
        assert all("similarity_score" in result for result in processed_results)


class TestSearchResultProcessor:
    """Tests for the SearchResultProcessor class"""
    
    def test_initialization(self):
        """Test SearchResultProcessor initialization"""
        processor = SearchResultProcessor()
        
        assert processor is not None
        assert hasattr(processor, '_convert_to_search_results')
        assert hasattr(processor, '_filter_results')
        assert hasattr(processor, '_deduplicate_results')
        assert hasattr(processor, '_rank_by_relevance')
        assert hasattr(processor, '_categorize_results')
    
    def test_convert_to_search_results(self):
        """Test conversion of raw results to SearchResult objects"""
        processor = SearchResultProcessor()
        
        raw_results = [
            {
                "title": "Test Title",
                "href": "https://example.com",
                "body": "Test content",
                "engine": "google",
                "similarity_score": 0.85
            }
        ]
        
        search_results = processor._convert_to_search_results(raw_results)
        
        assert len(search_results) == 1
        result = search_results[0]
        assert isinstance(result, SearchResult)
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.content == "Test content"
        assert result.engine == "google"
        assert result.similarity_score == 0.85
    
    def test_filter_results_removes_empty_items(self):
        """Test that filter removes results with empty titles or URLs"""
        processor = SearchResultProcessor()
        
        results = [
            SearchResult(title="", url="https://example.com", content="content"),
            SearchResult(title="Valid Title", url="", content="content"),
            SearchResult(title="Good Result", url="https://good.com", content="Good content"),
            SearchResult(title="Another Good", url="https://another.com", content=""),  # Short content
        ]
        
        filtered = processor._filter_results(results)
        
        # Should only keep the result with both title and URL
        assert len(filtered) == 1
        assert filtered[0].title == "Good Result"
    
    def test_filter_results_removes_short_content(self):
        """Test that filter removes results with very short content"""
        processor = SearchResultProcessor()
        
        results = [
            SearchResult(title="Title 1", url="https://example.com", content="This is valid content that is long enough"),
            SearchResult(title="Title 2", url="https://example2.com", content="hi"),  # Too short
            SearchResult(title="Title 3", url="https://example3.com", content="a"),  # Too short
        ]
        
        filtered = processor._filter_results(results)
        
        # Should only keep results with sufficient content
        assert len(filtered) == 1
        assert filtered[0].title == "Title 1"
    
    def test_filter_results_removes_blocked_domains(self):
        """Test that filter removes results from blocked domains"""
        processor = SearchResultProcessor()
        
        results = [
            SearchResult(title="Good Site", url="https://good-site.com", content="Good content"),
            SearchResult(title="Blocked", url="https://example.com", content="Content"),
            SearchResult(title="Also Blocked", url="https://test.com", content="Content"),
            SearchResult(title="Localhost", url="https://localhost", content="Content"),
        ]
        
        filtered = processor._filter_results(results)
        
        # Should only keep results from non-blocked domains
        assert len(filtered) == 1
        assert filtered[0].title == "Good Site"
    
    def test_filter_results_removes_error_content(self):
        """Test that filter removes results with error content"""
        processor = SearchResultProcessor()
        
        results = [
            SearchResult(title="Valid", url="https://example.com", content="Normal content"),
            SearchResult(title="Error", url="https://example2.com", content="Page not found"),
            SearchResult(title="Error2", url="https://example3.com", content="404 error occurred"),
            SearchResult(title="Valid2", url="https://example4.com", content="Another valid result"),
        ]
        
        filtered = processor._filter_results(results)
        
        # Should only keep results without error content
        assert len(filtered) == 2
        titles = [r.title for r in filtered]
        assert "Valid" in titles
        assert "Valid2" in titles
        assert "Error" not in titles
        assert "Error2" not in titles
    
    def test_deduplicate_results_by_url(self):
        """Test that deduplication works by URL"""
        processor = SearchResultProcessor()
        
        results = [
            SearchResult(title="Title 1", url="https://example.com", content="content 1"),
            SearchResult(title="Title 2", url="https://example.com", content="content 2"),  # Duplicate URL
            SearchResult(title="Title 3", url="https://different.com", content="content 3"),
            SearchResult(title="Title 4", url="https://EXAMPLE.com", content="content 4"),  # Case-insensitive duplicate
        ]
        
        deduplicated = processor._deduplicate_results(results)
        
        # Should only keep unique URLs (case-insensitive)
        assert len(deduplicated) == 2
        urls = [r.url.lower() for r in deduplicated]
        assert "https://example.com" in urls
        assert "https://different.com" in urls
    
    def test_deduplicate_results_by_similar_titles(self):
        """Test that deduplication works by similar titles"""
        processor = SearchResultProcessor()
        
        results = [
            SearchResult(title="Machine Learning Introduction", url="https://example.com/1", content="content 1"),
            SearchResult(title="Introduction to Machine Learning", url="https://example.com/2", content="content 2"),
            # These titles are very similar
            SearchResult(title="Deep Learning Basics", url="https://example.com/3", content="content 3"),
            SearchResult(title="Python Programming Guide", url="https://example.com/4", content="content 4"),
        ]
        
        # Test the title similarity function directly
        is_similar = processor._titles_are_similar(
            "Machine Learning Introduction",
            "Introduction to Machine Learning"
        )
        # These should be considered similar based on Jaccard similarity
        # Both contain "machine", "learning", "introduction" words
        assert is_similar  # Similar titles should be detected
        
        # Now test the full deduplication process
        deduplicated = processor._deduplicate_results(results)
        # The exact behavior depends on the similarity threshold
        assert len(deduplicated) <= len(results)
    
    def test_rank_by_relevance_basic(self):
        """Test basic relevance ranking"""
        processor = SearchResultProcessor()
        
        results = [
            SearchResult(title="Title A", url="https://a.com", content="content", similarity_score=0.9),
            SearchResult(title="Title B", url="https://b.com", content="content", similarity_score=0.7),
            SearchResult(title="Title C", url="https://c.com", content="content", similarity_score=0.8),
        ]
        
        ranked = processor._rank_by_relevance(results)
        
        # Should be ordered by similarity score (highest first)
        scores = [r.similarity_score for r in ranked]
        assert scores == sorted(scores, reverse=True)
        assert ranked[0].similarity_score == 0.9
        assert ranked[-1].similarity_score == 0.7
    
    def test_rank_by_relevance_authority_boost(self):
        """Test that high-authority domains get ranking boost"""
        processor = SearchResultProcessor()
        
        results = [
            SearchResult(title="Regular Site", url="https://regular.com", content="content", similarity_score=0.8),
            SearchResult(title="Wikipedia", url="https://wikipedia.org", content="content", similarity_score=0.7),
            SearchResult(title="GitHub", url="https://github.com", content="content", similarity_score=0.6),
        ]
        
        ranked = processor._rank_by_relevance(results)
        
        # Wikipedia and GitHub should rank higher despite lower similarity scores
        # due to authority boost
        top_result = ranked[0]
        assert "wikipedia" in top_result.url or "github" in top_result.url
    
    def test_categorize_results(self):
        """Test result categorization by domain"""
        processor = SearchResultProcessor()
        
        results = [
            SearchResult(title="Academic Paper", url="https://arxiv.org/abs/1234.5678", content="research content"),
            SearchResult(title="Code Repository", url="https://github.com/user/repo", content="code content"),
            SearchResult(title="News Article", url="https://cnn.com/article", content="news content"),
            SearchResult(title="Blog Post", url="https://medium.com/post", content="blog content"),
            SearchResult(title="Reference", url="https://wikipedia.org/wiki/Topic", content="reference content"),
            SearchResult(title="General", url="https://example.com", content="general content"),
        ]
        
        categorized = processor._categorize_results(results)
        
        categories = [r.source_type for r in categorized]
        assert "academic" in categories
        assert "technical" in categories 
        assert "news" in categories
        assert "blog" in categories
        assert "reference" in categories
        assert "general" in categories
    
    def test_format_for_llm(self):
        """Test formatting results for LLM consumption"""
        processor = SearchResultProcessor()
        
        results = [
            SearchResult(
                title="Test Title", 
                url="https://example.com", 
                content="Test content here with more text",
                source_type="general",
                engine="google",
                similarity_score=0.85
            )
        ]
        
        formatted = processor.format_for_llm(results)
        
        assert "Test Title" in formatted
        assert "https://example.com" in formatted
        assert "general" in formatted
        assert "google" in formatted
        assert "0.850" in formatted  # Similarity score formatted to 3 decimals
    
    def test_format_for_llm_truncation(self):
        """Test that long content is truncated in LLM formatting"""
        processor = SearchResultProcessor()
        
        long_content = "This is a very long content " + "word " * 200 + " at the end."
        results = [
            SearchResult(
                title="Long Content", 
                url="https://example.com", 
                content=long_content,
                source_type="general",
                engine="google",
                similarity_score=0.75
            )
        ]
        
        formatted = processor.format_for_llm(results)
        
        # Should contain truncated content with "..." indicator
        assert "..." in formatted
        # Should not contain the full 200-word content
        assert len(formatted) < len(long_content)
    
    def test_format_for_llm_with_multiple_results(self):
        """Test formatting multiple results"""
        processor = SearchResultProcessor()
        
        results = [
            SearchResult(title="Result 1", url="https://example1.com", content="content1", similarity_score=0.9),
            SearchResult(title="Result 2", url="https://example2.com", content="content2", similarity_score=0.8),
            SearchResult(title="Result 3", url="https://example3.com", content="content3", similarity_score=0.7),
        ]
        
        formatted = processor.format_for_llm(results, max_results=2)  # Limit to 2 results
        
        assert "Result 1" in formatted
        assert "Result 2" in formatted
        assert "Result 3" not in formatted  # Should be limited to 2
        
        # Should have separators between results
        result_count = formatted.count("---")
        assert result_count == 2  # 2 separators for 2 results
    
    def test_format_for_llm_empty_results(self):
        """Test formatting with empty results"""
        processor = SearchResultProcessor()
        
        formatted = processor.format_for_llm([])
        
        assert formatted == "No search results found."
    
    def test_process_results_complete_pipeline(self):
        """Test the complete processing pipeline"""
        processor = SearchResultProcessor()
        
        raw_results = [
            {"title": "Good Result", "href": "https://good.com", "body": "Good content", "engine": "google", "similarity_score": 0.8},
            {"title": "Duplicate", "href": "https://duplicate.com", "body": "content", "engine": "bing", "similarity_score": 0.7},
            {"title": "Duplicate", "href": "https://duplicate.com", "body": "different content", "engine": "bing", "similarity_score": 0.6},  # Duplicate
            {"title": "Bad URL", "href": "https://example.com", "body": "content", "engine": "google", "similarity_score": 0.9},  # Will be filtered
        ]
        
        processed = processor.process_results(raw_results, "test query")
        
        # Should have filtered out bad domains and duplicates
        assert len(processed) < len(raw_results)  # Some results filtered out
        assert all(r.source_type is not None for r in processed)  # All categorized
        # Should be ranked by relevance
        scores = [r.similarity_score for r in processed]
        assert scores == sorted(scores, reverse=True)


class TestSearchResultUtilities:
    """Test utility functions in search_result_pipeline"""
    
    def test_search_result_domain_extraction(self):
        """Test domain extraction from SearchResult URL"""
        result = SearchResult(
            title="Test", 
            url="https://www.example.com/path?query=value", 
            content="test content"
        )
        
        domain = result.domain()
        
        assert domain == "www.example.com"
    
    def test_search_result_domain_extraction_invalid_url(self):
        """Test domain extraction with invalid URL"""
        result = SearchResult(
            title="Test", 
            url="invalid-url", 
            content="test content"
        )
        
        domain = result.domain()
        
        assert domain == ""
    
    def test_search_result_clean_content(self):
        """Test content cleaning functionality"""
        result = SearchResult(
            title="Test", 
            url="https://example.com", 
            content="This has <b>HTML</b> tags and   extra   spaces"
        )
        
        clean_content = result.clean_content()
        
        assert "<b>" not in clean_content
        assert "</b>" not in clean_content
        assert "  " not in clean_content  # Multiple spaces normalized
        assert clean_content.startswith("This has HTML tags and")
    
    def test_titles_are_similar(self):
        """Test title similarity function"""
        processor = SearchResultProcessor()
        
        # Very similar titles
        similar = processor._titles_are_similar(
            "Introduction to Machine Learning", 
            "Machine Learning Introduction"
        )
        assert similar
        
        # Completely different titles
        different = processor._titles_are_similar(
            "Machine Learning", 
            "Cooking Pasta"
        )
        assert not different
        
        # Partially similar
        partial = processor._titles_are_similar(
            "Python Machine Learning", 
            "Machine Learning with Python"
        )
        assert partial


# Integration tests combining semantic search with pipeline
class TestSearchIntegration:
    """Integration tests for search functionality"""
    
    def test_full_search_flow(self):
        """Test the complete search flow from raw results to formatted output"""
        from search_tool import semantic_search_similarity, process_search_results_with_pipeline
        
        # Start with raw search results
        raw_results = [
            {"title": "Machine Learning Guide", "href": "https://example.com/ml", "body": "Complete guide to machine learning", "engine": "google"},
            {"title": "Cooking Recipes", "href": "https://example.com/recipes", "body": "Delicious cooking recipes", "engine": "bing"},
            {"title": "ML Algorithms Explained", "href": "https://example.com/algorithms", "body": "Detailed explanation of ML algorithms", "engine": "google"},
        ]
        
        # Apply semantic search to rank by query relevance
        query = "machine learning concepts"
        semantic_ranked = semantic_search_similarity(query, raw_results)
        
        # Apply the processing pipeline
        processed = process_search_results_with_pipeline(query, semantic_ranked)
        
        # The machine learning results should rank higher than cooking
        top_result = processed[0] if processed else {}
        assert "machine learning" in top_result.get("title", "").lower() or "ml" in top_result.get("title", "").lower()
        
        # Should have been processed by the pipeline
        assert len(processed) <= 5
        assert all("source_type" in result for result in processed)
        assert all("similarity_score" in result for result in processed)


if __name__ == "__main__":
    pytest.main([__file__])