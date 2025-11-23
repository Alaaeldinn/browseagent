"""
Unit tests for SearXNG search tool functionality
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from search_tool import SearXNGSearchTool, OldSearchTool
from langchain_community.utilities import SearxSearchWrapper


class TestSearXNGSearchTool:
    """Unit tests for SearXNGSearchTool class"""
    
    def test_initialization_with_default_values(self):
        """Test SearXNGSearchTool initializes with default values"""
        tool = SearXNGSearchTool()
        
        assert tool.searx_host == "https://search.us.projectsegfau.lt"
        assert tool.k == 5
        assert tool.language == "en"
        assert tool.name == "searxng_search_tool"
        assert "searching the web" in tool.description.lower()
        
    def test_initialization_with_custom_values(self):
        """Test SearXNGSearchTool initializes with custom values"""
        custom_host = "https://custom-searx.example.com"
        custom_k = 10
        custom_lang = "es"
        
        tool = SearXNGSearchTool(
            searx_host=custom_host,
            k=custom_k,
            language=custom_lang
        )
        
        assert tool.searx_host == custom_host
        assert tool.k == custom_k
        assert tool.language == custom_lang
        
    def test_run_method_with_valid_query(self):
        """Test _run method with a valid query"""
        tool = SearXNGSearchTool()
        
        # Mock the SearxSearchWrapper
        with patch('search_tool.SearxSearchWrapper') as mock_wrapper:
            mock_instance = Mock()
            mock_instance.results.return_value = [
                {
                    "title": "Test Title",
                    "url": "https://example.com",
                    "content": "Test content",
                    "engine": "google"
                }
            ]
            mock_wrapper.return_value = mock_instance
            
            # Mock the semantic search functions
            with patch('search_tool.process_search_results_with_pipeline') as mock_pipeline:
                mock_pipeline.return_value = [
                    {
                        "title": "Test Title",
                        "href": "https://example.com",
                        "body": "Test content",
                        "engine": "google",
                        "similarity_score": 0.95
                    }
                ]
                
                result = tool._run("test query")
                
                # Verify the results are properly formatted
                assert "Test Title" in result
                assert "https://example.com" in result
                assert "Test content" in result
                
                # Verify the wrapper was called with correct parameters
                mock_wrapper.assert_called_once()
                
    def test_run_method_with_empty_query(self):
        """Test _run method with empty query"""
        tool = SearXNGSearchTool()
        
        result = tool._run("")
        
        assert "Error: Query cannot be empty" in result
        
    def test_run_method_with_none_query(self):
        """Test _run method with None query"""
        tool = SearXNGSearchTool()
        
        result = tool._run(None)
        
        assert "Error: Query cannot be empty" in result
        
    def test_run_method_with_whitespace_query(self):
        """Test _run method with whitespace-only query"""
        tool = SearXNGSearchTool()
        
        result = tool._run("   ")
        
        assert "Error: Query cannot be empty" in result
        
    def test_run_method_with_invalid_host(self):
        """Test _run method with invalid host"""
        tool = SearXNGSearchTool(searx_host="invalid-host")
        
        result = tool._run("test query")
        
        assert "Invalid SearXNG host configuration" in result
        
    def test_run_method_with_language_parameter_error(self):
        """Test _run method handles language parameter error gracefully"""
        tool = SearXNGSearchTool()
        
        with patch('search_tool.SearxSearchWrapper') as mock_wrapper:
            # First call (with language) fails, second call (without language) succeeds
            mock_wrapper.side_effect = [Exception("Language not supported"), Mock()]
            
            with patch('search_tool.process_search_results_with_pipeline') as mock_pipeline:
                mock_pipeline.return_value = [
                    {
                        "title": "Test Title",
                        "href": "https://example.com",
                        "body": "Test content",
                        "engine": "google",
                        "similarity_score": 0.95
                    }
                ]
                
                result = tool._run("test query")
                
                # Should succeed after falling back to wrapper without language
                assert "Test Title" in result
                
    def test_run_method_with_processing_error(self):
        """Test _run method handles pipeline processing error gracefully"""
        tool = SearXNGSearchTool()
        
        with patch('search_tool.SearxSearchWrapper') as mock_wrapper:
            mock_instance = Mock()
            mock_instance.results.return_value = [
                {
                    "title": "Test Title",
                    "url": "https://example.com",
                    "content": "Test content",
                    "engine": "google"
                }
            ]
            mock_wrapper.return_value = mock_instance
            
            # Mock pipeline to raise an exception, so raw results are used
            with patch('search_tool.process_search_results_with_pipeline') as mock_pipeline:
                mock_pipeline.side_effect = Exception("Pipeline error")
                
                result = tool._run("test query")
                
                # Should still return results even if pipeline fails
                assert "Error occurred during SearXNG search" not in result
                
    def test_run_method_with_search_error(self):
        """Test _run method handles search error gracefully"""
        tool = SearXNGSearchTool()
        
        with patch('search_tool.SearxSearchWrapper') as mock_wrapper:
            # Mock both the main wrapper and fallback to raise exceptions
            mock_wrapper.side_effect = [
                Exception("Main search failed"),
                Exception("Fallback search failed")
            ]
            
            result = tool._run("test query")
            
            # Should return error message
            assert "Unable to retrieve search results" in result
            
    def test_engines_optimization(self):
        """Test that query-based engine optimization works"""
        tool = SearXNGSearchTool()
        
        # The optimization functions are tested separately,
        # but we verify they're used in the run method
        with patch('search_tool.get_optimized_engines') as mock_optimized:
            mock_optimized.return_value = (["duckduckgo"], ["general"])
            
            with patch('search_tool.SearxSearchWrapper') as mock_wrapper:
                mock_instance = Mock()
                mock_instance.results.return_value = []
                mock_wrapper.return_value = mock_instance
                
                with patch('search_tool.process_search_results_with_pipeline') as mock_pipeline:
                    mock_pipeline.return_value = []
                    
                    tool._run("general test query")
                    
                    # Verify that optimized engines were used
                    mock_optimized.assert_called_once()
                    
    def test_error_handling_in_run_method(self):
        """Test comprehensive error handling in run method"""
        tool = SearXNGSearchTool()
        
        # Force an unexpected error
        with patch('search_tool.get_optimized_engines', side_effect=Exception("Unexpected error")):
            result = tool._run("test query")
            
            assert "Error occurred during SearXNG search" in result
            

class TestOldSearchTool:
    """Unit tests for OldSearchTool class (for fallback)"""
    
    def test_initialization(self):
        """Test OldSearchTool initializes correctly"""
        tool = OldSearchTool()
        
        assert tool.name == "search_tool"
        assert "DuckDuckGo" in tool.description
        
    def test_run_method_with_valid_query(self):
        """Test OldSearchTool _run method with valid query"""
        tool = OldSearchTool()
        
        # Mock the DDGS search
        with patch('search_tool.search_ddgs') as mock_search:
            mock_search.return_value = [
                {
                    "title": "Test Title",
                    "href": "https://example.com",
                    "body": "Test content"
                }
            ]
            
            with patch('search_tool.process_search_results_with_pipeline') as mock_pipeline:
                mock_pipeline.return_value = [
                    {
                        "title": "Test Title",
                        "href": "https://example.com",
                        "body": "Test content",
                        "similarity_score": 0.95
                    }
                ]
                
                result = tool._run("test query")
                
                assert "Test Title" in result
                mock_search.assert_called_once_with("test query", max_results=200)
                
    def test_run_method_with_empty_query(self):
        """Test OldSearchTool _run method with empty query"""
        tool = OldSearchTool()
        
        result = tool._run("")
        
        assert "Error: Query cannot be empty" in result
        
    def test_run_method_with_pipeline_error(self):
        """Test OldSearchTool handles pipeline error gracefully"""
        tool = OldSearchTool()
        
        with patch('search_tool.search_ddgs') as mock_search:
            mock_search.return_value = [
                {
                    "title": "Test Title",
                    "href": "https://example.com",
                    "body": "Test content"
                }
            ]
            
            with patch('search_tool.process_search_results_with_pipeline') as mock_pipeline:
                mock_pipeline.side_effect = Exception("Pipeline error")
                
                result = tool._run("test query")
                
                # Should still work with raw results if pipeline fails
                assert "Error occurred during search" not in result


if __name__ == "__main__":
    pytest.main([__file__])