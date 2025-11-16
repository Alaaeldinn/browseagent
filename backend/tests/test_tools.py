import pytest
from unittest.mock import Mock, patch
from tools import DDGSSearchTool, SemanticSearchTool
import json


class TestDDGSSearchTool:
    """Test cases for the DDGSSearchTool class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = DDGSSearchTool()
    
    @patch('tools.DDGS')
    def test_run_success(self, mock_ddgs):
        """Test successful execution of the search tool."""
        # Mock the DDGS response
        mock_results = [
            {
                "title": "Test Result",
                "href": "https://example.com",
                "body": "This is a test result body",
                "source": "Example Source",
                "date": "2023-01-01"
            }
        ]
        
        mock_ddgs_instance = Mock()
        mock_ddgs_instance.text.return_value = mock_results
        mock_ddgs.return_value = mock_ddgs_instance
        
        # Execute the tool
        result = self.tool._run("test query")
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["title"] == "Test Result"
        assert result[0]["link"] == "https://example.com"
        assert result[0]["body"] == "This is a test result body"
        assert result[0]["source"] == "Example Source"
        assert result[0]["date"] == "2023-01-01"
        
        # Verify DDGS was called with correct parameters
        mock_ddgs_instance.text.assert_called_once_with(
            "test query",
            max_results=10,
            region="wt-wt",
            safesearch="off",
            timelimit="y"
        )
    
    @patch('tools.DDGS')
    def test_run_with_custom_parameters(self, mock_ddgs):
        """Test execution with custom parameters."""
        mock_ddgs_instance = Mock()
        mock_ddgs_instance.text.return_value = []
        mock_ddgs.return_value = mock_ddgs_instance
        
        # Execute with custom parameters
        result = self.tool._run(
            "test query",
            max_results=5,
            region="us-en",
            safesearch="moderate",
            timelimit="m"
        )
        
        # Verify DDGS was called with custom parameters
        mock_ddgs_instance.text.assert_called_once_with(
            "test query",
            max_results=5,
            region="us-en",
            safesearch="moderate",
            timelimit="m"
        )
    
    @patch('tools.DDGS')
    def test_run_error_handling(self, mock_ddgs):
        """Test error handling in the search tool."""
        # Mock an exception
        mock_ddgs.side_effect = Exception("Search failed")
        
        # Execute the tool
        result = self.tool._run("test query")
        
        # Verify error handling
        assert isinstance(result, list)
        assert len(result) == 1
        assert "error" in result[0]
        assert "Search failed" in result[0]["error"]
    
    @patch('tools.DDGS')
    def test_run_empty_results(self, mock_ddgs):
        """Test handling of empty search results."""
        mock_ddgs_instance = Mock()
        mock_ddgs_instance.text.return_value = []
        mock_ddgs.return_value = mock_ddgs_instance
        
        # Execute the tool
        result = self.tool._run("test query")
        
        # Verify empty results
        assert isinstance(result, list)
        assert len(result) == 0
    
    @patch('tools.DDGS')
    def test_run_partial_results(self, mock_ddgs):
        """Test handling of partial/None results."""
        # Mock results with some None values
        mock_results = [
            {"title": "Valid Result", "href": "https://example.com", "body": "Valid body"},
            None,
            {"title": "Another Result", "href": "https://example2.com", "body": "Another body"}
        ]
        
        mock_ddgs_instance = Mock()
        mock_ddgs_instance.text.return_value = mock_results
        mock_ddgs.return_value = mock_ddgs_instance
        
        # Execute the tool
        result = self.tool._run("test query")
        
        # Verify only valid results are returned
        assert isinstance(result, list)
        assert len(result) == 2  # Only non-None results
        assert result[0]["title"] == "Valid Result"
        assert result[1]["title"] == "Another Result"


class TestSemanticSearchTool:
    """Test cases for the SemanticSearchTool class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = SemanticSearchTool()
    
    @patch('tools.SentenceTransformer')
    def test_rank_results_success(self, mock_sentence_transformer):
        """Test successful ranking of search results."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        # Mock embeddings
        mock_model.encode.side_effect = [
            [0.1, 0.2, 0.3],  # Query embedding
            [0.1, 0.2, 0.3],  # Result 1 embedding (high similarity)
            [0.9, 0.8, 0.7],  # Result 2 embedding (low similarity)
            [0.2, 0.3, 0.4]   # Result 3 embedding (medium similarity)
        ]
        
        # Test data
        query = "test query"
        search_results = [
            {"title": "Result 1", "body": "Body 1"},
            {"title": "Result 2", "body": "Body 2"},
            {"title": "Result 3", "body": "Body 3"}
        ]
        
        # Execute the tool
        result = self.tool.rank_results(query, search_results, top_k=2)
        
        # Verify the result
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["title"] == "Result 1"  # Highest similarity
        assert result[1]["title"] == "Result 3"  # Medium similarity
        assert "similarity_score" in result[0]
        assert "similarity_score" in result[1]
        assert result[0]["similarity_score"] > result[1]["similarity_score"]
    
    def test_rank_results_empty_input(self):
        """Test handling of empty input."""
        result = self.tool.rank_results("test query", [], top_k=5)
        assert isinstance(result, list)
        assert len(result) == 0
    
    @patch('tools.SentenceTransformer')
    def test_rank_results_error_handling(self, mock_sentence_transformer):
        """Test error handling in semantic search."""
        # Mock an exception
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Embedding failed")
        mock_sentence_transformer.return_value = mock_model
        
        # Test data
        query = "test query"
        search_results = [
            {"title": "Result 1", "body": "Body 1"},
            {"title": "Result 2", "body": "Body 2"}
        ]
        
        # Execute the tool
        result = self.tool.rank_results(query, search_results, top_k=2)
        
        # Verify fallback to original results
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["title"] == "Result 1"
        assert result[1]["title"] == "Result 2"
        assert "similarity_score" in result[0]
        assert "similarity_score" in result[1]
        assert result[0]["similarity_score"] == 0.0
        assert result[1]["similarity_score"] == 0.0
    
    @patch('tools.SentenceTransformer')
    def test_rank_results_top_k(self, mock_sentence_transformer):
        """Test top_k parameter functionality."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        # Mock embeddings with same similarity for simplicity
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        
        # Test data
        query = "test query"
        search_results = [
            {"title": f"Result {i}", "body": f"Body {i}"}
            for i in range(10)
        ]
        
        # Execute with top_k=3
        result = self.tool.rank_results(query, search_results, top_k=3)
        
        # Verify only top_k results are returned
        assert isinstance(result, list)
        assert len(result) == 3
    
    @patch('tools.SentenceTransformer')
    def test_rank_results_text_combination(self, mock_sentence_transformer):
        """Test that title and body are combined for embeddings."""
        # Mock the sentence transformer
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        # Mock embeddings
        mock_model.encode.side_effect = [
            [0.1, 0.2, 0.3],  # Query embedding
            [0.1, 0.2, 0.3],  # Combined text embedding
        ]
        
        # Test data
        query = "test query"
        search_results = [
            {"title": "Test Title", "body": "Test Body Content"}
        ]
        
        # Execute the tool
        result = self.tool.rank_results(query, search_results, top_k=1)
        
        # Verify that encode was called with combined text
        expected_text = "Test Title Test Body Content"
        mock_model.encode.assert_any_call(expected_text)
        
        # Verify the result
        assert len(result) == 1
        assert result[0]["title"] == "Test Title"


if __name__ == "__main__":
    pytest.main([__file__])
