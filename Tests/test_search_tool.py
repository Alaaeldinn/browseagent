from search_tool import SearXNGSearchTool, OldSearchTool

def test_search_tool():
    """
    Test the search tool independently
    """
    tool = SearXNGSearchTool()  # Use the correct class name

    # Test with a sample query
    query = "What is the latest AI research in language models?"
    result = tool._run(query)

    print(f"Query: {query}")
    print(f"Result: {result}")

if __name__ == "__main__":
    test_search_tool()
