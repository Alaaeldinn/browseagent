from agent import BrowseAgent

def test_agent():
    """
    Test the BrowseAgent with the search tool
    """
    # Initialize the agent
    agent = BrowseAgent(llm_provider="openai/gpt-3.5-turbo")
    
    # Test with a sample query
    query = "What is the latest AI research in language models?"
    result = agent.run_query(query)
    
    print(f"Query: {query}")
    print(f"Result: {result}")
    
    # Test keyword extraction
    keywords = agent.extract_keywords(query)
    print(f"Extracted keywords: {keywords}")

if __name__ == "__main__":
    test_agent()
