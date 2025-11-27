from agent import ResearchAgent
import json

try:
    agent = ResearchAgent()
    
    # Test 1: Check if keyword generation works (tests OpenRouter LLM)
    print("=" * 60)
    print("TEST 1: Testing OpenRouter LLM (keyword generation)")
    print("=" * 60)
    keywords = agent.generate_keywords("artificial intelligence")
    print(f"✓ OpenRouter LLM is working!")
    print(f"Generated keywords: {keywords}")
    print()
    
    # Test 2: Try a full query
    print("=" * 60)
    print("TEST 2: Testing full agent flow")
    print("=" * 60)
    response = agent.run("What is Python programming language?")
    print(f"\nKeywords: {response.keywords}")
    print(f"\nNumber of results found: {len(response.results)}")
    if response.results:
        print("\nTop Results:")
        for r in response.results:
            print(f"- {r.title[:60]}...")
    print(f"\nAnswer: {response.answer[:200]}...")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
