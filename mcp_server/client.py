import asyncio
import json
from pathlib import Path
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters


async def test_research_query():
    """Test the research query functionality via MCP."""
    print("🔍 Testing Research Query via MCP...")

    # Setup server parameters to run our MCP server
    server_path = Path(__file__).parent / "server.py"
    server_params = StdioServerParameters(command="python", args=[str(server_path)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print(f"✅ Available tools: {[tool.name for tool in tools.tools]}")

            # Test the research query tool
            print("\n🔍 Running research query: 'What is quantum computing?'")
            result = await session.call_tool(
                "research_query",
                arguments={"query": "What is quantum computing?", "max_results": 3}
            )

            # The result.content might be a list of content objects
            if result.content:
                # Convert to string and parse as JSON if needed
                content_str = str(result.content[0])
                print(f"✅ Research completed! Content type: {type(result.content[0])}")

                # MCP returns content as objects, need to access them properly
                # The content might be in different formats
                if hasattr(result.content[0], 'text'):
                    # If it's a text content object
                    content_text = result.content[0].text
                    print(f"🔍 Content preview: {content_text[:100]}...")
                else:
                    # Try to print the raw content
                    print(f"🔍 Raw content: {result.content[0]}")

                # In this case, we want to access the returned dictionary from the tool
                # The result.content[0] should be the dictionary returned by the tool
                import json
                try:
                    # The result.content[0] should be the dictionary returned by our tool
                    content_dict = result.content[0].data if hasattr(result.content[0], 'data') else result.content[0]
                    print(f"📝 Keywords preview: {str(content_dict)[:200]}...")
                except:
                    print(f"🔍 Could not parse content: {result.content[0]}")

    print("✅ Research query test completed!")


async def test_generate_keywords():
    """Test the keyword generation functionality via MCP."""
    print("\n🔍 Testing Keyword Generation via MCP...")

    # Setup server parameters to run our MCP server
    server_path = Path(__file__).parent / "server.py"
    server_params = StdioServerParameters(command="python", args=[str(server_path)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Test the keyword generation tool
            print("\n🔍 Generating keywords for: 'artificial intelligence safety'")
            result = await session.call_tool(
                "generate_keywords",
                arguments={"query": "artificial intelligence safety"}
            )

            print(f"✅ Keywords generated: {result.content}")

    print("✅ Keyword generation test completed!")


async def test_semantic_search():
    """Test the semantic search functionality via MCP."""
    print("\n🔍 Testing Semantic Search via MCP...")

    # Setup server parameters to run our MCP server
    server_path = Path(__file__).parent / "server.py"
    server_params = StdioServerParameters(command="python", args=[str(server_path)])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Sample search results to test semantic filtering
            sample_results = [
                {
                    "title": "Introduction to Quantum Computing",
                    "url": "https://example.com/quantum-intro",
                    "content": "Quantum computing is a new paradigm that uses quantum bits to process information"
                },
                {
                    "title": "Machine Learning Basics",
                    "url": "https://example.com/ml-basics",
                    "content": "Machine learning involves training algorithms to recognize patterns in data"
                },
                {
                    "title": "Quantum Algorithms Explained",
                    "url": "https://example.com/quantum-algorithms",
                    "content": "Quantum algorithms leverage quantum mechanical phenomena to solve problems"
                }
            ]

            # Test the semantic search tool
            print(f"\n🔍 Performing semantic search for 'quantum computing' with {len(sample_results)} results")
            result = await session.call_tool(
                "semantic_search",
                arguments={
                    "query": "quantum computing",
                    "search_results": sample_results,
                    "top_k": 2
                }
            )

            print(f"✅ Semantic search completed! Selected {len(result.content) if result.content else 0} results")
            if result.content:
                # result.content contains TextContent objects, need to extract the text
                # and parse as JSON to get the actual dictionary results
                import json
                try:
                    # Get the text from the first content item and parse as JSON
                    content_text = result.content[0].text
                    parsed_results = json.loads(content_text)

                    # Now we can access the results
                    titles = [r['title'] for r in parsed_results]
                    print(f"📊 Selected results: {titles}")
                except Exception as e:
                    print(f"🔍 Could not parse semantic search results: {e}")
                    print(f"Raw content: {result.content[0].text if result.content else 'No content'}")

    print("✅ Semantic search test completed!")


async def main():
    """Run all MCP client tests."""
    print("🚀 Starting MCP Client Tests for BrowseAgent...")

    try:
        await test_research_query()
        await test_generate_keywords()
        await test_semantic_search()

        print("\n🎉 All MCP client tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during MCP testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())