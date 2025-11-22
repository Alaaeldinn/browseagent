"""
Simple test to verify SearXNG tool functionality
"""
from search_tool import SearXNGSearchTool
import requests


def test_searxng_tool():
    """
    Test the SearXNG search tool
    """
    print("Testing SearXNG Search Tool...")

    # Test multiple SearXNG instances to find one that works
    searx_instances = [
        "https://search.us.projectsegfau.lt",  # Default
        "https://searx.be",                   # Alternative
        "https://search.ononoki.org",         # Alternative
        "https://searx.space"                 # Original
    ]

    test_query = "What is the capital of France?"
    print(f"Query: {test_query}")

    for instance in searx_instances:
        print(f"\nTrying SearXNG instance: {instance}")

        try:
            # Create the tool instance with this specific instance
            tool = SearXNGSearchTool(searx_host=instance, k=3)

            # Test the connection first
            test_url = f"{instance}/about"
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200:
                print(f"  ✓ Connection to {instance} successful")
            else:
                print(f"  ✗ Connection to {instance} failed with status {response.status_code}")
                continue

            # Test the search
            result = tool._run(test_query)
            print(f"  Result: {result}")

            if "Error occurred during SearXNG search" not in result and result.strip() != "[]":
                print(f"  ✓ SearXNG tool test completed successfully with {instance}!")
                return True
            else:
                print(f"  ! SearXNG tool returned an empty result with {instance}")

        except Exception as e:
            print(f"  ✗ Error during SearXNG tool test with {instance}: {e}")
            continue

    print("  ✗ All SearXNG instances failed")
    return False


if __name__ == "__main__":
    success = test_searxng_tool()
    if success:
        print("\n✓ SearXNG integration is working properly!")
    else:
        print("\n✗ SearXNG integration needs debugging.")
        print("Note: This may be due to public instance limitations.")
        print("For production use, consider setting up your own SearXNG instance.")