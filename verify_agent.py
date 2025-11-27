from agent import agent
import json

try:
    print("Running agent with query: 'latest advancements in solid state batteries'")
    response = agent.run("latest advancements in solid state batteries")
    print("\nKeywords:", response.keywords)
    print("\nTop Results:")
    for r in response.results:
        print(f"- {r.title} ({r.link})")
    print("\nAnswer:", response.answer)
except Exception as e:
    print(f"Error: {e}")
