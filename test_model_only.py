from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

# Test OpenRouter model directly
print("=" * 60)
print("Testing OpenRouter Model (Normal Chat Mode)")
print("=" * 60)

llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL"),
    model=os.getenv("OPENROUTER_MODEL"),
    temperature=0.7
)

# Test 1: Simple question
print("\nTest 1: Simple question")
print("-" * 60)
prompt = ChatPromptTemplate.from_template("Answer this question: {question}")
chain = prompt | llm | StrOutputParser()
response = chain.invoke({"question": "What is the capital of France?"})
print(f"Q: What is the capital of France?")
print(f"A: {response}")

# Test 2: More complex question
print("\nTest 2: Complex question")
print("-" * 60)
response = chain.invoke({"question": "Explain quantum computing in simple terms."})
print(f"Q: Explain quantum computing in simple terms.")
print(f"A: {response}")

# Test 3: Creative task
print("\nTest 3: Creative task")
print("-" * 60)
response = chain.invoke({"question": "Write a haiku about programming."})
print(f"Q: Write a haiku about programming.")
print(f"A: {response}")

print("\n" + "=" * 60)
print("✓ OpenRouter model is working perfectly in normal chat mode!")
print("=" * 60)
