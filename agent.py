from typing import List, Dict, Any
from search_tool import SearchTool

# Import LiteLLM wrapper
import litellm
import os


import litellm
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import LLMResult
from langchain_core.callbacks import CallbackManager
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import BaseTool
from langchain_core.tools import Tool
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI


# Use ChatOpenAI with LiteLLM provider
from langchain_openai import ChatOpenAI


def get_llm_instance(model: str = "openai/gpt-3.5-turbo"):
    """
    Get an LLM instance using LiteLLM through ChatOpenAI
    """
    # For LiteLLM, we don't necessarily need an API key if using it as a proxy
    # However, we need the appropriate API key for the selected provider in the environment
    # Check what API key is needed based on the model

    if model.startswith("openai/"):
        if "OPENAI_API_KEY" not in os.environ or not os.environ["OPENAI_API_KEY"]:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
    elif model.startswith("anthropic/"):
        if "ANTHROPIC_API_KEY" not in os.environ or not os.environ["ANTHROPIC_API_KEY"]:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic models")
    elif model.startswith("google/"):
        if "GOOGLE_API_KEY" not in os.environ or not os.environ["GOOGLE_API_KEY"]:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Google models")
    elif model.startswith("mistral/"):
        if "MISTRAL_API_KEY" not in os.environ or not os.environ["MISTRAL_API_KEY"]:
            raise ValueError("MISTRAL_API_KEY environment variable is required for Mistral models")
    elif model.startswith("perplexity/"):
        if "PERPLEXITY_API_KEY" not in os.environ or not os.environ["PERPLEXITY_API_KEY"]:
            raise ValueError("PERPLEXITY_API_KEY environment variable is required for Perplexity models")
    elif model.startswith("groq/"):
        if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
            raise ValueError("GROQ_API_KEY environment variable is required for Groq models")

    # Create ChatOpenAI instance with the specified model
    # LiteLLM supports multiple providers via the model parameter (e.g., "openai/gpt-3.5-turbo", "anthropic/claude-3", etc.)
    return ChatOpenAI(
        model=model,
        temperature=0.1
    )


class BrowseAgent:
    def __init__(self, llm_provider: str = "openai/gpt-3.5-turbo"):
        """
        Initialize the BrowseAgent with a specific LLM provider
        """
        # Initialize the LLM using LiteLLM directly
        # We'll use the litellm package to interface with various LLM providers

        # Create a custom LLM that uses LiteLLM
        self.llm = get_llm_instance(model=llm_provider)

        # Initialize the search tool
        self.search_tool = SearchTool()

        # Create tools list for the agent
        self.tools = [
            Tool(
                name=self.search_tool.name,
                func=self.search_tool._run,
                description=self.search_tool.description
            )
        ]

        # Define a proper prompt template for the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that can search the web for information."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Get a pre-built agent prompt from LangChain Hub
        try:
            # Try to get the react agent prompt
            agent_prompt = hub.pull("hwchase17/openai-tools-agent")
            self.agent = create_tool_calling_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=agent_prompt
            )
        except:
            # Fallback to using the custom prompt
            from langchain.agents import create_openai_tools_agent
            self.agent = create_openai_tools_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )

        # Create the agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def extract_keywords(self, query: str) -> str:
        """
        Extract the best keywords from the query for searching
        """
        # Use the LLM to extract keywords from the query
        keyword_extraction_prompt = f"""
        Extract the most important keywords from the following query that would be useful for a web search:
        Query: {query}

        Return only the keywords, separated by spaces.
        """

        try:
            response = litellm.completion(
                model=self.llm.model,
                messages=[{"role": "user", "content": keyword_extraction_prompt}]
            )
            keywords = response.choices[0].message.content.strip()
            return keywords if keywords else query  # fallback to original query if extraction fails
        except Exception:
            # If LLM keyword extraction fails, return the original query
            return query

    def run_query(self, query: str) -> str:
        """
        Run a query through the agent
        """
        try:
            # Extract keywords from the query
            keywords = self.extract_keywords(query)

            # Run the query through the agent executor
            result = self.agent_executor.invoke({"input": keywords})

            return result
        except Exception as e:
            # More detailed error handling for different types of errors
            error_msg = str(e)
            if "api_key" in error_msg.lower() or "401" in error_msg or "API key" in error_msg:
                # Handle API key issues by using the search tool directly
                search_result = self.search_tool._run(keywords)
                return f"API key issue encountered. Search results: {search_result}"
            else:
                return f"Error occurred during agent execution: {str(e)}"


def process_query_with_agent(query: str, llm_provider: str = "openai/gpt-3.5-turbo") -> Dict[str, Any]:
    """
    Process a query with the BrowseAgent
    """
    agent = BrowseAgent(llm_provider=llm_provider)
    result = agent.run_query(query)

    return {
        "query": query,
        "llm_provider": llm_provider,
        "result": result
    }
