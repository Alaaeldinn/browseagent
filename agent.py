from typing import List, Dict, Any
import os

from search_tool import SearXNGSearchTool, OldSearchTool
# Import LiteLLM wrapper
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


# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

def get_llm_instance(model: str = "openrouter/gpt-3.5-turbo", temperature: float = 0.1, max_tokens: int = None):
    """
    Get an LLM instance using OpenRouter with fallback mechanisms and user config
    """
    # Validate OpenRouter API key
    if "OPENROUTER_API_KEY" not in os.environ or not os.environ["OPENROUTER_API_KEY"]:
        raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter models")

    # For OpenRouter models we should use a different approach since the model name format is different
    if model.startswith("openrouter/"):
        # Extract the actual OpenRouter model name (without the openrouter/ prefix)
        openrouter_model = model.replace("openrouter/", "")
    else:
        # If model doesn't start with openrouter/, assume it's an OpenRouter model and use as is
        openrouter_model = model

    try:
        # Create ChatOpenAI instance with model-specific configuration
        llm_params = {
            "model": openrouter_model,
            "temperature": temperature,
            "openai_api_base": "https://openrouter.ai/api/v1",
            "openai_api_key": os.environ["OPENROUTER_API_KEY"],
            "default_headers": {
                "HTTP-Referer": os.getenv("YOUR_SITE_URL", "https://browseagent.example"),
                "X-Title": os.getenv("YOUR_APP_NAME", "BrowseAgent"),
            },
            "request_timeout": 30
        }

        # Add max_tokens if specified
        if max_tokens:
            llm_params["max_tokens"] = max_tokens

        return ChatOpenAI(**llm_params)
    except Exception as e:
        # If OpenRouter model fails, try a fallback model - without printing
        try:
            # Fallback to a known free model with user config
            fallback_model = "gpt-3.5-turbo"  # This is typically available
            fallback_params = {
                "model": fallback_model,
                "temperature": temperature,
                "openai_api_base": "https://openrouter.ai/api/v1",
                "openai_api_key": os.environ["OPENROUTER_API_KEY"],
                "default_headers": {
                    "HTTP-Referer": os.getenv("YOUR_SITE_URL", "https://browseagent.example"),
                    "X-Title": os.getenv("YOUR_APP_NAME", "BrowseAgent"),
                },
                "request_timeout": 30
            }

            if max_tokens:
                fallback_params["max_tokens"] = max_tokens

            return ChatOpenAI(**fallback_params)
        except Exception:
            # If fallback also fails, raise the original error
            raise e


class BrowseAgent:
    def __init__(self, llm_provider: str = "openrouter/gpt-3.5-turbo", searx_host: str = "https://searx.space", use_searxng: bool = True,
                 temperature: float = 0.1, max_tokens: int = None):
        """
        Initialize the BrowseAgent with OpenRouter as the LLM provider and configuration
        """
        # Initialize the LLM using OpenRouter
        self.llm = get_llm_instance(model=llm_provider, temperature=temperature, max_tokens=max_tokens)

        # Initialize the search tool based on configuration
        if use_searxng:
            self.search_tool = SearXNGSearchTool(searx_host=searx_host)
        else:
            self.search_tool = OldSearchTool()  # Fallback to old tool

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
            # Use the underlying OpenAI client to extract keywords
            from openai import OpenAI
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )

            response = client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": keyword_extraction_prompt}],
                temperature=0.1
            )
            keywords = response.choices[0].message.content.strip()
            return keywords if keywords else query  # fallback to original query if extraction fails
        except Exception as e:
            # If LLM keyword extraction fails, return the original query
            # print(f"Keyword extraction failed: {str(e)}")  # Commented out to avoid undefined warning
            return query

    def run_query(self, query: str) -> str:
        """
        Run a query through the agent with enhanced error handling and fallbacks
        """
        try:
            # Extract keywords from the query
            keywords = self.extract_keywords(query)

            # Run the query through the agent executor
            result = self.agent_executor.invoke({"input": keywords})

            # If result is a dict, extract the output part
            if isinstance(result, dict):
                if "output" in result:
                    return str(result["output"])
                elif "response" in result:
                    return str(result["response"])
                else:
                    # If we can't find a clear output, return the string representation
                    return str(result)
            else:
                # If result is not a dict, return as string
                return str(result)
        except Exception as e:
            # More detailed error handling for different types of errors
            error_msg = str(e)
            # Remove logging for unit testing - no logging here

            if "api_key" in error_msg.lower() or "401" in error_msg or "API key" in error_msg:
                # Handle API key issues by using the search tool directly
                try:
                    search_result = self.search_tool._run(keywords)
                    return f"API key issue encountered. Search results: {search_result}"
                except Exception as search_error:
                    return f"API key issue encountered and search also failed: {str(search_error)}"
            elif "not a valid model ID" in error_msg or "model" in error_msg.lower():
                # Handle model name format issues with fallback to default model
                try:
                    # Try to reinitialize with a default model - preserving original temperature
                    original_temperature = self.llm.temperature if hasattr(self.llm, 'temperature') else 0.1
                    original_llm = self.llm
                    self.llm = get_llm_instance(model="openai/gpt-3.5-turbo", temperature=original_temperature)

                    # Recreate the agent with the new LLM
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "You are a helpful assistant that can search the web for information."),
                        ("human", "{input}"),
                        ("placeholder", "{agent_scratchpad}"),
                    ])

                    self.agent = create_openai_tools_agent(
                        llm=self.llm,
                        tools=self.tools,
                        prompt=prompt
                    )

                    self.agent_executor = AgentExecutor(
                        agent=self.agent,
                        tools=self.tools,
                        verbose=True,
                        handle_parsing_errors=True
                    )

                    # Retry the query with the fallback model
                    result = self.agent_executor.invoke({"input": keywords})
                    if isinstance(result, dict):
                        if "output" in result:
                            return str(result["output"])
                        else:
                            return str(result)
                    else:
                        return str(result)
                except Exception:
                    # If all fallbacks fail, return search results directly
                    search_result = self.search_tool._run(keywords)
                    return f"Model issue encountered. Using direct search again later. Error: {str(e)}"
            else:
                # For other errors, try to get search results directly
                try:
                    search_result = self.search_tool._run(keywords)
                    return f"Error occurred during agent execution, but here are direct search results: {search_result}"
                except Exception as search_error:
                    return f"Error occurred during agent execution: {str(e)}. Search also failed: {str(search_error)}"


def process_query_with_agent(
    query: str,
    llm_provider: str = "openrouter/gpt-3.5-turbo",
    searx_host: str = "https://searx.space",
    use_searxng: bool = True,
    temperature: float = 0.1,
    max_tokens: int = None
) -> Dict[str, Any]:
    """
    Process a query with the BrowseAgent with error handling and configuration
    """
    # Remove logging for unit testing

    try:
        agent = BrowseAgent(
            llm_provider=llm_provider,
            searx_host=searx_host,
            use_searxng=use_searxng,
            temperature=temperature,
            max_tokens=max_tokens
        )
        result = agent.run_query(query)

        return {
            "query": query,
            "llm_provider": llm_provider,
            "result": result,
            "search_engine": "searxng" if use_searxng else "ddgs",
            "status": "success"
        }
    except Exception as e:
        return {
            "query": query,
            "llm_provider": llm_provider,
            "result": f"Error processing query: {str(e)}",
            "search_engine": "searxng" if use_searxng else "ddgs",
            "status": "error"
        }
