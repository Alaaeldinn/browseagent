"""
BrowseAgent with LLM Integration

This module implements the BrowseAgent class with support for multiple LLM providers
through LiteLLM, providing a unified interface for different AI models.
"""

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.tools import Tool
from tools import DDGSSearchTool, SemanticSearchTool
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from llm_config import LLMManager, LLMConfig, LLMProvider

# Load environment variables
load_dotenv()

class BrowseAgent:
    """
    The main BrowseAgent class that orchestrates web search and semantic analysis.
    
    This agent uses LangChain's framework to:
    1. Search the web using DuckDuckGo
    2. Apply semantic search to filter and rank results
    3. Generate responses based on the most relevant information
    4. Support multiple LLM providers through LiteLLM
    """
    
    def __init__(self, model_name: str = None, api_key: str = None, llm_manager: LLMManager = None):
        """
        Initialize the BrowseAgent.
        
        Args:
            model_name: The LLM model to use (default: from LLMManager)
            api_key: API key for the model (if not in LLMManager)
            llm_manager: Optional LLMManager instance (creates default if not provided)
        """
        # Initialize LLM manager
        self.llm_manager = llm_manager or LLMManager()
        
        # Set model and API key
        if model_name:
            self.set_model(model_name, api_key)
        else:
            self.model_name = self.llm_manager.get_default_model()
            if not self.model_name:
                raise ValueError("No default model configured. Please provide a model name or configure models in LLMManager.")
        
        # Initialize tools
        self.search_tool = DDGSSearchTool()
        self.semantic_search = SemanticSearchTool()
        
        # Create LangChain tools
        self.tools = [
            Tool(
                name="web_search",
                description="Search the web for current information using DuckDuckGo",
                func=self._search_wrapper,
                return_direct=False
            ),
            Tool(
                name="semantic_search",
                description="Apply semantic search to rank and filter search results",
                func=self._semantic_search_wrapper,
                return_direct=False
            )
        ]
        
        # Initialize the agent
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """
        Create the LangChain agent with the appropriate tools and prompt.
        """
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are BrowseAgent, an AI-powered research assistant.
            
Your capabilities:
1. Search the web for current information using the web_search tool
2. Apply semantic search to find the most relevant results
3. Synthesize information to provide comprehensive answers

Guidelines:
- Use web_search when you need current information beyond your knowledge cutoff
- Use semantic_search to filter and rank search results for relevance
- Provide accurate, well-sourced information with proper citations
- Be concise but comprehensive in your responses
- Always cite your sources using the provided links

Process:
1. Analyze the user's query to understand what information is needed
2. Use web_search with appropriate keywords
3. Use semantic_search to filter results if many are returned
4. Synthesize the most relevant information into your response
5. Cite your sources properly

Remember: You have access to current web search capabilities, so you can provide up-to-date information."""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Get model configuration
        config = self.llm_manager.get_config(self.model_name)
        if not config:
            raise ValueError(f"Configuration not found for model: {self.model_name}")
        
        # Initialize the LLM using LiteLLM
        llm_params = self.llm_manager.get_litellm_params(self.model_name)
        
        # Create the agent
        agent = create_openai_tools_agent(
            llm=llm_params,  # Pass the parameters directly
            tools=self.tools,
            prompt=prompt
        )
        
        # Create the agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        
        return agent_executor
    
    def _search_wrapper(self, query: str, max_results: int = 10) -> str:
        """
        Wrapper for the web search tool to format results for the agent.
        """
        try:
            results = self.search_tool._run(query, max_results=max_results)
            if results and "error" not in results[0]:
                # Format results for the agent
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("body", "")[:200] + "..." if len(result.get("body", "")) > 200 else result.get("body", ""),
                        "source": result.get("source", ""),
                        "date": result.get("date", "")
                    })
                return f"Found {len(formatted_results)} search results:\n\n" + "\n\n".join(
                    f"**{r['title']}**\n{r['snippet']}\nSource: {r['source']}\nLink: {r['link']}\n" 
                    for r in formatted_results
                )
            else:
                return f"Search failed: {results[0].get('error', 'Unknown error')}"
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def _semantic_search_wrapper(self, query: str, search_results: str, top_k: int = 5) -> str:
        """
        Wrapper for the semantic search tool to format results for the agent.
        """
        try:
            # Parse the search results string back to a list of dictionaries
            # This is a simplified parsing - in a real implementation, you might want to pass structured data
            lines = search_results.split('\n\n')
            parsed_results = []
            
            for line in lines:
                if line.strip() and "**" in line:
                    parts = line.split('\n')
                    title = parts[0].strip('**')
                    snippet = parts[1] if len(parts) > 1 else ""
                    source_link = parts[2] if len(parts) > 2 else ""
                    
                    # Extract source and link
                    if "Source: " in source_link:
                        source = source_link.split("Source: ")[1].split(" Link: ")[0]
                        link = source_link.split("Link: ")[1] if "Link: " in source_link else ""
                    else:
                        source = ""
                        link = ""
                    
                    parsed_results.append({
                        "title": title,
                        "body": snippet,
                        "source": source,
                        "link": link
                    })
            
            # Apply semantic search
            ranked_results = self.semantic_search.rank_results(query, parsed_results, top_k)
            
            # Format results for the agent
            formatted_results = []
            for result in ranked_results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("body", "")[:200] + "..." if len(result.get("body", "")) > 200 else result.get("body", ""),
                    "source": result.get("source", ""),
                    "link": result.get("link", ""),
                    "similarity": result.get("similarity_score", 0.0)
                })
            
            return f"Ranked {len(formatted_results)} most relevant results:\n\n" + "\n\n".join(
                f"**{r['title']}** (Similarity: {r['similarity']:.2f})\n{r['snippet']}\nSource: {r['source']}\nLink: {r['link']}\n" 
                for r in formatted_results
            )
            
        except Exception as e:
            return f"Semantic search error: {str(e)}"
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the agent with a given query.
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary containing the response and intermediate steps
        """
        try:
            result = self.agent.invoke({
                "input": query,
                "chat_history": []  # Add chat history support later
            })
            
            return {
                "response": result.get("output", ""),
                "intermediate_steps": result.get("intermediate_steps", []),
                "chat_history": result.get("chat_history", []),
                "model_used": self.model_name
            }
            
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "intermediate_steps": [],
                "chat_history": [],
                "model_used": self.model_name
            }
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of available model names
        """
        return self.llm_manager.list_available_models()
    
    def get_models_by_provider(self, provider: LLMProvider) -> List[str]:
        """
        Get list of models from a specific provider.
        
        Args:
            provider: LLM provider to filter by
            
        Returns:
            List of model names from the specified provider
        """
        return self.llm_manager.list_models_by_provider(provider)
    
    def set_model(self, model_name: str, api_key: str = None) -> None:
        """
        Change the model being used by the agent.
        
        Args:
            model_name: The new model name
            api_key: Optional new API key
        """
        if model_name not in self.llm_manager.configs:
            # Add new configuration if not exists
            config = self.llm_manager.get_config(model_name)
            if not config:
                raise ValueError(f"Model {model_name} not found in configurations")
            
            # If API key is provided, update the configuration
            if api_key:
                config.api_key = api_key
                self.llm_manager.add_config(config)
        
        self.model_name = model_name
        
        # Recreate the agent with the new model
        self.agent = self._create_agent()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        config = self.llm_manager.get_config(self.model_name)
        if not config:
            return {}
        
        return {
            "model_name": config.model_name,
            "provider": config.provider.value,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "is_valid": self.llm_manager.validate_config(self.model_name)
        }
    
    def test_current_model(self, test_query: str = "Hello, how are you?") -> Dict[str, Any]:
        """
        Test the current model with a sample query.
        
        Args:
            test_query: Test query to send to the model
            
        Returns:
            Dictionary containing test results
        """
        return self.llm_manager.test_model(self.model_name, test_query)
    
    def test_all_models(self, test_query: str = "Hello, how are you?") -> Dict[str, Any]:
        """
        Test all available models with a sample query.
        
        Args:
            test_query: Test query to send to the models
            
        Returns:
            Dictionary containing test results for all models
        """
        return self.llm_manager.test_all_models(test_query)
