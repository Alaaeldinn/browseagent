"""
BrowseAgent LangChain Agent Implementation

This module implements a proper AI Agent Level 2 with tool calling using LangChain.
The agent can use the DDGSSearchTool to perform web searches.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Union

# Import LangChain components with error handling
LANGCHAIN_AVAILABLE = False
try:
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_core.tools import BaseTool
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.schema import SystemMessage, HumanMessage, AIMessage
    from langchain_core.runnables import RunnableConfig
    from langchain_core.messages import BaseMessage
    LANGCHAIN_AVAILABLE = True
    print("LangChain components imported successfully")
except ImportError as e:
    print(f"LangChain import error: {e}")
    LANGCHAIN_AVAILABLE = False

# Import tools with error handling
TOOLS_AVAILABLE = False
try:
    from tools import DDGSSearchTool
    TOOLS_AVAILABLE = True
    print("Tools imported successfully")
except ImportError as e:
    print(f"Tools import error: {e}")
    TOOLS_AVAILABLE = False

from llm_config import get_available_models_info

class BrowseAgent:
    """BrowseAgent using LangChain with proper tool calling"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the agent with specified model"""
        self.model_name = model_name
        self.llm_config = get_available_models_info()
        self.tools = []
        self.agent = None
        self.agent_executor = None
        
        # Initialize tools if available
        if TOOLS_AVAILABLE:
            try:
                self.tools = [DDGSSearchTool()]
                print(f"Available tools: {[type(tool).__name__ for tool in self.tools]}")
            except Exception as e:
                print(f"Error initializing tools: {e}")
        
        self._setup_agent()
    
    def _setup_agent(self):
        """Setup the LangChain agent with tool calling"""
        if not LANGCHAIN_AVAILABLE:
            print("Warning: LangChain not available, using simplified implementation")
            self._setup_simple_agent()
            return
        
        try:
            print(f"Setting up LangChain agent with model: {self.model_name}")
            print(f"Available models: {list(self.llm_config.keys())}")
            
            # Initialize LLM
            llm_config = self.llm_config.get(self.model_name, {})
            if not llm_config.get('is_valid', False):
                print(f"Warning: Model {self.model_name} not properly configured")
                self._setup_simple_agent()
                return
            
            # Create LLM instance
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=0.7,
                openai_api_key=llm_config.get('api_key', '')
            )
            
            # Create the prompt template
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a helpful research assistant with access to web search capabilities.

When answering questions:
1. Use the search tool to find current and relevant information
2. Analyze the search results and provide a comprehensive answer
3. Cite specific sources when possible by including their titles and links
4. Be thorough, accurate, and up-to-date in your responses
5. If the search results are insufficient, explain what you found and suggest alternative approaches

Available tool:
- ddgs_search: Use this to search the web for current information

Example usage:
User: "What are the latest developments in artificial intelligence?"
Assistant: [Uses search tool to find current information about AI developments]

Remember to use the search tool when you need current information that might not be in your training data."""),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ])
            
            # Create the agent
            self.agent = create_openai_tools_agent(
                llm=llm,
                tools=self.tools,
                prompt=prompt
            )
            
            # Create the executor
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                max_iterations=10,
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
            
            print("LangChain agent setup completed successfully")
            
        except Exception as e:
            print(f"Error setting up LangChain agent: {e}")
            print("Falling back to simplified implementation")
            self._setup_simple_agent()
    
    def _setup_simple_agent(self):
        """Setup a simple agent that works without LangChain"""
        print("Using simplified agent implementation")
        self.agent = None
        self.agent_executor = None
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a research query using LangChain agent with tool calling"""
        try:
            if LANGCHAIN_AVAILABLE and self.agent_executor and len(self.tools) > 0:
                return await self._process_query_with_agent(query)
            else:
                return await self._process_query_simple(query)
                
        except Exception as e:
            return {
                "error": f"Error processing query: {str(e)}",
                "response": "",
                "sources": [],
                "intermediate_steps": [],
                "model_used": self.model_name
            }
    
    async def _process_query_with_agent(self, query: str) -> Dict[str, Any]:
        """Process query using LangChain agent with tool calling"""
        try:
            # Run the agent
            result = await self.agent_executor.ainvoke({"input": query})
            
            # Extract the response
            response = result.get("output", "No response generated.")
            
            # Extract intermediate steps
            intermediate_steps = []
            for step in result.get("intermediate_steps", []):
                if isinstance(step, tuple) and len(step) >= 2:
                    action = step[0]
                    observation = step[1]
                    intermediate_steps.append(f"Action: {action}\nObservation: {str(observation)[:500]}...")
            
            # Extract sources from the agent's response or intermediate steps
            sources = self._extract_sources_from_response(response, intermediate_steps)
            
            return {
                "response": response,
                "sources": sources,
                "intermediate_steps": intermediate_steps,
                "model_used": self.model_name
            }
            
        except Exception as e:
            return {
                "error": f"Error processing query with LangChain agent: {str(e)}",
                "response": "",
                "sources": [],
                "intermediate_steps": [],
                "model_used": self.model_name
            }
    
    def _extract_sources_from_response(self, response: str, intermediate_steps: List[str]) -> List[Dict[str, Any]]:
        """Extract sources from the agent's response and intermediate steps"""
        sources = []
        
        # Try to extract sources from intermediate steps first
        for step in intermediate_steps:
            if "http" in step:
                # Extract URLs from the step
                lines = step.split('\n')
                for line in lines:
                    if "http" in line:
                        sources.append({
                            "title": f"Source {len(sources) + 1}",
                            "link": line.strip()
                        })
        
        # If no sources found in steps, try to extract from response
        if not sources and "http" in response:
            lines = response.split('\n')
            for line in lines:
                if "http" in line and len(line.strip()) > 10:
                    sources.append({
                        "title": f"Source {len(sources) + 1}",
                        "link": line.strip()
                    })
        
        return sources[:5]  # Limit to 5 sources
    
    async def _process_query_simple(self, query: str) -> Dict[str, Any]:
        """Process query using a simple approach (fallback)"""
        try:
            # Use the search tool directly if available
            if len(self.tools) > 0:
                return await self._process_with_tool(query)
            else:
                return await self._process_without_tool(query)
                
        except Exception as e:
            return {
                "error": f"Error processing query with simple approach: {str(e)}",
                "response": "",
                "sources": [],
                "intermediate_steps": [],
                "model_used": self.model_name
            }
    
    async def _process_with_tool(self, query: str) -> Dict[str, Any]:
        """Process query using the search tool directly"""
        search_result = ""
        intermediate_steps = []
        
        try:
            # Use the search tool
            search_tool = self.tools[0]
            search_results = await search_tool._arun(query)
            
            if search_results and isinstance(search_results, list) and len(search_results) > 0:
                # Format the search results
                search_result = "Search Results:\n"
                for result in search_results[:5]:  # Limit to 5 results
                    if isinstance(result, dict) and 'title' in result:
                        search_result += f"- {result.get('title', 'No title')}: {result.get('link', 'No link')}\n"
                        intermediate_steps.append(f"Found result: {result.get('title', 'No title')}")
            else:
                search_result = "No search results found."
                intermediate_steps.append("Search tool returned no results")
                
        except Exception as e:
            intermediate_steps.append(f"Search tool failed: {str(e)}")
            search_result = f"Search failed: {str(e)}"
        
        # Create a response based on the search results
        response = f"I searched for information about: {query}\n\n"
        response += f"Search results: {search_result[:1000]}...\n\n"
        response += "Based on the search results, here's what I found:\n\n"
        
        # Add some analysis
        if "http" in search_result:
            response += "I found several relevant sources that you can explore for more detailed information.\n"
            sources = []
            lines = search_result.split('\n')
            for line in lines:
                if "http" in line and len(line.strip()) > 10:
                    sources.append({
                        "title": f"Source {len(sources) + 1}",
                        "link": line.strip()
                    })
            response += f"Found {len(sources)} sources.\n"
        else:
            response += "The search returned results, but no specific web links were found.\n"
            sources = []
        
        # Add model information
        model_info = self.llm_config.get(self.model_name, {})
        if model_info:
            response += f"\nModel Information:\n"
            response += f"- Model: {self.model_name}\n"
            response += f"- Provider: {model_info.get('provider', 'unknown')}\n"
            response += f"- Valid: {'Yes' if model_info.get('is_valid', False) else 'No'}\n"
        
        return {
            "response": response,
            "sources": sources[:5],
            "intermediate_steps": intermediate_steps,
            "model_used": self.model_name
        }
    
    async def _process_without_tool(self, query: str) -> Dict[str, Any]:
        """Process query without search tool (fallback)"""
        response = f"I searched for information about: {query}\n\n"
        response += "Unfortunately, the search tool is not currently available. "
        response += "This could be due to dependency conflicts or missing configurations.\n\n"
        response += "Please check the following:\n"
        response += "1. Ensure all dependencies are properly installed\n"
        response += "2. Check that the search tool is correctly configured\n"
        response += "3. Verify that the environment variables are set correctly\n\n"
        response += "Model Information:\n"
        
        # Add model information
        model_info = self.llm_config.get(self.model_name, {})
        if model_info:
            response += f"- Model: {self.model_name}\n"
            response += f"- Provider: {model_info.get('provider', 'unknown')}\n"
            response += f"- Valid: {'Yes' if model_info.get('is_valid', False) else 'No'}\n"
        
        return {
            "response": response,
            "sources": [],
            "intermediate_steps": ["Search tool not available"],
            "model_used": self.model_name
        }
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.llm_config.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_name in self.llm_config:
            return {
                "name": model_name,
                "provider": self.llm_config[model_name].get("provider", "unknown"),
                "available": True,
                **self.llm_config[model_name]
            }
        return {"name": model_name, "available": False}

# Test function
async def test_agent():
    """Test the agent with a sample query"""
    try:
        agent = BrowseAgent()
        result = await agent.process_query("What are the latest developments in artificial intelligence?")
        print("Agent test result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Agent test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_agent())
