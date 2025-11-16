from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from tools import DDGSSearchTool, SemanticSearchTool
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BrowseAgent:
    """
    The main BrowseAgent class that orchestrates web search and semantic analysis.
    
    This agent uses LangChain's framework to:
    1. Search the web using DuckDuckGo
    2. Apply semantic search to filter and rank results
    3. Generate responses based on the most relevant information
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None):
        """
        Initialize the BrowseAgent.
        
        Args:
            model_name: The LLM model to use (default: "gpt-3.5-turbo")
            api_key: OpenAI API key (can also be set via OPENAI_API_KEY environment variable)
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
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
        
        # Initialize the LLM
        llm = ChatOpenAI(
            model_name=self.model_name,
            api_key=self.api_key,
            temperature=0.1  # Lower temperature for more consistent responses
        )
        
        # Create the agent
        agent = create_openai_tools_agent(
            llm=llm,
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
                "chat_history": result.get("chat_history", [])
            }
            
        except Exception as e:
            return {
                "response": f"Error: {str(e)}",
                "intermediate_steps": [],
                "chat_history": []
            }
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List of available model names
        """
        return [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-32k"
        ]
    
    def set_model(self, model_name: str, api_key: str = None) -> None:
        """
        Change the model being used by the agent.
        
        Args:
            model_name: The new model name
            api_key: Optional new API key
        """
        self.model_name = model_name
        self.api_key = api_key or self.api_key
        
        # Recreate the agent with the new model
        self.agent = self._create_agent()
