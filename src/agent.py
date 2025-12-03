#import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
#from langsmith import Client
from model2vec import StaticModel
from langchain_ollama import ChatOllama
import torch
# Import models and tools from other modules
from src.models import SearchResult, ResearchResponse
from src.tools import searxng_search

load_dotenv()


# ============================================================================
# RESEARCH AGENT
# ============================================================================

class ResearchAgent:
    def __init__(self):
        # Initialize LLM with OpenRouter configuration (keep as is)
        self.llm =  ChatOllama(
            model="llama3.2",
            temperature=0.7,
        )

        # Initialize Model2Vec for faster semantic similarity
        # Using a pre-trained model from HuggingFace that provides good performance with speed
        self.embedding_model = StaticModel.from_pretrained('minishlab/M2V_base_output')

        # Create the agent with search tool
        self.tools = [searxng_search]
        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def _create_agent(self):
        """Create a tool-calling agent for research"""
        from langchain_core.prompts import ChatPromptTemplate

        # Create a proper prompt for the tool-calling agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant. When the user asks a question:
1. Generate optimal search keywords or statements to find relevant information
2. Use the searxng_search tool to search for information
3. You will receive search results to work with

Be concise and focused in your searches."""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        return create_tool_calling_agent(self.llm, self.tools, prompt)

    def generate_keywords(self, query: str) -> List[str]:
        """Generate search keywords using the LLM"""
        from langchain_core.output_parsers import StrOutputParser
        import re
        import json

        prompt = ChatPromptTemplate.from_template(
            "Generate 3-5 optimal search keywords or short phrases for this research topic: {topic}\n"
            "Return ONLY a JSON array of strings with no additional text. Example: [\"keyword1\", \"keyword2\", \"keyword3\"]"
        )

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"topic": query})

        # Extract JSON from response (in case the LLM adds extra text)
        # Look for JSON array in the response
        json_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
        if json_match:
            json_str = '[' + json_match.group(1) + ']'
            try:
                keywords = json.loads(json_str)
                # Ensure all items are strings
                keywords = [str(k) for k in keywords]
                print(f"\n📝 Generated keywords: {keywords}")
                return keywords
            except json.JSONDecodeError:
                # If parsing fails, return a default list
                print(f"\n📝 Failed to parse keywords, using default approach")
                # Fallback: split the cleaned match by commas and clean each part
                items = [item.strip().strip('"\'') for item in json_str.split(',')]
                items = [item for item in items if item]
                return items
        else:
            # If no JSON array found, try to parse the entire response
            try:
                keywords = json.loads(response.strip())
                print(f"\n📝 Generated keywords: {keywords}")
                return keywords
            except json.JSONDecodeError:
                # Last resort: simple split
                print(f"\n📝 Failed to parse keywords, using default: {query}")
                return [query]

    def semantic_search(self, query: str, results: List[Dict[str, str]], top_k: int = 5) -> List[Dict[str, str]]:
        """
        Perform semantic similarity search on results using Model2Vec.
        Returns top_k most similar results to the original query.
        """
        if not results:
            return []

        # Embed the original query
        query_embedding = self.embedding_model.encode([query])

        # Convert to torch tensor
        query_embedding = torch.tensor(query_embedding)

        # Embed all result contents (metadata + body)
        # Combine title and content for better semantic matching
        result_texts = [f"{r['title']} {r['content']}" for r in results]
        result_embeddings = self.embedding_model.encode(result_texts)

        # Convert to torch tensor
        result_embeddings = torch.tensor(result_embeddings)

        # Calculate cosine similarity correctly
        # query_embedding shape: [1, embedding_dim], result_embeddings shape: [num_results, embedding_dim]
        # Need to compute similarity between query and each result
        query_norm = torch.nn.functional.normalize(query_embedding, p=2, dim=1)  # [1, embedding_dim]
        result_norm = torch.nn.functional.normalize(result_embeddings, p=2, dim=1)  # [num_results, embedding_dim]

        # Matrix multiplication to get cosine similarities
        cosine_scores = torch.mm(query_norm, result_norm.transpose(0, 1)).squeeze(0)  # [num_results]

        # Handle case where there's only one result
        if cosine_scores.dim() == 0:
            cosine_scores = cosine_scores.unsqueeze(0)

        # Get top-k results
        top_k = min(top_k, len(results))
        top_indices = torch.topk(cosine_scores, k=top_k).indices

        # Convert tensor indices to Python integers for list indexing
        top_results = [results[i.item()] for i in top_indices]
        print(f"\n🔍 Semantic search: Selected top {len(top_results)} most relevant results")
        return top_results

    def generate_answer(self, query: str, context_results: List[Dict[str, str]]) -> str:
        """Generate an answer based on the search results context"""
        # Format context from search results
        context_str = "\n\n".join([
            f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
            for r in context_results
        ])

        prompt = ChatPromptTemplate.from_template(
            """You are a research assistant. Answer the user's question based ONLY on the following search results:

Context:
{context}

Question: {question}

Provide a comprehensive answer based on the context. If the context doesn't contain enough information, say so."""
        )

        from langchain_core.output_parsers import StrOutputParser
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "question": query})
        return answer

    def run(self, user_query: str) -> ResearchResponse:
        """
        Main research flow:
        1. Generate keywords
        2. Search using tool calling
        3. Semantic similarity filtering
        4. Generate answer from context
        """
        print(f"\n{'='*70}")
        print(f"🔬 Research Query: {user_query}")
        print(f"{'='*70}")

        # Step 1: Generate keywords
        keywords = self.generate_keywords(user_query)

        # Step 2: Perform search with generated keywords
        search_query = " ".join(keywords[:3])  # Use top 3 keywords
        print(f"\n🔎 Searching for: {search_query}")

        search_results = searxng_search.invoke(search_query)

        # Step 3: Semantic similarity search
        top_results = self.semantic_search(user_query, search_results, top_k=5)

        # Step 4: Generate answer from context
        if top_results:
            print(f"\n💬 Generating answer from {len(top_results)} results...")
            answer = self.generate_answer(user_query, top_results)
        else:
            answer = "I'm sorry, I couldn't find relevant information to answer your question."

        # Format response
        formatted_results = [
            SearchResult(
                title=r['title'],
                link=r['url'],
                snippet=r['content']
            )
            for r in top_results
        ]

        return ResearchResponse(
            keywords=keywords,
            results=formatted_results,
            answer=answer
        )


# ============================================================================
# GLOBAL AGENT INSTANCE
# ============================================================================

agent = ResearchAgent()
