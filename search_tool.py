from typing import List, Dict, Any
import DDGS
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


def search_ddgs(query: str, max_results: int = 200) -> List[Dict[str, Any]]:
    """
    Perform a search using DuckDuckGo Search
    """
    results = DDGS().text(
        query, 
        max_results=max_results,
        region="wt-wt",
        safesearch="off",
        timelimit='y'
    )
    return results

def semantic_search_similarity(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Perform semantic search similarity using sentence transformers
    """
    if not results:
        return []
    
    # Initialize the model and tokenizer
    model_name = "all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Encode the query
    query_embedding = model(**tokenizer(query, return_tensors="pt", padding=True, truncation=True))
    query_embedding = query_embedding.last_hidden_state.mean(dim=1)
    
    # Prepare result texts for encoding
    result_texts = [result.get('body', '') or result.get('snippet', '') or result.get('title', '') for result in results]
    
    # Encode all result texts
    inputs = tokenizer(result_texts, return_tensors="pt", padding=True, truncation=True)
    result_embeddings = model(**inputs)
    result_embeddings = result_embeddings.last_hidden_state.mean(dim=1)
    
    # Calculate cosine similarity between query and results
    similarities = util.cos_sim(query_embedding, result_embeddings)[0]
    
    # Add similarity scores to results
    for i, result in enumerate(results):
        result['similarity_score'] = similarities[i].item()
    
    # Sort results by similarity score in descending order
    sorted_results = sorted(results, key=lambda x: x.get('similarity_score', 0), reverse=True)
    
    return sorted_results

def top_5(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return the top 5 results based on similarity scores
    """
    return results[:5]


class SearchToolInput(BaseModel):
    query: str = Field(description="The search query to find information about")


class SearchTool(BaseTool):
    name = "search_tool"
    description = "Useful for searching the web for current information on any topic"
    args_schema = SearchToolInput

    def _run(self, query: str) -> str:
        """
        Run the search tool with the given query
        """
        try:
            # Step 1: Perform initial search
            initial_results = search_ddgs(query, max_results=200)
            
            # Step 2: Apply semantic search similarity
            semantic_results = semantic_search_similarity(query, initial_results)
            
            # Step 3: Get top 5 results
            final_results = top_5(semantic_results)
            
            # Format results for the LLM
            formatted_results = []
            for result in final_results:
                formatted_result = {
                    "title": result.get("title", ""),
                    "href": result.get("href", ""),
                    "body": result.get("body", ""),
                    "similarity_score": round(result.get("similarity_score", 0), 4)
                }
                formatted_results.append(formatted_result)
            
            return str(formatted_results)
        
        except Exception as e:
            return f"Error occurred during search: {str(e)}"
    
    def _arun(self, query: str):
        """
        Async version of the search tool
        """
        raise NotImplementedError("SearchTool does not support async")
