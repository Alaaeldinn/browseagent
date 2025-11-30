"""
Models module for BrowseAgent
Contains Pydantic models and data schemas
"""

from typing import List
from pydantic import BaseModel


class SearchResult(BaseModel):
    title: str
    link: str  # Using 'link' instead of 'url' for consistency
    snippet: str  # Using 'snippet' instead of 'content' for API compatibility


class ResearchResponse(BaseModel):
    keywords: List[str]
    results: List[SearchResult]
    answer: str