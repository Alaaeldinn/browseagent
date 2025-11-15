import pandas as pd
from ddgs import DDGS
search_query = "DDGS Documentation"

def search_ddgs(query, max_results=200) -> list:
    results = DDGS().text(
        query, 
        max_results=max_results,
        region="wt-wt",
        safesearch="off",
        timelimit='y'
    )
    return results
