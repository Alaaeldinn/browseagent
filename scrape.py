import pandas as pd
from ddgs import DDGS
search_query = "DDGS Documentation"
results = DDGS().text(
    search_query, 
    max_results=250,
    region="wt-wt",
    safesearch="off",
    timelimit='y'
    )
df = pd.DataFrame(results)
df.to_csv("python_tutorials_search_results.csv", index=False)