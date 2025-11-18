# BrowseAgent

BrowseAgent is a  AI research agent designed to search, browse, and synthesize information from the web. 

It helps you turn any query into structured insights using LLM reasoning and automated search tools.

help me think through time how to break this into iterative pieces , todos list and write a plane.md that have phases and todoslist only

browseagent is ai agent level 2 applying tool calling 

Requirements:
	the user enters quey 
    the llm extract the best words from the query 
    the llm making tool calling to the search tool with the best keywords
        the search tool :
            results = ddgs cutsom search function 
            semantic results = seamantic search similarity(query , results) 
            final results = top_5 (semantic results)
    llm synthesize final results 
    the user can select various llm via litellm 
    the ui is too simple , no complex patterns or coding 
    write requirements.txt for the project 
    imports the libs correctly is a must 

rules: 
    no complex coding or patterns 

ddgs custom search function: 
    def search_ddgs(query, max_results=200) -> list:
    results = DDGS().text(
        query, 
        max_results=max_results,
        region="wt-wt",
        safesearch="off",
        timelimit='y'
    )
    return results

Desgin:
	minimal functional , practical 
	intentional use of color 

Frontend : 	
	fasthtml
    
Backend:
	Fastapi 
    langchain
    sentence transformers 
    litellm
    DDGS

check off iterms in the plan as we accomplish them as todolist , if you have open questions that require my input , add those in the plan as well , we will excute phase by phase under my command


