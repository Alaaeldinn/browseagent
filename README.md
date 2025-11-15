# BrowseAgent

BrowseAgent is a lightweight AI-powered research agent designed to search, browse, and synthesize information from the web.  
It helps you turn any query into structured insights using LLM reasoning + automated search tools.
lets build a web app AI Research Agent to assist users in conducting research efficiently. It takes a user query, uses an LLM to extract and optimize keywords, searches for relevant information using a custom search tool, and synthesizes the results into a coherent response.

use whatever libraries for the logic core as not everything is covered 
help me think through time how to break this into iterative pieces and write a plane.md 


app logic behavior:
    user enters query 
    ai agent refine query and suggest the best keywords to search with 
    ai agent tool calling the custom search tool 
    add the the top 5 results to the context 
    generate response 
    
User Query --> LLm keyword Extraction ----> rag semantic search + custom search tool  ---> embed top 5 topics to context ---> llm response

Requirements 
	provide multiple llms using litellm lib
	allow user to use these multiple by using their api token
	add unit tests for busniess logic 
	use git and use description commits 

additional: 
    custom search tool :
        use the snippet code in scrape.py as custom search tool 
        get the results 
        apply rag semantic search ( query , results )
        return top 5 topics 

Desgin:

	minimal functional , practical 
	intentional use of color 
	waremer tones 
	inspired by browser and ai browsers


Frontend : 	
	Next.js and React
	Tailwind CSS v4
    
Backend:
	Fastapi 
    langchain

check off iterms in the plan as we accomplish them as todolist , if you have open questions that require my input , add those in the plan as well , we will excute phase by phase under my command


