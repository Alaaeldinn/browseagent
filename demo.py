#!/usr/bin/env python3
"""
BrowseAgent Demo Script

This script allows users to:
1. Choose an LLM provider from a list
2. Enter their API key
3. Enter a query
4. Get the response from the agent
"""

import os
import requests
from getpass import getpass

def main():
    print("Welcome to BrowseAgent Demo!")
    print("="*50)
    
    # List of supported providers
    providers = [
        "openai/gpt-3.5-turbo",
        "openai/gpt-4",
        "anthropic/claude-3-opus",
        "anthropic/claude-3-sonnet",
        "anthropic/claude-3-haiku",
        "google/gemini-pro",
        "mistral/mistral-large-latest"
    ]
    
    print("\nAvailable LLM Providers:")
    for i, provider in enumerate(providers, 1):
        print(f"{i}. {provider}")
    
    # Get provider selection
    while True:
        try:
            choice = int(input(f"\nSelect a provider (1-{len(providers)}): "))
            if 1 <= choice <= len(providers):
                selected_provider = providers[choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(providers)}")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\nYou selected: {selected_provider}")
    
    # Get API key securely
    api_key = getpass("Enter your API key: ")
    
    # Set the API key in environment based on provider type
    if selected_provider.startswith("openai/"):
        os.environ["OPENAI_API_KEY"] = api_key
    elif selected_provider.startswith("anthropic/"):
        os.environ["ANTHROPIC_API_KEY"] = api_key
    elif selected_provider.startswith("google/"):
        os.environ["GOOGLE_API_KEY"] = api_key
    elif selected_provider.startswith("mistral/"):
        os.environ["MISTRAL_API_KEY"] = api_key
    
    # Get user query
    query = input("\nEnter your query: ").strip()
    if not query:
        print("Query cannot be empty!")
        return
    
    print("\nProcessing your request...")
    print("This may take a moment...\n")
    
    try:
        # Call the BrowseAgent API
        backend_url = "http://localhost:8000/query"
        payload = {
            "query": query,
            "llm_provider": selected_provider
        }
        
        response = requests.post(backend_url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("Response from BrowseAgent:")
            print("-" * 30)
            print(result.get("result", "No result returned"))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to BrowseAgent backend.")
        print("Please make sure the FastAPI server is running on http://localhost:8000")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    print("\nThank you for using BrowseAgent!")

if __name__ == "__main__":
    main()