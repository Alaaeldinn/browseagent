#!/usr/bin/env python3
"""
Example script demonstrating LLM Integration in BrowseAgent

This script shows how to:
1. Configure multiple LLM providers
2. Switch between different models
3. Test model performance
4. Use the agent with different LLMs
"""

import os
from dotenv import load_dotenv
from llm_config import LLMManager, LLMConfig, LLMProvider
from agent import BrowseAgent

# Load environment variables
load_dotenv()

def main():
    """Main function demonstrating LLM integration"""
    
    print("=== BrowseAgent LLM Integration Demo ===\n")
    
    # Initialize LLM Manager
    print("1. Initializing LLM Manager...")
    llm_manager = LLMManager()
    
    # Show available models
    print("\n2. Available Models:")
    models = llm_manager.list_available_models()
    for i, model in enumerate(models, 1):
        config = llm_manager.get_config(model)
        provider = config.provider.value if config else "Unknown"
        is_valid = llm_manager.validate_config(model)
        status = "✓" if is_valid else "✗"
        print(f"   {i}. {model} ({provider}) {status}")
    
    if not models:
        print("   No models configured. Please set up API keys in your .env file.")
        print("\nSupported providers and their environment variables:")
        print("   - OpenAI: OPENAI_API_KEY")
        print("   - Anthropic: ANTHROPIC_API_KEY")
        print("   - Google: GOOGLE_API_KEY")
        print("   - Cohere: COHERE_API_KEY")
        print("   - Azure: AZURE_API_KEY, AZURE_API_BASE, AZURE_API_VERSION")
        return
    
    # Initialize BrowseAgent
    print(f"\n3. Initializing BrowseAgent with default model: {llm_manager.get_default_model()}")
    try:
        agent = BrowseAgent(llm_manager=llm_manager)
        print("   ✓ Agent initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize agent: {e}")
        return
    
    # Show current model info
    print("\n4. Current Model Information:")
    model_info = agent.get_model_info()
    for key, value in model_info.items():
        print(f"   {key}: {value}")
    
    # Test current model
    print("\n5. Testing Current Model:")
    test_result = agent.test_current_model("Hello! Can you tell me what 2+2 equals?")
    if test_result["success"]:
        print(f"   ✓ Model test successful")
        print(f"   Response: {test_result['response'][:100]}...")
        if test_result.get("tokens_used"):
            print(f"   Tokens used: {test_result['tokens_used']}")
    else:
        print(f"   ✗ Model test failed: {test_result['error']}")
    
    # Demonstrate model switching
    if len(models) > 1:
        print("\n6. Model Switching Demo:")
        original_model = agent.model_name
        print(f"   Original model: {original_model}")
        
        # Switch to a different model
        new_model = models[1] if models[0] == original_model else models[0]
        print(f"   Switching to: {new_model}")
        
        agent.set_model(new_model)
        print(f"   ✓ Current model is now: {agent.model_name}")
        
        # Test the new model
        test_result = agent.test_current_model("What's the capital of France?")
        if test_result["success"]:
            print(f"   ✓ New model test successful")
            print(f"   Response: {test_result['response'][:100]}...")
        else:
            print(f"   ✗ New model test failed: {test_result['error']}")
        
        # Switch back to original
        agent.set_model(original_model)
        print(f"   ✓ Switched back to: {agent.model_name}")
    
    # Test all models
    print("\n7. Testing All Models:")
    all_results = agent.test_all_models("Briefly introduce yourself.")
    print(f"   Tested with query: 'Briefly introduce yourself.'")
    
    for model, result in all_results.items():
        status = "✓" if result["success"] else "✗"
        print(f"   {status} {model}: {result.get('error', 'Success')}")
    
    # Demonstrate agent with different models
    print("\n8. Agent Query with Different Models:")
    test_query = "What are the main benefits of renewable energy?"
    
    if len(models) >= 2:
        # Test with first two models
        for i, model in enumerate(models[:2]):
            print(f"\n   Testing with {model}:")
            agent.set_model(model)
            
            # Run a simple query
            result = agent.run(test_query)
            
            if result["response"]:
                print(f"   ✓ Response received")
                print(f"   Preview: {result['response'][:150]}...")
                print(f"   Model used: {result.get('model_used', model)}")
            else:
                print(f"   ✗ No response received")
    
    print("\n=== Demo Complete ===")
    print("\nTo use different models, set their API keys in your .env file:")
    print("   - OpenAI: OPENAI_API_KEY=your_key")
    print("   - Anthropic: ANTHROPIC_API_KEY=your_key")
    print("   - Google: GOOGLE_API_KEY=your_key")
    print("   - etc.")


if __name__ == "__main__":
    main()
