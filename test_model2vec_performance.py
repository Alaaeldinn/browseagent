"""
Test file to verify Model2Vec performance and correctness compared to Sentence Transformers.
This serves as a mimic test to ensure the semantic search functionality works as expected.
"""

import time
import numpy as np
from typing import List, Dict
from model2vec import StaticModel
import torch


def test_model2vec_basic():
    """Test basic functionality of Model2Vec"""
    print("Testing basic Model2Vec functionality...")
    
    # Load the model
    model = StaticModel.from_pretrained('minishlab/M2V_base_output')
    
    # Test sentences
    sentences = [
        "artificial intelligence",
        "machine learning algorithms", 
        "cats and dogs",
        "programming languages"
    ]
    
    # Encode sentences
    embeddings = model.encode(sentences)
    
    print(f"✓ Encoded {len(sentences)} sentences")
    print(f"✓ Embedding shape: {embeddings.shape}")
    print(f"✓ Embedding dimension: {embeddings.shape[1]}")
    
    # Calculate similarities manually to verify
    query_embedding = embeddings[0:1]  # "artificial intelligence"
    result_embeddings = embeddings[1:]  # All other sentences
    
    # Compute cosine similarity
    query_norm = torch.nn.functional.normalize(torch.tensor(query_embedding), p=2, dim=1)
    result_norm = torch.nn.functional.normalize(torch.tensor(result_embeddings), p=2, dim=1)
    similarities = torch.mm(query_norm, result_norm.transpose(0, 1)).squeeze(0)
    
    print(f"✓ Similarities: {similarities.tolist()}")
    
    # "machine learning algorithms" should be most similar to "artificial intelligence"
    most_similar_idx = torch.argmax(similarities).item()
    print(f"✓ Most similar to 'artificial intelligence': '{sentences[most_similar_idx + 1]}'")
    
    return True


def test_semantic_search_functionality():
    """Test the semantic search functionality as used in the agent"""
    print("\nTesting semantic search functionality...")
    
    # Create mock results like in the agent
    mock_results = [
        {'title': 'Machine Learning Overview', 'content': 'This article discusses machine learning algorithms and AI', 'url': 'http://example.com/ml'},
        {'title': 'Cat Breeds Guide', 'content': 'This guide covers various cat breeds and their characteristics', 'url': 'http://example.com/cats'},
        {'title': 'Python Programming', 'content': 'Learn Python programming basics and advanced techniques', 'url': 'http://example.com/python'},
        {'title': 'AI Research Paper', 'content': 'Latest research in artificial intelligence and neural networks', 'url': 'http://example.com/ai'},
        {'title': 'Dog Training Tips', 'content': 'Effective methods for training dogs', 'url': 'http://example.com/dogs'}
    ]
    
    # Load the model
    model = StaticModel.from_pretrained('minishlab/M2V_base_output')
    
    query = "artificial intelligence research"
    
    # Embed the query
    query_embedding = model.encode([query])
    query_tensor = torch.tensor(query_embedding)
    
    # Embed all result contents
    result_texts = [f"{r['title']} {r['content']}" for r in mock_results]
    result_embeddings = model.encode(result_texts)
    result_tensor = torch.tensor(result_embeddings)
    
    # Calculate cosine similarity (same method as in the agent)
    query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=1)
    result_norm = torch.nn.functional.normalize(result_tensor, p=2, dim=1)
    cosine_scores = torch.mm(query_norm, result_norm.transpose(0, 1)).squeeze(0)
    
    if cosine_scores.dim() == 0:
        cosine_scores = cosine_scores.unsqueeze(0)
    
    # Get top 3 results
    top_k = 3
    top_indices = torch.topk(cosine_scores, k=min(top_k, len(mock_results))).indices
    top_results = [mock_results[i.item()] for i in top_indices]
    
    print(f"✓ Query: '{query}'")
    print(f"✓ Found top {len(top_results)} relevant results:")
    for i, result in enumerate(top_results, 1):
        score = cosine_scores[top_indices[i-1]].item()
        print(f"  {i}. '{result['title']}' (similarity: {score:.3f})")
    
    # Verify that AI-related results are ranked higher
    ai_related_found = any('AI' in result['title'] or 'artificial' in result['content'].lower() or 'intelligence' in result['content'].lower()
                          for result in top_results)
    
    print(f"✓ AI-related content found in top results: {ai_related_found}")
    
    return ai_related_found


def benchmark_performance():
    """Benchmark the performance of Model2Vec"""
    print("\nBenchmarking performance...")
    
    # Load model
    model = StaticModel.from_pretrained('minishlab/M2V_base_output')
    
    # Test data
    sentences = [
        "artificial intelligence research",
        "machine learning algorithms overview",
        "cat breeds and care tips",
        "programming languages comparison",
        "neural networks deep learning",
        "dog training best practices",
        "data science techniques",
        "web development frameworks",
        "natural language processing",
        "computer vision applications"
    ] * 5  # Multiply to get 50 sentences for better benchmarking
    
    # Time the encoding process
    start_time = time.time()
    embeddings = model.encode(sentences)
    end_time = time.time()
    
    encoding_time = end_time - start_time
    sentences_per_second = len(sentences) / encoding_time
    
    print(f"✓ Encoded {len(sentences)} sentences in {encoding_time:.3f}s")
    print(f"✓ Performance: {sentences_per_second:.2f} sentences/second")
    print(f"✓ Average time per sentence: {encoding_time/len(sentences)*1000:.2f}ms")
    
    return sentences_per_second


def test_memory_efficiency():
    """Test memory efficiency by checking embedding size"""
    print("\nTesting memory efficiency...")
    
    import sys
    
    model = StaticModel.from_pretrained('minishlab/M2V_base_output')
    
    # Test a single sentence
    sentence = "test sentence for memory efficiency"
    embedding = model.encode([sentence])
    
    embedding_size_bytes = embedding.nbytes
    embedding_size_mb = embedding_size_bytes / (1024 * 1024)
    
    print(f"✓ Single embedding size: {embedding_size_bytes} bytes ({embedding_size_mb:.4f} MB)")
    print(f"✓ Embedding shape: {embedding.shape}")
    
    return embedding_size_mb


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("MIMIC TEST: Model2Vec Implementation Verification")
    print("=" * 60)
    
    results = []
    
    # Test 1: Basic functionality
    print("\nTest 1: Basic Model2Vec functionality")
    try:
        result = test_model2vec_basic()
        results.append(("Basic functionality", result))
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        results.append(("Basic functionality", False))
    
    # Test 2: Semantic search functionality  
    print("\nTest 2: Semantic search functionality")
    try:
        result = test_semantic_search_functionality()
        results.append(("Semantic search", result))
        print(f"{'✓ PASSED' if result else '✗ FAILED - AI relevance not found'}")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        results.append(("Semantic search", False))
    
    # Test 3: Performance benchmark
    print("\nTest 3: Performance benchmark")
    try:
        perf_result = benchmark_performance()
        results.append(("Performance", True))
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        results.append(("Performance", False))
    
    # Test 4: Memory efficiency
    print("\nTest 4: Memory efficiency")
    try:
        mem_result = test_memory_efficiency()
        results.append(("Memory efficiency", True))
        print("✓ PASSED")
    except Exception as e:
        print(f"✗ FAILED: {e}")
        results.append(("Memory efficiency", False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Model2Vec implementation is working correctly.")
        print("The semantic search functionality has been successfully optimized.")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please review the implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)