#!/usr/bin/env python3
"""
Script to add sample test data to OpenSearch for hybrid search testing
"""
import json
import requests
import numpy as np

# Generate a 1024-dimensional test embedding vector
def generate_test_embedding():
    # Create a normalized random vector
    vector = np.random.normal(0, 1, 1024)
    # Normalize to unit length for cosine similarity
    vector = vector / np.linalg.norm(vector)
    return vector.tolist()

# Sample documents
sample_docs = [
    {
        "title": "OpenSearch Hybrid Search Guide",
        "body": "This guide explains how to implement hybrid search combining BM25 and vector similarity search in OpenSearch. It covers configuration, indexing, and querying techniques.",
        "section": "documentation",
        "page_id": "test-page-1",
        "content_type": "confluence",
        "source": "test",
        "embedding": generate_test_embedding()
    },
    {
        "title": "Vector Search Best Practices",
        "body": "Learn about vector search optimization, embedding models, and performance tuning for large-scale semantic search applications.",
        "section": "best-practices",
        "page_id": "test-page-2", 
        "content_type": "confluence",
        "source": "test",
        "embedding": generate_test_embedding()
    },
    {
        "title": "BM25 Scoring Algorithm",
        "body": "Understanding BM25 scoring function, term frequency, inverse document frequency, and how it ranks documents based on keyword relevance.",
        "section": "algorithms",
        "page_id": "test-page-3",
        "content_type": "confluence", 
        "source": "test",
        "embedding": generate_test_embedding()
    }
]

# Index the documents
base_url = "http://localhost:9200"
index_name = "khub-opensearch-index"

for i, doc in enumerate(sample_docs, 1):
    url = f"{base_url}/{index_name}/_doc/{i}"
    response = requests.post(url, json=doc, headers={"Content-Type": "application/json"})
    
    if response.status_code == 201:
        print(f"✓ Successfully indexed document {i}: {doc['title']}")
    else:
        print(f"✗ Failed to index document {i}: {response.text}")

# Refresh the index to make documents searchable
refresh_url = f"{base_url}/{index_name}/_refresh"
refresh_response = requests.post(refresh_url)
if refresh_response.status_code == 200:
    print("✓ Index refreshed successfully")
else:
    print(f"✗ Failed to refresh index: {refresh_response.text}")

print(f"\nAdded {len(sample_docs)} test documents to {index_name}")