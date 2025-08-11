# opensearch.py - OpenSearch integration module
import logging
import os
import requests

from client_manager import ClientSingleton
from utils import load_config

os.environ["http_proxy"] = "proxy.jpmchase.net:10443"
os.environ["https_proxy"] = "proxy.jpmchase.net:10443"
if 'no_proxy' in os.environ:
    os.environ['no_proxy'] = os.environ['no_proxy'] + ",jpmchase.net" + ",openai.azure.com"
else:
    os.environ['no_proxy'] = 'localhost,127.0.0.1,jpmchase.net,openai.azure.com'

logger = logging.getLogger()
logger.setLevel(logging.INFO)
config = load_config()
opensearch_endpoint = config['aws_info']['opensearch_endpoint']
awsauth = ClientSingleton.get_awsauth()


def query_opensearchToFetchAllIndices():
    """Query OpenSearch to retrieve chunk+metadata doc based on the index name and document ID prefix."""
    
    url = f"{opensearch_endpoint}/_cat/indices?v"
    response = requests.get(url, auth=awsauth)
    
    if response.status_code == 200:
        indices_info = response.text
        print("List of indices and their details:")
        print(indices_info)
        # result = response.json()
        # return result['hits']['hits']
    else:
        logger.error("Failed to query OpenSearch. Status code: %d, Response: %s",
                     response.status_code, response.text)
        return []


def get_doc_ids(index_name):
    """Query OpenSearch to get document IDs from the specified index."""
    
    url = f"{opensearch_endpoint}/{index_name}/_search"
    
    # Query to retrieve only document IDs
    query = {
        "query": {
            "match_all": {}
        },
        "_source": False,  # Don't return document content, only metadata
        "size": 1000  # Adjust size as needed
    }
    
    response = requests.post(url, json=query, auth=awsauth)
    
    if response.status_code == 200:
        results = response.json()
        doc_ids = [hit['_id'] for hit in results['hits']['hits']]
        return doc_ids
    else:
        logger.error("Failed to get document IDs. Status code: %d, Response: %s",
                     response.status_code, response.text)
        return []


def opensearch_bulk_search(index_name, doc_ids):
    """Perform bulk search to retrieve multiple documents by their IDs."""
    
    url = f"{opensearch_endpoint}/{index_name}/_mget"
    
    # Build the request body for bulk retrieval
    body = {
        "ids": doc_ids
    }
    
    response = requests.post(url, json=body, auth=awsauth)
    
    if response.status_code == 200:
        results = response.json()
        documents = []
        
        for doc in results.get('docs', []):
            if doc.get('found'):
                documents.append(doc['_source'])
        
        return documents
    else:
        logger.error("Failed bulk search. Status code: %d, Response: %s",
                     response.status_code, response.text)
        return []


def opensearch_semantic_search(index_name, query_vector, size=5):
    """Perform semantic search using vector similarity."""
    
    url = f"{opensearch_endpoint}/{index_name}/_search"
    
    # KNN query for vector similarity search
    search_body = {
        "size": size,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": size
                }
            }
        }
    }
    
    response = requests.post(url, json=search_body, auth=awsauth)
    
    if response.status_code == 200:
        results = response.json()
        hits = results['hits']['hits']
        return hits
    else:
        logger.error("Failed semantic search. Status code: %d, Response: %s",
                     response.status_code, response.text)
        return []


def opensearch_hybrid_search(index_name, query_text, query_vector, size=10):
    """Perform hybrid search combining text and vector search."""
    
    url = f"{opensearch_endpoint}/{index_name}/_search"
    
    # Hybrid query combining text search and vector similarity
    search_body = {
        "size": size,
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": ["content", "title"],
                            "boost": 1.0
                        }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_vector,
                                "k": size,
                                "boost": 2.0
                            }
                        }
                    }
                ]
            }
        }
    }
    
    response = requests.post(url, json=search_body, auth=awsauth)
    
    if response.status_code == 200:
        results = response.json()
        hits = results['hits']['hits']
        return hits
    else:
        logger.error("Failed hybrid search. Status code: %d, Response: %s",
                     response.status_code, response.text)
        return []


def opensearch_filter_search(index_name, filters, size=10):
    """Perform filtered search based on metadata fields."""
    
    url = f"{opensearch_endpoint}/{index_name}/_search"
    
    # Build filter conditions
    filter_conditions = []
    for field, value in filters.items():
        filter_conditions.append({
            "term": {field: value}
        })
    
    search_body = {
        "size": size,
        "query": {
            "bool": {
                "filter": filter_conditions
            }
        }
    }
    
    response = requests.post(url, json=search_body, auth=awsauth)
    
    if response.status_code == 200:
        results = response.json()
        hits = results['hits']['hits']
        return hits
    else:
        logger.error("Failed filter search. Status code: %d, Response: %s",
                     response.status_code, response.text)
        return []