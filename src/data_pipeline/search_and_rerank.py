# search_and_rerank.py - Search and reranking module for hybrid search
import asyncio
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict

from client_manager import ClientSingleton
import requests
import logging
import numpy as np
from utils import load_config
# Configure logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
config = load_config()
index_name = config['aws_info']['index_name']
opensearch_endpoint = config['aws_info']['opensearch_endpoint']


def extract_chunks_from_response(response):
    chunk_map = {}
    for hit in response['hits']['hits']:
        doc_id = hit['_id']
        source = hit.get('_source', {})
        page_url = source.get('page_url', "UNKNOWN_URL")
        page_title = source.get('api_name', "UNKNOWN_TITLE")
        inner_hits = hit.get('inner_hits', {}).get('matched_sections', {}).get('hits', {}).get('hits', [])
        chunks = []
        for section in inner_hits:
            section_source = section.get('_source', {})
            heading = section_source.get('heading', '')
            content = section_source.get('content', '')
            if content:
                chunks.append({
                    "heading": heading,
                    "content": content
                })
        if chunks:
            chunk_map[doc_id] = {
                "page_url": page_url,
                "page_title": page_title,
                "chunks": chunks
            }
    return chunk_map


async def bm25_search(index, query, utility_name, api_name, filters, awsauth):
    """Perform BM25 search using OpenSearch."""
    logging.info("Starting BM25 search.")
    must_clauses = [
        {
            "nested": {
                "path": "sections",
                "query": {
                    "match": {
                        "sections.content": {
                            "query": query
                        }
                    }
                },
                "inner_hits": {
                    "name": "matched_sections",
                    "size": 5,
                    "highlight": {
                        "fields": {
                            "sections.content": {}
                        }
                    },
                    "sort": [{"_score": "desc"}]
                }
            }
        }
    ]
    
    if len(api_name) > 0:
        must_clauses.append({
            "bool": {
                "should": [
                    {
                        "match": {
                            "api_name": {
                                "query": name,
                                "fuzziness": "AUTO"
                            }
                        }
                    } for name in api_name
                ],
                "minimum_should_match": 1
            }
        })
    
    filter_clauses = []
    if len(utility_name) > 0:
        filter_clauses.append({
            "terms": {
                "utility_name.keyword": utility_name
            }
        })
    
    search_body = {
        "size": 3,
        "_source": ["page_url", "api_name", "utility_name"],
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses
            }
        }
    }
    
    # Send the query to OpenSearch
    response = requests.get(
        url=f"{opensearch_endpoint}/{index}/_search",
        auth=awsauth,
        json=search_body
    )
    
    if response.status_code == 200:
        results = response.json()
        # with open('bm25results.json', 'w') as f:
        #     json.dump(results, f)
        scores = [hit['_score'] for hit in results['hits']['hits']]
        document_ids = [hit['_id'] for hit in results['hits']['hits']]
        print("Docs and Scores", document_ids, scores)
        logging.info("BM25 search completed.")
        resp = extract_chunks_from_response(results)
        return document_ids, scores, resp
    else:
        logging.error(f"BM25 search failed. Status code: {response.status_code}. Response: {response.text}")
        return []


async def compute_query_embedding(query, token_manager):
    """Compute embedding for the query using AzureOpenAI Embeddings asynchronously."""
    logging.info("Computing query embedding.")
    try:
        client_singleton = ClientSingleton.get_instance(token_manager)
        embeddings_client = client_singleton.get_embeddings_client()
        response = embeddings_client.embed_query(query)  # Remove await if this is not asynchronous
        logging.info("Query embedding computed successfully.")
        return response
    except Exception as e:
        logging.error(f"Failed to compute query embedding: {e}")
        return None


async def semantic_search(index, query_embedding, utility_name, api_name, filters, awsauth):
    """Perform semantic search using OpenSearch."""
    logging.info("Starting semantic search.")
    print("Utility name and api_name in semantic search", utility_name, api_name)
    must_clauses = [
        {
            "nested": {
                "path": "sections",
                "query": {
                    "knn": {
                        "sections.embedding": {
                            "vector": query_embedding,
                            "k": 5
                        }
                    }
                },
                "inner_hits": {
                    "name": "matched_sections",
                    "size": 5,
                    "sort": [
                        {
                            "_score": "desc"
                        }
                    ]
                }
            }
        }
    ]
    
    if len(api_name) > 0:
        must_clauses.append({
            "bool": {
                "should": [
                    {
                        "match": {
                            "api_name": {
                                "query": name,
                                "fuzziness": "AUTO"
                            }
                        }
                    } for name in api_name
                ],
                "minimum_should_match": 1
            }
        })
    
    filter_clauses = []
    if len(utility_name) > 0:
        filter_clauses.append({
            "terms": {
                "utility_name.keyword": utility_name
            }
        })
    
    search_body = {
        "size": 3,
        "_source": [
            "page_url",
            "api_name",
            "utility_name"
        ],
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses
            }
        }
    }
    
    # Send the query to OpenSearch
    response = requests.get(
        url=f"{opensearch_endpoint}/{index}/_search",
        auth=awsauth,
        json=search_body
    )
    
    if response.status_code == 200:
        results = response.json()
        # print(results)
        scores = [hit['_score'] for hit in results['hits']['hits']]
        document_ids = [hit['_id'] for hit in results['hits']['hits']]
        print("SDocs and Scores", document_ids, scores)
        logging.info("Semantic search completed.")
        resp = extract_chunks_from_response(results)
        return document_ids, scores, resp
    else:
        logging.error(f"Semantic search failed. Status code: {response.status_code}. Response: {response.text}")
        return []


def normalize_scores(scores):
    logging.info("Normalizing scores.")
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        logging.warning("All scores are equal. Returning uniform normalized scores.")
        return np.ones_like(scores) / len(scores)
    normalized_scores = (scores - min_score) / (max_score - min_score)
    logging.info(f"Normalized Scores: {normalized_scores}")
    return normalized_scores


def get_doc_chunk_data(bm25_data, semantic_data):
    # print("bm25_data", bm25_data)
    final_chunk_map = defaultdict(lambda: {"page_url": "", "chunks": []})
    for source_data in [bm25_data, semantic_data]:
        for doc_id, chunk_data in source_data.items():
            final_chunk_map[doc_id]["page_url"] = chunk_data["page_url"]
            final_chunk_map[doc_id]["page_title"] = chunk_data["page_title"]
            existing_chunks_set = {
                (chunk.get('heading', '').strip(), chunk.get('content', '').strip()) for chunk in
                final_chunk_map[doc_id]['chunks']
            }
            for chunk in chunk_data["chunks"]:
                heading = chunk.get('heading', '').strip()
                content = chunk.get('content', '').strip()
                chunk_key = (heading, content)
                if chunk_key not in existing_chunks_set:
                    final_chunk_map[doc_id]['chunks'].append({
                        "heading": heading,
                        "content": content
                    })
                    existing_chunks_set.add(chunk_key)
    return final_chunk_map


def combine_search_results(bm25_results, semantic_results, is_sentence_based):
    
    bm25_doc_ids, bm25_scores, bm25_data = bm25_results
    semantic_doc_ids, semantic_scores, semantic_data = semantic_results
    combined_scores_dict = {}
    normalized_bm25_scores = normalize_scores(np.array(bm25_scores))
    
    if semantic_results is not None and len(semantic_results[1]) > 0:
        semantic_doc_ids, semantic_scores, semantic_data = semantic_results
        normalized_semantic_scores = normalize_scores(np.array(semantic_scores))
        
        for doc_id, score in zip(bm25_doc_ids, normalized_bm25_scores):
            if doc_id not in combined_scores_dict:
                combined_scores_dict[doc_id] = [0.0, 0.0]
            combined_scores_dict[doc_id][0] = score
        
        for doc_id, score in zip(semantic_doc_ids, normalized_semantic_scores):
            if doc_id not in combined_scores_dict:
                combined_scores_dict[doc_id] = [0.0, 0.0]
            combined_scores_dict[doc_id][1] = score
    else:
        for doc_id, score in zip(bm25_doc_ids, normalized_bm25_scores):
            combined_scores_dict[doc_id] = [score, 0.0]
    
    # Compute final combined scores
    combined_scores = {}
    for doc_id, (bm25_score, semantic_score) in combined_scores_dict.items():
        if is_sentence_based:
            combined_score = 0.3 * bm25_score + 0.7 * semantic_score
        else:
            combined_score = 0.7 * bm25_score + 0.3 * semantic_score
        combined_scores[doc_id] = combined_score
    
    # Use only bm25_data if semantic_data is unavailable
    semantic_data = semantic_data if 'semantic_data' in locals() else {}
    final_chunk_map = get_doc_chunk_data(bm25_data, semantic_data)
    
    # Sort and select top-k documents
    top_k_doc_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)[:5]
    logging.info(f"Top-k Document IDs: {top_k_doc_ids}")
    return top_k_doc_ids, final_chunk_map


async def adaptive_search_conf(index, user_input, utility_name, api_name, filters, token_manager, awsauth):
    
    logging.info("Starting adaptive search.")
    logging.info(f"tokenization of: {user_input}")
    tokenized_query = word_tokenize(user_input)
    logging.info("tokenization done")
    stopword_count = sum(1 for word in tokenized_query if word.lower() in stopwords.words('english'))
    logging.info("stop_words count done")
    is_sentence_based = stopword_count < len(tokenized_query) / 2
    logging.info("Query is {'sentence-based' if is_sentence_based else 'keyword-heavy'}.")
    query_embedding = await compute_query_embedding(user_input, token_manager)
    if query_embedding is None:
        logging.warning("Skipping semantic search due to missing embedding.")
        return []
    # Execute BM25 & Semantic search in parallel
    bm25_results, semantic_results = await asyncio.gather(
        bm25_search(index, user_input, utility_name, api_name, filters, awsauth),
        semantic_search(index, query_embedding, utility_name, api_name, filters, awsauth)
    )
    top_k_doc_ids, final_map = combine_search_results(bm25_results, semantic_results, is_sentence_based)
    return top_k_doc_ids, final_map