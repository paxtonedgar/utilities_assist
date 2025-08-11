#!/usr/bin/env python3
"""
Index mock documents into local OpenSearch instance.
Reads data/mock_docs/*.json and indexes them into mock_confluence index.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.exceptions import RequestError, ConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_opensearch_client() -> OpenSearch:
    """Create OpenSearch client for local development."""
    return OpenSearch(
        hosts=[{'host': 'localhost', 'port': 9200}],
        http_compress=True,
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
        connection_class=RequestsHttpConnection,
        timeout=30
    )

def create_index_mapping() -> Dict[str, Any]:
    """Create index mapping optimized for BM25 search with vector support."""
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "content_analyzer": {
                        "type": "standard",
                        "stopwords": "_english_"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "page_url": {
                    "type": "keyword"
                },
                "api_name": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    },
                    "analyzer": "content_analyzer"
                },
                "utility_name": {
                    "type": "text", 
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "sections": {
                    "type": "nested",
                    "properties": {
                        "heading": {
                            "type": "text",
                            "analyzer": "content_analyzer"
                        },
                        "content": {
                            "type": "text",
                            "analyzer": "content_analyzer"
                        },
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 1536,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "created_date": {"type": "date"},
                "last_updated": {"type": "date"},
                "doc_type": {"type": "keyword"},
                "tags": {"type": "keyword"},
                "version": {"type": "keyword"}
            }
        }
    }

def ensure_index_exists(client: OpenSearch, index_name: str) -> bool:
    """Ensure the index exists with proper mapping."""
    try:
        if client.indices.exists(index=index_name):
            logger.info(f"Index '{index_name}' already exists")
            return True
        
        logger.info(f"Creating index '{index_name}' with BM25-optimized mapping...")
        mapping = create_index_mapping()
        
        response = client.indices.create(
            index=index_name,
            body=mapping
        )
        
        logger.info(f"Index '{index_name}' created successfully")
        return True
        
    except RequestError as e:
        if e.error == 'resource_already_exists_exception':
            logger.info(f"Index '{index_name}' already exists (400 error ignored)")
            return True
        else:
            logger.error(f"Failed to create index: {e}")
            return False
    except Exception as e:
        logger.error(f"Unexpected error creating index: {e}")
        return False

def load_mock_documents(docs_dir: Path) -> List[Dict[str, Any]]:
    """Load all mock documents from JSON files."""
    documents = []
    doc_files = list(docs_dir.glob("doc-*.json"))
    
    if not doc_files:
        logger.error(f"No doc-*.json files found in {docs_dir}")
        return documents
    
    logger.info(f"Loading {len(doc_files)} documents from {docs_dir}")
    
    for doc_file in sorted(doc_files):
        try:
            with open(doc_file, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Failed to load {doc_file}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(documents)} documents")
    return documents

def index_document(client: OpenSearch, index_name: str, doc: Dict[str, Any]) -> bool:
    """Index a single document."""
    doc_id = doc.get('_id')
    if not doc_id:
        logger.error("Document missing _id field")
        return False
    
    try:
        response = client.index(
            index=index_name,
            id=doc_id,
            body=doc,
            refresh=False  # We'll refresh once at the end
        )
        
        if response.get('result') in ['created', 'updated']:
            logger.info(f"✅ Indexed document: {doc_id}")
            return True
        else:
            logger.warning(f"⚠️  Unexpected result for {doc_id}: {response.get('result')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to index document {doc_id}: {e}")
        return False

def index_all_documents(client: OpenSearch, index_name: str, documents: List[Dict[str, Any]]) -> Dict[str, int]:
    """Index all documents and return statistics."""
    stats = {
        'total': len(documents),
        'success': 0,
        'failed': 0
    }
    
    logger.info(f"Starting to index {stats['total']} documents...")
    
    for i, doc in enumerate(documents, 1):
        logger.info(f"Indexing document {i}/{stats['total']}: {doc.get('_id', 'unknown')}")
        
        if index_document(client, index_name, doc):
            stats['success'] += 1
        else:
            stats['failed'] += 1
    
    # Refresh index to make documents searchable
    logger.info("Refreshing index to make documents searchable...")
    try:
        client.indices.refresh(index=index_name)
        logger.info("Index refreshed successfully")
    except Exception as e:
        logger.error(f"Failed to refresh index: {e}")
    
    return stats

def verify_indexing(client: OpenSearch, index_name: str, expected_count: int) -> bool:
    """Verify that documents were indexed correctly."""
    try:
        # Get document count
        count_response = client.count(index=index_name)
        actual_count = count_response.get('count', 0)
        
        logger.info(f"Index contains {actual_count} documents (expected {expected_count})")
        
        if actual_count != expected_count:
            logger.warning(f"Document count mismatch: expected {expected_count}, got {actual_count}")
            return False
        
        # Test a simple search
        search_response = client.search(
            index=index_name,
            body={
                "query": {"match_all": {}},
                "size": 1
            }
        )
        
        hits = search_response.get('hits', {}).get('hits', [])
        if hits:
            sample_doc = hits[0]['_source']
            logger.info(f"Sample document: {sample_doc.get('_id', 'unknown')} - {sample_doc.get('api_name', 'unknown')}")
            return True
        else:
            logger.error("No documents found in search results")
            return False
            
    except Exception as e:
        logger.error(f"Failed to verify indexing: {e}")
        return False

def main():
    """Main function to index mock documents."""
    print("🔍 OpenSearch Mock Document Indexer")
    print("=" * 50)
    
    # Configuration
    index_name = "mock_confluence"
    docs_dir = Path(__file__).parent.parent / "data" / "mock_docs"
    
    # Check if documents directory exists
    if not docs_dir.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        logger.error("Please run generate_mock_docs.py first")
        sys.exit(1)
    
    # Create OpenSearch client
    try:
        client = get_opensearch_client()
        
        # Test connection
        cluster_health = client.cluster.health()
        logger.info(f"Connected to OpenSearch cluster: {cluster_health.get('cluster_name', 'unknown')}")
        
    except ConnectionError as e:
        logger.error("Failed to connect to OpenSearch. Is the container running?")
        logger.error("Start it with: ./scripts/start_opensearch_local.sh")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to create OpenSearch client: {e}")
        sys.exit(1)
    
    # Ensure index exists
    if not ensure_index_exists(client, index_name):
        logger.error("Failed to create or verify index")
        sys.exit(1)
    
    # Load documents
    documents = load_mock_documents(docs_dir)
    if not documents:
        logger.error("No documents to index")
        sys.exit(1)
    
    # Index documents
    stats = index_all_documents(client, index_name, documents)
    
    # Verify indexing
    verification_passed = verify_indexing(client, index_name, stats['success'])
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 INDEXING SUMMARY")
    print("=" * 50)
    print(f"Total documents: {stats['total']}")
    print(f"Successfully indexed: {stats['success']}")
    print(f"Failed to index: {stats['failed']}")
    print(f"Verification: {'✅ PASSED' if verification_passed else '❌ FAILED'}")
    print("")
    print("🔍 Test your index:")
    print(f"   curl http://localhost:9200/{index_name}/_count")
    print(f"   curl http://localhost:9200/{index_name}/_search?size=1&pretty")
    
    if stats['failed'] > 0 or not verification_passed:
        sys.exit(1)
    
    print("\n🎉 All documents indexed successfully!")

if __name__ == "__main__":
    main()