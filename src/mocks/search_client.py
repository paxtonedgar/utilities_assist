# search_client.py - Enhanced mock search client with BM25 ranking

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    logger.warning("rank_bm25 not available. Install with: pip install rank-bm25")
    BM25_AVAILABLE = False


class MockSearchClient:
    """Enhanced mock search client using BM25 ranking with real mock documents."""
    
    def __init__(self):
        self.documents = []
        self.bm25 = None
        self.corpus = []
        self._load_mock_documents()
        self._build_bm25_index()
    
    def _load_mock_documents(self):
        """Load documents from data/mock_docs directory."""
        # Find the mock_docs directory relative to this file
        current_dir = Path(__file__).parent
        mock_docs_dir = current_dir.parent.parent / "data" / "mock_docs"
        
        if not mock_docs_dir.exists():
            logger.warning(f"Mock docs directory not found: {mock_docs_dir}")
            logger.info("Using fallback static documents")
            self._load_fallback_documents()
            return
        
        # Load all doc-*.json files
        doc_files = list(mock_docs_dir.glob("doc-*.json"))
        if not doc_files:
            logger.warning("No doc-*.json files found in mock_docs directory")
            self._load_fallback_documents()
            return
        
        logger.info(f"Loading {len(doc_files)} documents from {mock_docs_dir}")
        
        for doc_file in sorted(doc_files):
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    self.documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load {doc_file}: {e}")
        
        logger.info(f"Loaded {len(self.documents)} mock documents for BM25 search")
    
    def _load_fallback_documents(self):
        """Load static fallback documents if mock_docs are not available."""
        self.documents = [
            {
                "_id": "fallback-001",
                "page_url": "https://mock.example.com/api/auth",
                "api_name": "Authentication Service - REST API",
                "utility_name": "Authentication Service",
                "sections": [
                    {
                        "heading": "API Authentication",
                        "content": "How to authenticate with the utilities API using OAuth tokens and JWT"
                    }
                ]
            },
            {
                "_id": "fallback-002", 
                "page_url": "https://mock.example.com/api/search",
                "api_name": "Search Engine - Search API",
                "utility_name": "Search Engine",
                "sections": [
                    {
                        "heading": "Search Functionality",
                        "content": "Search functionality for finding utilities and resources using BM25 and vector search"
                    }
                ]
            },
            {
                "_id": "fallback-003",
                "page_url": "https://mock.example.com/api/pipeline",
                "api_name": "Data Pipeline - Data API",
                "utility_name": "Data Pipeline",
                "sections": [
                    {
                        "heading": "Data Processing",
                        "content": "Data processing and pipeline management endpoints for ETL operations"
                    }
                ]
            }
        ]
        logger.info(f"Using {len(self.documents)} fallback documents")
    
    def _build_bm25_index(self):
        """Build BM25 index from document content."""
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available, using simple text matching")
            return
        
        # Create corpus from all document text
        self.corpus = []
        for doc in self.documents:
            # Combine all text from document
            text_parts = []
            
            # Add api_name and utility_name
            if 'api_name' in doc:
                text_parts.append(doc['api_name'])
            if 'utility_name' in doc:
                text_parts.append(doc['utility_name'])
            
            # Add all section content
            if 'sections' in doc:
                for section in doc['sections']:
                    if 'heading' in section:
                        text_parts.append(section['heading'])
                    if 'content' in section:
                        text_parts.append(section['content'])
            
            # Join and tokenize
            combined_text = ' '.join(text_parts)
            tokens = combined_text.lower().split()
            self.corpus.append(tokens)
        
        # Build BM25 index
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            logger.info(f"Built BM25 index with {len(self.corpus)} documents")
        else:
            logger.warning("No content found for BM25 indexing")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents using BM25 ranking."""
        logger.info(f"MockSearchClient: BM25 search for '{query}' (top_k={top_k})")
        
        if not self.documents:
            logger.warning("No documents available for search")
            return []
        
        if BM25_AVAILABLE and self.bm25:
            return self._bm25_search(query, top_k)
        else:
            return self._simple_search(query, top_k)
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform BM25 search using rank_bm25."""
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k document indices
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:top_k]
        
        results = []
        for idx in top_indices:
            if idx >= len(self.documents):
                continue
                
            doc = self.documents[idx]
            score = float(scores[idx])
            
            # Skip very low scores (likely irrelevant)
            if score < 0.1:
                continue
            
            # Convert to expected format
            result = {
                "doc_id": doc.get("_id", f"doc-{idx}"),
                "score": score,
                "text": self._extract_text_summary(doc),
                "title": self._extract_title(doc),
                "page_url": doc.get("page_url", ""),
                "api_name": doc.get("api_name", "Unknown"),
                "utility_name": doc.get("utility_name", "Unknown")
            }
            results.append(result)
        
        logger.info(f"BM25 search returned {len(results)} results")
        return results
    
    def _simple_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback simple text matching search."""
        query_terms = query.lower().split()
        results = []
        
        for i, doc in enumerate(self.documents):
            score = self._calculate_simple_score(query_terms, doc)
            
            if score > 0:
                result = {
                    "doc_id": doc.get("_id", f"doc-{i}"),
                    "score": score,
                    "text": self._extract_text_summary(doc),
                    "title": self._extract_title(doc),
                    "page_url": doc.get("page_url", ""),
                    "api_name": doc.get("api_name", "Unknown"),
                    "utility_name": doc.get("utility_name", "Unknown")
                }
                results.append(result)
        
        # Sort by score and take top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:top_k]
        
        logger.info(f"Simple search returned {len(results)} results")
        return results
    
    def _calculate_simple_score(self, query_terms: List[str], doc: Dict[str, Any]) -> float:
        """Calculate simple relevance score for fallback search."""
        score = 0.0
        
        # Check api_name
        api_name = doc.get("api_name", "").lower()
        for term in query_terms:
            if term in api_name:
                score += 2.0  # Higher weight for API name matches
        
        # Check utility_name  
        utility_name = doc.get("utility_name", "").lower()
        for term in query_terms:
            if term in utility_name:
                score += 1.5
        
        # Check sections
        if "sections" in doc:
            for section in doc["sections"]:
                heading = section.get("heading", "").lower()
                content = section.get("content", "").lower()
                
                for term in query_terms:
                    if term in heading:
                        score += 1.2
                    if term in content:
                        score += 1.0
        
        return score
    
    def _extract_text_summary(self, doc: Dict[str, Any]) -> str:
        """Extract a text summary from document."""
        if "sections" in doc and doc["sections"]:
            # Return content from first section
            first_section = doc["sections"][0]
            content = first_section.get("content", "")
            # Truncate to reasonable length
            return content[:200] + "..." if len(content) > 200 else content
        
        return doc.get("api_name", "No content available")
    
    def _extract_title(self, doc: Dict[str, Any]) -> str:
        """Extract a title from document."""
        if "sections" in doc and doc["sections"]:
            return doc["sections"][0].get("heading", doc.get("api_name", "Untitled"))
        
        return doc.get("api_name", "Untitled")

# For backward compatibility, also support the old simple interface
def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Module-level search function for backward compatibility."""
    global _search_client
    if '_search_client' not in globals():
        _search_client = MockSearchClient()
    return _search_client.search(query, top_k)