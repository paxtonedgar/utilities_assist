# vector_search.py - Mock vector search using FAISS index

import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available. Install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False

# Try to import Azure OpenAI for query embeddings
try:
    import sys
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))
    
    from client_manager import ClientSingleton
    from token_manager import TokenManager
    AZURE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Azure OpenAI not available for vector search: {e}")
    AZURE_AVAILABLE = False


class MockVectorSearch:
    """Mock vector search client using FAISS index."""
    
    def __init__(self):
        self.index = None
        self.metadata = []
        self.embedding_dim = 1536
        self._load_index_and_metadata()
    
    def _get_data_directory(self) -> Path:
        """Get the data directory path relative to this file."""
        current_dir = Path(__file__).parent
        return current_dir.parent.parent.parent / "data"
    
    def _load_index_and_metadata(self):
        """Load FAISS index and corresponding metadata."""
        data_dir = self._get_data_directory()
        index_path = data_dir / "mock_faiss.index"
        metadata_path = data_dir / "mock_faiss_metadata.json"
        
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available - vector search disabled")
            return
        
        # Load FAISS index
        if index_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors, dim={self.index.d}")
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {index_path}: {e}")
                return
        else:
            logger.warning(f"FAISS index not found at {index_path}")
            logger.info("Run 'make embed-local' or 'python scripts/embed_mock_docs.py' to create it")
            return
        
        # Load metadata
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata)} documents")
            except Exception as e:
                logger.error(f"Failed to load metadata from {metadata_path}: {e}")
                return
        else:
            logger.warning(f"Metadata file not found at {metadata_path}")
            return
        
        # Validate consistency
        if self.index and len(self.metadata) != self.index.ntotal:
            logger.warning(f"Metadata count ({len(self.metadata)}) != index count ({self.index.ntotal})")
    
    def _generate_query_embedding(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for search query using Azure OpenAI."""
        if not AZURE_AVAILABLE:
            logger.error("Azure OpenAI not available for query embedding")
            return None
        
        try:
            # Set environment for local Azure
            os.environ["USE_LOCAL_AZURE"] = "true"
            
            token_manager = TokenManager()
            client_singleton = ClientSingleton.get_instance(token_manager)
            embeddings_client = client_singleton.get_embeddings_client()
            
            # Generate embedding
            response = embeddings_client.embed_query(query)
            
            if isinstance(response, list) and len(response) > 0:
                embedding = np.array(response, dtype=np.float32)
                # Normalize for cosine similarity
                faiss.normalize_L2(embedding.reshape(1, -1))
                return embedding
            else:
                logger.error(f"Unexpected embedding response: {type(response)}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return None
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        logger.info(f"MockVectorSearch: searching for '{query}' (top_k={top_k})")
        
        if not self._is_ready():
            logger.warning("Vector search not ready - returning empty results")
            return []
        
        # Generate query embedding
        query_embedding = self._generate_query_embedding(query)
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return []
        
        # Search FAISS index
        try:
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1), 
                min(top_k, self.index.ntotal)
            )
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                if idx >= len(self.metadata):
                    logger.warning(f"Index {idx} out of range for metadata")
                    continue
                
                metadata = self.metadata[idx]
                
                # Convert to expected format matching OpenSearch results
                result = {
                    "doc_id": metadata.get("doc_id", f"doc-{idx}"),
                    "score": float(score),
                    "text": metadata.get("text", "No content available")[:200],
                    "title": self._extract_title_from_metadata(metadata),
                    "page_url": metadata.get("page_url", ""),
                    "api_name": metadata.get("api_name", "Unknown"),
                    "utility_name": metadata.get("utility_name", "Unknown"),
                    "sections_count": metadata.get("sections_count", 0)
                }
                results.append(result)
            
            logger.info(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []
    
    def _extract_title_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """Extract a title from metadata."""
        # Try to get title from text (first sentence)
        text = metadata.get("text", "")
        if text:
            # Take first sentence or first 50 chars
            first_sentence = text.split('.')[0]
            if len(first_sentence) <= 50:
                return first_sentence
            return text[:50] + "..."
        
        # Fallback to API name
        return metadata.get("api_name", "Untitled")
    
    def _is_ready(self) -> bool:
        """Check if vector search is ready to use."""
        return (
            FAISS_AVAILABLE and 
            self.index is not None and 
            len(self.metadata) > 0 and
            AZURE_AVAILABLE
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get status information about the vector search."""
        return {
            "faiss_available": FAISS_AVAILABLE,
            "azure_available": AZURE_AVAILABLE,
            "index_loaded": self.index is not None,
            "metadata_loaded": len(self.metadata) > 0,
            "total_vectors": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.index.d if self.index else self.embedding_dim,
            "ready": self._is_ready()
        }

# Global instance for module-level access
_vector_search_client = None

def get_vector_search_client() -> MockVectorSearch:
    """Get or create the global vector search client."""
    global _vector_search_client
    if _vector_search_client is None:
        _vector_search_client = MockVectorSearch()
    return _vector_search_client

def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Module-level search function for backward compatibility."""
    client = get_vector_search_client()
    return client.search(query, top_k)

def get_status() -> Dict[str, Any]:
    """Get vector search status."""
    client = get_vector_search_client()
    return client.get_status()