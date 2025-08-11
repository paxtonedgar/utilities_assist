#!/usr/bin/env python3
"""
Generate embeddings for mock documents using Azure OpenAI and store in FAISS index.
Reads data/mock_docs/*.json, generates embeddings, and saves FAISS index + metadata.
"""

import json
import logging
import os
import sys
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    print("❌ FAISS not available. Install with: pip install faiss-cpu")
    FAISS_AVAILABLE = False
    sys.exit(1)

from utils import load_config
from client_manager import ClientSingleton
from token_manager import TokenManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for mock documents and create FAISS index."""
    
    def __init__(self, config_file: str = "config.local.ini"):
        self.config_file = config_file
        self.embedding_dim = 1536  # Default for text-embedding-3-small
        self.documents = []
        self.embeddings = []
        self.metadata = []
        
        # Set environment for local config
        os.environ["UTILITIES_CONFIG"] = config_file
        os.environ["USE_LOCAL_AZURE"] = "true"
        
        self.config = load_config()
        self.token_manager = None
        self.client_singleton = None
        
    def initialize_clients(self):
        """Initialize Azure OpenAI clients."""
        try:
            self.token_manager = TokenManager()
            self.client_singleton = ClientSingleton.get_instance(self.token_manager)
            
            # Test embeddings client
            embeddings_client = self.client_singleton.get_embeddings_client()
            logger.info("✅ Azure OpenAI clients initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure OpenAI clients: {e}")
            logger.error("Please check your config.local.ini Azure OpenAI settings")
            return False
    
    def load_documents(self, docs_dir: Path) -> bool:
        """Load documents from mock_docs directory."""
        if not docs_dir.exists():
            logger.error(f"❌ Documents directory not found: {docs_dir}")
            return False
        
        doc_files = list(docs_dir.glob("doc-*.json"))
        if not doc_files:
            logger.error(f"❌ No doc-*.json files found in {docs_dir}")
            return False
        
        logger.info(f"📚 Loading {len(doc_files)} documents...")
        
        for doc_file in sorted(doc_files):
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    self.documents.append(doc)
                    
            except Exception as e:
                logger.error(f"❌ Failed to load {doc_file}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(self.documents)} documents")
        return len(self.documents) > 0
    
    def extract_text_for_embedding(self, doc: Dict[str, Any]) -> str:
        """Extract combined text from document for embedding."""
        text_parts = []
        
        # Add API name and utility name for context
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
        
        return ' '.join(text_parts)
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        try:
            embeddings_client = self.client_singleton.get_embeddings_client()
            
            # Generate embedding
            response = embeddings_client.embed_query(text)
            
            if isinstance(response, list) and len(response) > 0:
                # Convert to numpy array
                embedding = np.array(response, dtype=np.float32)
                return embedding
            else:
                logger.error(f"❌ Unexpected embedding response format: {type(response)}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to generate embedding: {e}")
            return None
    
    def generate_all_embeddings(self) -> bool:
        """Generate embeddings for all documents."""
        if not self.documents:
            logger.error("❌ No documents loaded")
            return False
        
        logger.info(f"🔄 Generating embeddings for {len(self.documents)} documents...")
        
        success_count = 0
        failed_count = 0
        
        for i, doc in enumerate(self.documents, 1):
            doc_id = doc.get('_id', f'doc-{i}')
            
            try:
                # Extract text
                text = self.extract_text_for_embedding(doc)
                if not text.strip():
                    logger.warning(f"⚠️  Empty text for document {doc_id}")
                    failed_count += 1
                    continue
                
                logger.info(f"📝 Embedding {i}/{len(self.documents)}: {doc_id}")
                
                # Generate embedding
                embedding = self.generate_embedding(text)
                
                if embedding is not None:
                    self.embeddings.append(embedding)
                    
                    # Store metadata
                    metadata = {
                        'doc_id': doc_id,
                        'text': text[:500],  # Store first 500 chars for reference
                        'api_name': doc.get('api_name', ''),
                        'utility_name': doc.get('utility_name', ''),
                        'page_url': doc.get('page_url', ''),
                        'sections_count': len(doc.get('sections', []))
                    }
                    self.metadata.append(metadata)
                    
                    logger.info(f"✅ {doc_id} → vector size: {len(embedding)} → success")
                    success_count += 1
                else:
                    logger.error(f"❌ {doc_id} → failed to generate embedding")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"❌ {doc_id} → error: {e}")
                failed_count += 1
                continue
        
        logger.info(f"📊 Embedding generation complete: {success_count} success, {failed_count} failed")
        return success_count > 0
    
    def create_faiss_index(self) -> Tuple[faiss.Index, bool]:
        """Create FAISS index from embeddings."""
        if not self.embeddings:
            logger.error("❌ No embeddings to index")
            return None, False
        
        logger.info(f"🔧 Creating FAISS index with {len(self.embeddings)} vectors...")
        
        try:
            # Convert embeddings to numpy array
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Create FAISS index (Inner Product for normalized vectors = cosine similarity)
            index = faiss.IndexFlatIP(self.embedding_dim)
            
            # Add vectors to index
            index.add(embeddings_array)
            
            logger.info(f"✅ FAISS index created: {index.ntotal} vectors, dimension {index.d}")
            return index, True
            
        except Exception as e:
            logger.error(f"❌ Failed to create FAISS index: {e}")
            return None, False
    
    def save_index_and_metadata(self, index: faiss.Index, output_dir: Path) -> bool:
        """Save FAISS index and metadata to files."""
        try:
            output_dir.mkdir(exist_ok=True)
            
            # Save FAISS index
            index_path = output_dir / "mock_faiss.index"
            faiss.write_index(index, str(index_path))
            logger.info(f"💾 FAISS index saved to: {index_path}")
            
            # Save metadata
            metadata_path = output_dir / "mock_faiss_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Metadata saved to: {metadata_path}")
            
            # Save embeddings as numpy array for debugging
            embeddings_path = output_dir / "mock_embeddings.npy"
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            np.save(embeddings_path, embeddings_array)
            logger.info(f"💾 Raw embeddings saved to: {embeddings_path}")
            
            # Save summary
            summary = {
                'total_documents': len(self.documents),
                'successful_embeddings': len(self.embeddings),
                'embedding_dimension': self.embedding_dim,
                'index_type': 'IndexFlatIP',
                'similarity_metric': 'cosine',
                'created_at': str(os.popen('date').read().strip())
            }
            
            summary_path = output_dir / "embedding_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"💾 Summary saved to: {summary_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save index and metadata: {e}")
            return False

def main():
    """Main function to generate embeddings and create FAISS index."""
    print("🔮 Mock Documents Embedding Generator")
    print("=" * 50)
    
    # Paths
    script_dir = Path(__file__).parent.parent
    docs_dir = script_dir / "data" / "mock_docs"
    output_dir = script_dir / "data"
    
    # Initialize generator
    generator = EmbeddingGenerator()
    
    # Initialize Azure OpenAI clients
    if not generator.initialize_clients():
        logger.error("❌ Failed to initialize clients")
        sys.exit(1)
    
    # Load documents
    if not generator.load_documents(docs_dir):
        logger.error("❌ Failed to load documents")
        sys.exit(1)
    
    # Generate embeddings
    if not generator.generate_all_embeddings():
        logger.error("❌ Failed to generate embeddings")
        sys.exit(1)
    
    # Create FAISS index
    index, success = generator.create_faiss_index()
    if not success:
        logger.error("❌ Failed to create FAISS index")
        sys.exit(1)
    
    # Save index and metadata
    if not generator.save_index_and_metadata(index, output_dir):
        logger.error("❌ Failed to save index and metadata")
        sys.exit(1)
    
    # Success summary
    print("\n" + "=" * 50)
    print("🎉 EMBEDDING GENERATION COMPLETE")
    print("=" * 50)
    print(f"Documents processed: {len(generator.documents)}")
    print(f"Embeddings generated: {len(generator.embeddings)}")
    print(f"Vector dimension: {generator.embedding_dim}")
    print(f"FAISS index: {output_dir / 'mock_faiss.index'}")
    print(f"Metadata: {output_dir / 'mock_faiss_metadata.json'}")
    print("")
    print("🔍 Test your embeddings:")
    print("   Use scripts/test_vector_search.py to verify the index")
    print("   Or use make run-local to test in the application")
    
    logger.info("✅ Embedding generation completed successfully!")

if __name__ == "__main__":
    main()