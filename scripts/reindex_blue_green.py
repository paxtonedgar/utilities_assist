#!/usr/bin/env python3
"""
Blue/Green reindex script for Confluence v2 mappings.

Migrates data from existing confluence indices to new confluence_v2-* index
with root-level vector embeddings and updated mappings.

Usage:
    python scripts/reindex_blue_green.py [--dry-run] [--source-index SOURCE] [--batch-size BATCH]
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Generator

import httpx
from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import NotFoundError, RequestError

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.infra.config import get_settings
from src.infra.clients import make_search_session

logger = logging.getLogger(__name__)


class BlueGreenReindexer:
    """Handles blue/green reindexing for Confluence v2 mappings."""
    
    def __init__(self, 
                 client: OpenSearch,
                 source_index: str = "khub-test-md",
                 batch_size: int = 100,
                 dry_run: bool = False):
        self.client = client
        self.source_index = source_index
        self.batch_size = batch_size
        self.dry_run = dry_run
        
        # Generate timestamp-based target index name
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.target_index = f"confluence_v2-{timestamp}"
        self.alias_name = "confluence_current"
        
        # Load mapping from file
        mapping_file = Path(__file__).parent.parent / "src" / "search" / "mappings" / "confluence_v2.json"
        with open(mapping_file, 'r') as f:
            self.mapping_config = json.load(f)
    
    def create_target_index(self) -> bool:
        """Create the new target index with v2 mappings."""
        try:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would create index: {self.target_index}")
                return True
            
            logger.info(f"Creating target index: {self.target_index}")
            
            response = self.client.indices.create(
                index=self.target_index,
                body=self.mapping_config
            )
            
            if response.get("acknowledged"):
                logger.info(f"✅ Target index created successfully: {self.target_index}")
                return True
            else:
                logger.error(f"❌ Failed to create target index: {response}")
                return False
                
        except RequestError as e:
            logger.error(f"❌ Error creating target index: {e}")
            return False
    
    def get_source_stats(self) -> Dict[str, Any]:
        """Get statistics about the source index."""
        try:
            if not self.client.indices.exists(index=self.source_index):
                logger.warning(f"Source index '{self.source_index}' does not exist")
                return {"exists": False}
            
            stats = self.client.indices.stats(index=self.source_index)
            count_result = self.client.count(index=self.source_index)
            
            return {
                "exists": True,
                "doc_count": count_result["count"],
                "size_bytes": stats["indices"][self.source_index]["total"]["store"]["size_in_bytes"],
                "shards": len(stats["indices"][self.source_index]["shards"])
            }
            
        except Exception as e:
            logger.error(f"Error getting source stats: {e}")
            return {"exists": False, "error": str(e)}
    
    def transform_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Transform document from old format to new v2 format."""
        source = doc.get("_source", {})
        
        # Handle nested embedding -> root level embedding
        embedding = None
        if "embedding" in source:
            # Already at root level
            embedding = source["embedding"]
        elif "nested_content" in source and isinstance(source["nested_content"], list):
            # Extract from nested structure (take first non-empty embedding)
            for nested in source["nested_content"]:
                if nested.get("embedding"):
                    embedding = nested["embedding"]
                    break
        
        # Build new document structure
        new_doc = {
            "title": source.get("title", ""),
            "section": source.get("section", "main"),
            "body": source.get("content", source.get("body", "")),
            "updated_at": source.get("updated_at", source.get("timestamp", datetime.now(timezone.utc).isoformat())),
            "page_id": source.get("page_id", source.get("id", "")),
            "section_anchor": source.get("section_anchor", ""),
            "canonical_id": source.get("canonical_id", source.get("doc_id", doc["_id"])),
            "acl_hash": source.get("acl_hash", "public"),
            "content_type": source.get("content_type", "confluence"),
            "source": source.get("source", "confluence"),
        }
        
        # Add embedding if available
        if embedding:
            new_doc["embedding"] = embedding
        
        # Add metadata if available
        metadata = {}
        for key in ["author", "space_key", "version", "labels"]:
            if key in source:
                metadata[key] = source[key]
        
        if metadata:
            new_doc["metadata"] = metadata
        
        return new_doc
    
    def reindex_documents(self) -> Tuple[int, int, List[str]]:
        """Reindex all documents from source to target."""
        success_count = 0
        error_count = 0
        errors = []
        
        def document_generator() -> Generator[Dict[str, Any], None, None]:
            """Generate transformed documents for bulk indexing."""
            try:
                # Scan all documents from source
                for doc in helpers.scan(
                    self.client,
                    index=self.source_index,
                    scroll="5m",
                    size=self.batch_size
                ):
                    try:
                        transformed = self.transform_document(doc)
                        
                        yield {
                            "_index": self.target_index,
                            "_id": doc["_id"],
                            "_source": transformed
                        }
                        
                    except Exception as e:
                        nonlocal error_count
                        error_count += 1
                        error_msg = f"Error transforming doc {doc.get('_id', 'unknown')}: {e}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                        
            except Exception as e:
                logger.error(f"Error scanning source index: {e}")
                errors.append(f"Scan error: {e}")
        
        if self.dry_run:
            logger.info(f"[DRY RUN] Would reindex documents from {self.source_index} to {self.target_index}")
            # Count documents for dry run
            try:
                count_result = self.client.count(index=self.source_index)
                return count_result["count"], 0, []
            except:
                return 0, 0, ["Could not count source documents in dry run"]
        
        logger.info(f"Starting document reindexing from {self.source_index} to {self.target_index}")
        
        try:
            # Use bulk helper for efficient indexing
            for success, info in helpers.parallel_bulk(
                self.client,
                document_generator(),
                chunk_size=self.batch_size,
                thread_count=4,
                max_chunk_bytes=50 * 1024 * 1024  # 50MB chunks
            ):
                if success:
                    success_count += 1
                    if success_count % 100 == 0:
                        logger.info(f"Indexed {success_count} documents...")
                else:
                    error_count += 1
                    error_msg = f"Bulk index error: {info}"
                    errors.append(error_msg)
                    logger.error(error_msg)
        
        except Exception as e:
            error_msg = f"Fatal error during bulk indexing: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        return success_count, error_count, errors
    
    def update_alias(self) -> bool:
        """Atomically update the confluence_current alias to point to new index."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would update alias {self.alias_name} to point to {self.target_index}")
            return True
        
        try:
            # Get current alias info
            current_indices = []
            try:
                alias_info = self.client.indices.get_alias(name=self.alias_name)
                current_indices = list(alias_info.keys())
            except NotFoundError:
                logger.info(f"Alias {self.alias_name} does not exist yet")
            
            # Build atomic alias update
            actions = []
            
            # Remove alias from old indices
            for index in current_indices:
                actions.append({
                    "remove": {
                        "index": index,
                        "alias": self.alias_name
                    }
                })
            
            # Add alias to new index
            actions.append({
                "add": {
                    "index": self.target_index,
                    "alias": self.alias_name
                }
            })
            
            logger.info(f"Updating alias {self.alias_name}: {current_indices} -> {self.target_index}")
            
            response = self.client.indices.update_aliases(
                body={"actions": actions}
            )
            
            if response.get("acknowledged"):
                logger.info(f"✅ Alias updated successfully")
                return True
            else:
                logger.error(f"❌ Failed to update alias: {response}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating alias: {e}")
            return False
    
    def cleanup_old_indices(self, keep_count: int = 3) -> bool:
        """Clean up old confluence_v2-* indices, keeping the most recent ones."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would clean up old confluence_v2-* indices, keeping {keep_count} most recent")
            return True
        
        try:
            # Get all confluence_v2-* indices
            all_indices = self.client.indices.get("confluence_v2-*")
            
            if len(all_indices) <= keep_count:
                logger.info(f"Only {len(all_indices)} indices found, no cleanup needed")
                return True
            
            # Sort by name (timestamp) and keep most recent
            sorted_indices = sorted(all_indices.keys(), reverse=True)
            indices_to_delete = sorted_indices[keep_count:]
            
            logger.info(f"Deleting old indices: {indices_to_delete}")
            
            for index in indices_to_delete:
                try:
                    self.client.indices.delete(index=index)
                    logger.info(f"Deleted old index: {index}")
                except Exception as e:
                    logger.warning(f"Failed to delete index {index}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False
    
    def run_reindex(self) -> Dict[str, Any]:
        """Execute the complete blue/green reindex process."""
        start_time = time.time()
        
        logger.info(f"🚀 Starting blue/green reindex")
        logger.info(f"Source: {self.source_index}")
        logger.info(f"Target: {self.target_index}")
        logger.info(f"Alias: {self.alias_name}")
        logger.info(f"Dry run: {self.dry_run}")
        
        # Get source statistics
        source_stats = self.get_source_stats()
        if not source_stats.get("exists"):
            return {
                "success": False,
                "error": f"Source index '{self.source_index}' does not exist",
                "duration": time.time() - start_time
            }
        
        logger.info(f"📊 Source index stats: {source_stats['doc_count']} docs, "
                   f"{source_stats['size_bytes'] / (1024*1024):.1f} MB")
        
        # Step 1: Create target index
        if not self.create_target_index():
            return {
                "success": False,
                "error": "Failed to create target index",
                "duration": time.time() - start_time
            }
        
        # Step 2: Reindex documents
        success_count, error_count, errors = self.reindex_documents()
        
        if error_count > 0:
            logger.warning(f"⚠️  Reindexing completed with {error_count} errors")
        
        # Step 3: Update alias
        if success_count > 0:
            if not self.update_alias():
                return {
                    "success": False,
                    "error": "Failed to update alias",
                    "documents_indexed": success_count,
                    "errors": error_count,
                    "duration": time.time() - start_time
                }
        
        # Step 4: Cleanup old indices
        self.cleanup_old_indices()
        
        duration = time.time() - start_time
        
        logger.info(f"✅ Blue/green reindex completed successfully")
        logger.info(f"📈 Documents processed: {success_count}")
        logger.info(f"⚠️  Errors: {error_count}")
        logger.info(f"⏱️  Duration: {duration:.2f} seconds")
        
        return {
            "success": True,
            "source_index": self.source_index,
            "target_index": self.target_index,
            "alias": self.alias_name,
            "documents_processed": success_count,
            "errors": error_count,
            "error_details": errors[:10],  # First 10 errors
            "duration": duration,
            "throughput": success_count / duration if duration > 0 else 0,
            "dry_run": self.dry_run
        }


def create_opensearch_client(settings) -> OpenSearch:
    """Create OpenSearch client from settings with proper JPMC authentication."""
    # Parse host URL properly
    host_url = settings.search.host
    
    # Handle full URLs like "http://localhost:9200"
    if host_url.startswith(("http://", "https://")):
        from urllib.parse import urlparse
        parsed = urlparse(host_url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 9200)
        use_ssl = parsed.scheme == "https"
    else:
        # Handle bare host or host:port
        if ":" in host_url:
            host, port_str = host_url.rsplit(":", 1)
            port = int(port_str)
        else:
            host = host_url
            port = 9200
        use_ssl = False
    
    # For JPMC environment, we need to use the proper authentication
    if settings.profile == "jpmc_azure":
        # Import here to avoid circular imports
        import os
        from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
        
        # Setup proxy for JPMC environment
        _setup_jpmc_proxy()
        
        # Get AWS authentication for OpenSearch
        aws_auth = _get_aws_auth()
        if aws_auth:
            logger.info("Using AWS4Auth for OpenSearch in JPMC environment")
            http_auth = aws_auth
        else:
            logger.warning("AWS4Auth not available, falling back to no auth")
            http_auth = None
    else:
        # Local environment - use basic auth if provided
        http_auth = (settings.search.username, settings.search.password) if settings.search.username else None
    
    return OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=http_auth,
        use_ssl=use_ssl,
        verify_certs=False,  # Often false for local dev
        timeout=int(settings.search.timeout_s),
        max_retries=3,
        retry_on_timeout=True
    )


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Blue/Green reindex for Confluence v2 mappings")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode - don't make changes")
    parser.add_argument("--source-index", default="khub-test-md", help="Source index name")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for reindexing")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load settings
        settings = get_settings()
        
        # Create OpenSearch client
        client = create_opensearch_client(settings)
        
        # Test connection
        logger.info("Testing OpenSearch connection...")
        cluster_info = client.info()
        logger.info(f"Connected to OpenSearch cluster: {cluster_info['version']['number']}")
        
        # Run reindex
        reindexer = BlueGreenReindexer(
            client=client,
            source_index=args.source_index,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
        
        result = reindexer.run_reindex()
        
        # Print summary
        print("\n" + "="*60)
        print("🔄 BLUE/GREEN REINDEX SUMMARY")
        print("="*60)
        print(f"Status: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
        print(f"Source Index: {result.get('source_index', 'N/A')}")
        print(f"Target Index: {result.get('target_index', 'N/A')}")
        print(f"Alias: {result.get('alias', 'N/A')}")
        print(f"Documents Processed: {result.get('documents_processed', 0):,}")
        print(f"Errors: {result.get('errors', 0):,}")
        print(f"Duration: {result.get('duration', 0):.2f} seconds")
        print(f"Throughput: {result.get('throughput', 0):.1f} docs/sec")
        print(f"Dry Run: {'Yes' if result.get('dry_run') else 'No'}")
        
        if result.get('error_details'):
            print(f"\nFirst few errors:")
            for i, error in enumerate(result['error_details'], 1):
                print(f"  {i}. {error}")
        
        print("="*60)
        
        # Exit with appropriate code
        sys.exit(0 if result['success'] else 1)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()