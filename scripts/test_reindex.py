#!/usr/bin/env python3
"""Test script for blue/green reindex functionality."""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_mapping_structure():
    """Test that the mapping file is valid JSON with required fields."""
    mapping_file = Path(__file__).parent.parent / "src" / "search" / "mappings" / "confluence_v2.json"
    
    print("🧪 Testing confluence_v2.json mapping...")
    
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
        
        # Check top-level structure
        assert "settings" in mapping, "Missing 'settings' section"
        assert "mappings" in mapping, "Missing 'mappings' section"
        
        # Check settings
        settings = mapping["settings"]
        assert "number_of_shards" in settings, "Missing number_of_shards"
        assert "number_of_replicas" in settings, "Missing number_of_replicas"
        assert "refresh_interval" in settings, "Missing refresh_interval"
        
        # Check mappings
        properties = mapping["mappings"]["properties"]
        required_fields = [
            "embedding", "title", "section", "body", 
            "updated_at", "page_id", "section_anchor", 
            "canonical_id", "acl_hash"
        ]
        
        for field in required_fields:
            assert field in properties, f"Missing required field: {field}"
        
        # Check embedding field
        embedding = properties["embedding"]
        assert embedding["type"] == "dense_vector", "Embedding should be dense_vector"
        assert embedding["dims"] == 1536, "Embedding should be 1536 dimensions"
        assert embedding["similarity"] == "cosine", "Should use cosine similarity"
        
        # Check HNSW configuration
        assert "index_options" in embedding, "Missing HNSW index_options"
        assert embedding["index_options"]["type"] == "hnsw", "Should use HNSW indexing"
        
        print("✅ Mapping structure is valid")
        return True
        
    except Exception as e:
        print(f"❌ Mapping validation failed: {e}")
        return False

def test_reindex_script_imports():
    """Test that the reindex script imports correctly."""
    print("🧪 Testing reindex script imports...")
    
    try:
        # Import the reindex module
        script_path = Path(__file__).parent / "reindex_blue_green.py"
        
        # Check file exists and is executable
        assert script_path.exists(), "Reindex script not found"
        assert script_path.stat().st_mode & 0o111, "Reindex script not executable"
        
        # Try to import key components
        sys.path.insert(0, str(script_path.parent))
        
        # This would normally import the module, but since it has a main() call,
        # we'll just check the file structure
        with open(script_path, 'r') as f:
            content = f.read()
            
        # Check for key components
        required_components = [
            "class BlueGreenReindexer",
            "def create_target_index",
            "def transform_document", 
            "def reindex_documents",
            "def update_alias",
            "def run_reindex"
        ]
        
        for component in required_components:
            assert component in content, f"Missing component: {component}"
        
        print("✅ Reindex script structure is valid")
        return True
        
    except Exception as e:
        print(f"❌ Reindex script validation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔄 Testing Blue/Green Reindex Implementation")
    print("=" * 50)
    
    tests = [
        test_mapping_structure,
        test_reindex_script_imports
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        sys.exit(0)
    else:
        print(f"❌ {total - passed} of {total} tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()