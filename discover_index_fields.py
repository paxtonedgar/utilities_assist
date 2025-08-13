#!/usr/bin/env python3
"""
Diagnostic script to discover the actual field names in the JPMC OpenSearch index.
This will help us identify the correct field names for BM25 and kNN searches.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from infra.config import get_settings
from infra.opensearch_client import create_search_client
import json

def main():
    """Discover field names in the JPMC OpenSearch index."""
    try:
        # Set environment variables to match your setup
        os.environ["UTILITIES_CONFIG"] = "src/config.ini"
        os.environ["CLOUD_PROFILE"] = "jpmc_azure"
        
        # Load settings
        settings = get_settings()
        
        # Create search client  
        search_client = create_search_client(settings.search)
        
        # Get the index name from settings
        index_name = settings.search.index_alias
        print(f"🔍 Inspecting OpenSearch index: {index_name}")
        print(f"📍 OpenSearch host: {settings.search.host}")
        print(f"🔧 Profile: {os.getenv('CLOUD_PROFILE', 'not set')}")
        
        # Get index mapping
        print("🔄 Attempting to retrieve index mapping...")
        mapping = search_client.get_index_mapping(index_name)
        
        if not mapping:
            print("❌ Failed to retrieve index mapping")
            print("💡 Make sure:")
            print("   - Your VPN is connected to JPMC network")
            print("   - config.ini has correct OpenSearch endpoint")
            print("   - AWS credentials are available")
            return
        
        print(f"✅ Retrieved mapping for index: {index_name}")
        
        # Extract field information
        analyze_mapping(mapping, index_name)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def analyze_mapping(mapping_data: dict, index_name: str):
    """Analyze the mapping data to identify field names and types."""
    
    print(f"\n📋 Index Mapping Analysis for '{index_name}':")
    print("=" * 60)
    
    # Navigate to the properties section
    index_mapping = mapping_data.get(index_name, {})
    mappings = index_mapping.get("mappings", {})
    properties = mappings.get("properties", {})
    
    if not properties:
        print("❌ No properties found in mapping")
        print(f"Raw mapping structure: {json.dumps(mapping_data, indent=2)}")
        return
    
    # Find text/content fields (for BM25)
    text_fields = []
    vector_fields = []
    other_fields = []
    
    def extract_fields(props, prefix=""):
        """Recursively extract fields from nested properties."""
        for field_name, field_config in props.items():
            full_name = f"{prefix}.{field_name}" if prefix else field_name
            field_type = field_config.get("type", "unknown")
            
            # Categorize fields
            if field_type in ["text", "keyword"]:
                text_fields.append((full_name, field_type))
            elif field_type == "knn_vector":
                vector_fields.append((full_name, field_type, field_config.get("dimension", "unknown")))
            else:
                other_fields.append((full_name, field_type))
            
            # Recurse into nested properties
            nested_props = field_config.get("properties", {})
            if nested_props:
                extract_fields(nested_props, full_name)
    
    extract_fields(properties)
    
    # Display results
    print(f"\n🔤 TEXT/SEARCHABLE FIELDS (for BM25):")
    if text_fields:
        for field_name, field_type in sorted(text_fields):
            print(f"  • {field_name} ({field_type})")
        
        # Suggest best fields for BM25
        content_candidates = [f for f, t in text_fields if any(term in f.lower() for term in ["content", "text", "body", "description", "title"])]
        if content_candidates:
            print(f"\n💡 SUGGESTED BM25 fields: {content_candidates}")
    else:
        print("  ❌ No text fields found")
    
    print(f"\n🔢 VECTOR FIELDS (for kNN):")
    if vector_fields:
        for field_name, field_type, dimension in sorted(vector_fields):
            print(f"  • {field_name} ({field_type}, dim: {dimension})")
        
        # Suggest best field for kNN
        embedding_candidates = [f for f, t, d in vector_fields if any(term in f.lower() for term in ["embedding", "vector"])]
        if embedding_candidates:
            print(f"\n💡 SUGGESTED kNN field: {embedding_candidates[0]}")
    else:
        print("  ❌ No vector fields found - kNN search not possible")
    
    print(f"\n📊 OTHER FIELDS:")
    for field_name, field_type in sorted(other_fields)[:10]:  # Show first 10
        print(f"  • {field_name} ({field_type})")
    if len(other_fields) > 10:
        print(f"  ... and {len(other_fields) - 10} more fields")
    
    print(f"\n📈 SUMMARY:")
    print(f"  • Total text/searchable fields: {len(text_fields)}")
    print(f"  • Total vector fields: {len(vector_fields)}")
    print(f"  • Total other fields: {len(other_fields)}")
    
    # Generate field name recommendations
    generate_recommendations(text_fields, vector_fields)

def generate_recommendations(text_fields, vector_fields):
    """Generate recommendations for updating the search client configuration."""
    
    print(f"\n🛠️  CONFIGURATION RECOMMENDATIONS:")
    print("=" * 60)
    
    if text_fields:
        # Find best content fields
        content_fields = [f for f, t in text_fields if any(term in f.lower() for term in ["content", "body", "text", "description"])]
        title_fields = [f for f, t in text_fields if "title" in f.lower()]
        
        if content_fields or title_fields:
            print("✅ Update BM25 search fields in _build_simple_bm25_query:")
            fields_list = []
            if title_fields:
                fields_list.append(f'"{title_fields[0]}"')
            if content_fields:
                fields_list.append(f'"{content_fields[0]}"')
            print(f'   "fields": [{", ".join(fields_list)}]')
        else:
            print(f"⚠️  No obvious content fields found. Available options:")
            for field_name, field_type in text_fields[:5]:
                print(f"   - {field_name}")
    
    if vector_fields:
        print(f"\n✅ Update kNN search field in _build_simple_knn_query:")
        vector_field = vector_fields[0][0]  # Use first vector field
        print(f'   "{vector_field}": {{...}}')
    else:
        print(f"\n❌ kNN search not possible - no vector fields in index")
        print("   Consider adding vector embeddings to your index first")

if __name__ == "__main__":
    main()