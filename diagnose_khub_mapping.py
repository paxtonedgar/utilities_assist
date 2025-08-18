#!/usr/bin/env python3
"""
K-hub Cluster Index Mapping Diagnostic Tool

This script diagnoses the vector field mapping issue by:
1. Checking the actual index mapping from K-hub cluster
2. Identifying field types and structure
3. Comparing with application expectations
4. Providing corrective recommendations
"""

import json
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.infra.opensearch_client import create_search_client
from src.infra.settings import get_settings
from src.infra.search_config import OpenSearchConfig

def diagnose_index_mapping():
    """Diagnose the K-hub cluster index mapping and field types."""
    
    print("🔍 K-hub Cluster Index Mapping Diagnostic")
    print("=" * 50)
    
    try:
        # Get configuration
        settings = get_settings()
        print(f"✅ Configuration loaded: {settings.opensearch_host}")
        print(f"✅ Index alias: {settings.search_index_alias}")
        print(f"✅ Profile: {settings.cloud_profile}")
        
        # Get search client
        search_client = create_search_client(settings)
        print(f"✅ Search client initialized")
        
        # Get index mapping
        index_name = settings.search_index_alias
        print(f"\n📋 Fetching mapping for index: {index_name}")
        
        mapping = search_client.get_index_mapping(index_name)
        
        if not mapping:
            print("❌ Failed to retrieve index mapping")
            return
            
        print("✅ Index mapping retrieved successfully")
        
        # Analyze mapping structure
        print(f"\n🔍 Analyzing Index Mapping Structure")
        print("-" * 40)
        
        # Check if mapping has expected structure
        if index_name in mapping:
            properties = mapping[index_name].get("mappings", {}).get("properties", {})
        else:
            # Sometimes the response structure is different
            properties = mapping.get("mappings", {}).get("properties", {})
            if not properties:
                # Try alternative structure
                for key, value in mapping.items():
                    if "mappings" in value:
                        properties = value["mappings"]["properties"]
                        break
        
        if not properties:
            print("❌ Could not find properties in mapping")
            print(f"Raw mapping structure: {json.dumps(mapping, indent=2)}")
            return
            
        print(f"✅ Found {len(properties)} fields in mapping")
        
        # Check vector field specifically
        vector_field = OpenSearchConfig.get_vector_field(index_name)
        print(f"\n🎯 Checking Vector Field: '{vector_field}'")
        print("-" * 40)
        
        if vector_field in properties:
            field_config = properties[vector_field]
            print(f"✅ Vector field '{vector_field}' found")
            print(f"   Type: {field_config.get('type', 'UNKNOWN')}")
            print(f"   Dimensions: {field_config.get('dimension', field_config.get('dims', 'UNKNOWN'))}")
            
            # Check field type compatibility
            field_type = field_config.get('type')
            if field_type == 'dense_vector':
                print("⚠️  Field type is 'dense_vector' (Elasticsearch style)")
                print("   Application expects 'knn_vector' (OpenSearch style)")
                print("   🔧 SOLUTION: Update query structure for dense_vector")
            elif field_type == 'knn_vector':
                print("✅ Field type is 'knn_vector' (OpenSearch style)")
                print("   Application configuration is correct")
            else:
                print(f"❌ Unexpected field type: {field_type}")
                
        else:
            print(f"❌ Vector field '{vector_field}' not found in mapping")
            print("Available fields:")
            for field_name, field_config in properties.items():
                field_type = field_config.get('type', 'unknown')
                if 'vector' in field_type or 'embedding' in field_name.lower():
                    print(f"   - {field_name}: {field_type}")
        
        # Check content fields
        print(f"\n📝 Checking Content Fields")
        print("-" * 40)
        
        content_fields = OpenSearchConfig.get_content_fields(index_name)
        for field in content_fields:
            if field in properties:
                field_type = properties[field].get('type', 'unknown')
                print(f"✅ {field}: {field_type}")
            else:
                print(f"⚠️  {field}: NOT FOUND")
        
        # Check metadata fields
        print(f"\n🏷️  Checking Metadata Fields")
        print("-" * 40)
        
        metadata_fields = OpenSearchConfig.get_metadata_fields(index_name)
        found_metadata = 0
        for field in metadata_fields[:10]:  # Show first 10
            if field in properties:
                field_type = properties[field].get('type', 'unknown')
                print(f"✅ {field}: {field_type}")
                found_metadata += 1
            else:
                print(f"⚠️  {field}: NOT FOUND")
        
        print(f"\nMetadata fields found: {found_metadata}/{len(metadata_fields)}")
        
        # Generate recommendations
        print(f"\n💡 Recommendations")
        print("-" * 40)
        
        if vector_field in properties:
            field_type = properties[vector_field].get('type')
            if field_type == 'dense_vector':
                print("1. Update kNN query structure to use 'dense_vector' syntax")
                print("2. Change query from 'knn' to 'script_score' with cosine similarity")
                print("3. Test with corrected query structure")
            elif field_type == 'knn_vector':
                print("1. Current query structure should work")
                print("2. Check for other query syntax issues")
                print("3. Verify authentication and permissions")
        else:
            print("1. Identify correct vector field name in the index")
            print("2. Update OpenSearchConfig with correct field name")
            print("3. Re-test vector search")
            
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_index_mapping()