# test_mock_flow.py - End-to-end probe test using mock components

import pytest
import os
import sys
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.mark.asyncio
async def test_mock_search_flow():
    """Test the complete search flow using mock components."""
    # Set environment for mock mode
    os.environ["USE_MOCK_SEARCH"] = "true"
    os.environ["UTILITIES_CONFIG"] = "config.local.ini"
    
    try:
        from search_and_rerank import adaptive_search_conf
        
        # Test mock search
        test_query = "How do I authenticate with the API?"
        
        # Mock parameters (these won't be used in mock mode)
        index = "mock-index"
        utility_name = []
        api_name = []  
        filters = {}
        token_manager = None  # Mock doesn't need real token manager
        awsauth = None  # Mock doesn't need AWS auth
        
        # Run the search
        doc_ids, final_map = await adaptive_search_conf(
            index, test_query, utility_name, api_name, filters, token_manager, awsauth
        )
        
        # Verify results
        assert isinstance(doc_ids, list)
        assert isinstance(final_map, dict)
        assert len(doc_ids) > 0
        print(f"Mock search returned {len(doc_ids)} documents: {doc_ids}")
        
        # Verify result structure
        for doc_id in doc_ids:
            assert doc_id in final_map
            doc_info = final_map[doc_id]
            assert "page_url" in doc_info
            assert "page_title" in doc_info  
            assert "chunks" in doc_info
            assert isinstance(doc_info["chunks"], list)
            
            # Verify chunk structure
            for chunk in doc_info["chunks"]:
                assert "heading" in chunk
                assert "content" in chunk
        
        print("Mock search flow verification completed successfully")
        
    finally:
        # Clean up environment
        os.environ.pop("USE_MOCK_SEARCH", None)
        os.environ.pop("UTILITIES_CONFIG", None)


def test_config_switching():
    """Test configuration file switching based on environment."""
    from utils import load_config
    
    # Test default config
    original_env = os.environ.get("UTILITIES_CONFIG")
    
    try:
        # Test with default config
        if "UTILITIES_CONFIG" in os.environ:
            del os.environ["UTILITIES_CONFIG"]
            
        config = load_config()
        assert config is not None
        
        # Test with local config
        os.environ["UTILITIES_CONFIG"] = "config.local.ini"
        local_config = load_config()
        assert local_config is not None
        
        print("Config switching verification completed")
        
    finally:
        # Restore original environment
        if original_env is not None:
            os.environ["UTILITIES_CONFIG"] = original_env
        elif "UTILITIES_CONFIG" in os.environ:
            del os.environ["UTILITIES_CONFIG"]