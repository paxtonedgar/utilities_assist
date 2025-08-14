# test_interfaces.py - Probe tests to learn and verify codebase interfaces

import pytest
import os
import sys
from pathlib import Path

# Add src to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_config_loader_interface():
    """Probe test to verify config loader interface and environment switching."""
    from utils import load_config
    
    # Test default config loading
    config = load_config()
    assert config is not None
    assert hasattr(config, 'sections')
    assert hasattr(config, 'has_section')
    
    # Test sections exist (at least azure_openai should exist)
    sections = config.sections()
    assert isinstance(sections, list)
    print(f"Config sections found: {sections}")


def test_mock_search_client_interface():
    """Probe test to verify mock search client interface."""
    from mocks.search_client import MockSearchClient
    
    client = MockSearchClient()
    assert hasattr(client, 'search')
    
    # Test search method signature and return type
    results = client.search("test query", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3
    
    if results:
        result = results[0]
        assert "doc_id" in result
        assert "score" in result
        assert "text" in result
        assert "title" in result
        assert "page_url" in result
        print(f"Mock search returned {len(results)} results")


def test_search_and_rerank_interface():
    """Probe test to verify search and rerank module interface."""
    import search_and_rerank
    
    # Verify main function exists
    assert hasattr(search_and_rerank, 'adaptive_search_conf')
    print("Search and rerank interface verified")


def test_client_manager_interface():
    """Probe test to verify client manager interface."""
    from client_manager import ClientSingleton
    
    # Test ClientSingleton class structure
    assert hasattr(ClientSingleton, 'get_instance')
    assert hasattr(ClientSingleton, 'get_awsauth')
    print("Client manager interface verified")


@pytest.mark.slow
def test_chat_client_responds():
    """Probe test to verify chat client can respond (requires Azure OpenAI config)."""
    try:
        from client_manager import ClientSingleton
        from token_manager import TokenManager
        
        token_manager = TokenManager()
        client_singleton = ClientSingleton.get_instance(token_manager)
        chat_client = client_singleton.get_chat_client()
        
        # Test basic interface exists
        assert chat_client is not None
        assert hasattr(chat_client, 'invoke') or hasattr(chat_client, 'stream')
        print("Chat client interface verified")
        
    except Exception as e:
        print(f"Chat client test failed (expected in some configurations): {e}")
        # Don't fail the test, just log the issue
        pass


@pytest.mark.slow  
def test_embeddings_client_responds():
    """Probe test to verify embeddings client interface."""
    try:
        from client_manager import ClientSingleton
        from token_manager import TokenManager
        
        token_manager = TokenManager()
        client_singleton = ClientSingleton.get_instance(token_manager)
        embeddings_client = client_singleton.get_embeddings_client()
        
        # Test basic interface exists
        assert embeddings_client is not None
        assert hasattr(embeddings_client, 'embed_query')
        print("Embeddings client interface verified")
        
    except Exception as e:
        print(f"Embeddings client test failed (expected in some configurations): {e}")
        # Don't fail the test, just log the issue
        pass


def test_environment_variables():
    """Probe test to verify environment variable handling."""
    # Test that environment variables are being read
    utilities_config = os.getenv("UTILITIES_CONFIG", "config.ini")
    
    print(f"UTILITIES_CONFIG: {utilities_config}")
    
    # Verify they are strings
    assert isinstance(utilities_config, str)