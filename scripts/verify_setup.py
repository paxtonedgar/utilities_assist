#!/usr/bin/env python3
"""
Verify that the local development setup is working correctly.
Tests all components: mock documents, BM25 search, config loading, etc.
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_mock_documents():
    """Test that mock documents are available."""
    print("📚 Testing mock documents...")
    
    docs_dir = Path(__file__).parent.parent / "data" / "mock_docs"
    if not docs_dir.exists():
        print(f"❌ Mock docs directory not found: {docs_dir}")
        return False
    
    doc_files = list(docs_dir.glob("doc-*.json"))
    if len(doc_files) < 50:
        print(f"❌ Expected 50+ documents, found {len(doc_files)}")
        return False
    
    print(f"✅ Found {len(doc_files)} mock documents")
    return True

def test_config_switching():
    """Test configuration file switching."""
    print("⚙️  Testing configuration switching...")
    
    try:
        from utils import load_config
        
        # Test without env var
        if 'UTILITIES_CONFIG' in os.environ:
            del os.environ['UTILITIES_CONFIG']
        config1 = load_config()
        
        # Test with env var
        os.environ['UTILITIES_CONFIG'] = 'config.local.ini'
        config2 = load_config()
        
        print("✅ Configuration switching works")
        return True
        
    except Exception as e:
        print(f"❌ Configuration switching failed: {e}")
        return False

def test_bm25_search():
    """Test BM25 search functionality."""
    print("🔍 Testing BM25 search...")
    
    try:
        from mocks.search_client import MockSearchClient
        
        client = MockSearchClient()
        if len(client.documents) == 0:
            print("❌ No documents loaded in search client")
            return False
        
        results = client.search("API authentication", top_k=3)
        if len(results) == 0:
            print("❌ Search returned no results")
            return False
        
        print(f"✅ BM25 search working: {len(client.documents)} docs, found {len(results)} results")
        return True
        
    except Exception as e:
        print(f"❌ BM25 search failed: {e}")
        return False

def test_dependencies():
    """Test that required dependencies are installed."""
    print("📦 Testing dependencies...")
    
    dependencies = [
        ('rank_bm25', 'rank_bm25'),
        ('opensearch-py', 'opensearchpy'),
        ('faiss-cpu', 'faiss'),
        ('numpy', 'numpy'),
        ('json', 'json')
    ]
    
    missing = []
    for name, module in dependencies:
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Run: pip install " + " ".join(missing))
        return False
    
    print("✅ All dependencies installed")
    return True

def test_makefile_targets():
    """Test that Makefile targets exist."""
    print("🎯 Testing Makefile targets...")
    
    makefile_path = Path(__file__).parent.parent / "Makefile"
    if not makefile_path.exists():
        print("❌ Makefile not found")
        return False
    
    with open(makefile_path, 'r') as f:
        content = f.read()
    
    required_targets = [
        'run-local', 'test-local', 'index-local', 'embed-local',
        'start-opensearch', 'stop-opensearch'
    ]
    
    missing = []
    for target in required_targets:
        if f"{target}:" not in content:
            missing.append(target)
    
    if missing:
        print(f"❌ Missing Makefile targets: {', '.join(missing)}")
        return False
    
    print("✅ All Makefile targets present")
    return True

def test_scripts_executable():
    """Test that scripts are executable."""
    print("🔧 Testing script permissions...")
    
    script_files = [
        'start_opensearch_local.sh',
        'stop_opensearch_local.sh',
        'index_mock_docs.py',
        'embed_mock_docs.py',
        'test_vector_search.py'
    ]
    
    scripts_dir = Path(__file__).parent
    non_executable = []
    
    for script in script_files:
        script_path = scripts_dir / script
        if not script_path.exists():
            non_executable.append(f"{script} (not found)")
        elif not os.access(script_path, os.X_OK):
            non_executable.append(f"{script} (not executable)")
    
    if non_executable:
        print(f"❌ Script issues: {', '.join(non_executable)}")
        return False
    
    print("✅ All scripts are executable")
    return True

def main():
    """Run all verification tests."""
    print("🔬 Verifying Local Development Setup")
    print("=" * 50)
    
    tests = [
        test_dependencies,
        test_mock_documents,
        test_config_switching,
        test_bm25_search,
        test_makefile_targets,
        test_scripts_executable
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print("")
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
            print("")
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Local development setup is ready.")
        print("")
        print("🚀 Next steps:")
        print("1. Configure Azure OpenAI in config.local.ini (for embeddings)")
        print("2. Run: make setup-local (starts OpenSearch + indexes documents)")  
        print("3. Run: make embed-local (generates embeddings - optional)")
        print("4. Run: make run-local (starts the application)")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())