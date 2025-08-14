#!/usr/bin/env python3
"""
Test script for centralized settings integration.

Verifies that:
1. Settings load correctly from config.ini
2. Profile-aware configuration works
3. Index aliases are properly configured
4. Authentication settings are preserved
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, 'src')

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_settings_loading():
    """Test basic settings loading."""
    print("\n🧪 Testing Settings Loading")
    print("=" * 50)
    
    try:
        from src.infra.settings import get_settings, refresh_settings
        
        # Test loading settings
        settings = get_settings()
        print(f"✅ Settings loaded successfully")
        print(f"   Cloud Profile: {settings.cloud_profile}")
        print(f"   Search Index Alias: {settings.search_index_alias}")
        print(f"   OpenSearch Host: {settings.opensearch_host}")
        print(f"   Requires AWS Auth: {settings.requires_aws_auth}")
        print(f"   Requires Azure Auth: {settings.requires_azure_auth}")
        
        # Test configuration sections
        if settings.azure_openai:
            print(f"   Azure OpenAI: ✅ Configured (endpoint: {settings.azure_openai.azure_openai_endpoint[:50]}...)")
        else:
            print(f"   Azure OpenAI: ❌ Not configured")
            
        if settings.aws_info:
            print(f"   AWS Info: ✅ Configured (region: {settings.aws_info.aws_region})")
        else:
            print(f"   AWS Info: ❌ Not configured")
            
        if settings.opensearch:
            print(f"   OpenSearch: ✅ Configured (endpoint: {settings.opensearch.endpoint})")
        else:
            print(f"   OpenSearch: ❌ Not configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Settings loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_opensearch_client():
    """Test OpenSearch client with centralized settings."""
    print("\n🔍 Testing OpenSearch Client Integration")
    print("=" * 50)
    
    try:
        from src.infra.opensearch_client import create_search_client
        from src.infra.settings import get_settings
        
        settings = get_settings()
        client = create_search_client(settings)
        
        print(f"✅ OpenSearch client created successfully")
        print(f"   Base URL: {client.base_url}")
        print(f"   Settings Index Alias: {settings.search_index_alias}")
        
        # Test health check if available
        try:
            health = client.health_check()
            print(f"   Health Status: {health.get('status', 'unknown')}")
        except Exception as he:
            print(f"   Health Check: ⚠️  {he}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenSearch client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_azure_auth_integration():
    """Test Azure authentication with centralized settings."""
    print("\n🔐 Testing Azure Auth Integration")
    print("=" * 50)
    
    try:
        from src.infra.settings import get_settings
        
        settings = get_settings()
        
        if settings.requires_azure_auth:
            print(f"✅ Azure authentication required for profile: {settings.cloud_profile}")
            
            if settings.azure_openai:
                print(f"   Tenant ID: {'***' if settings.azure_openai.azure_tenant_id else 'Not set'}")
                print(f"   Client ID: {'***' if settings.azure_openai.azure_client_id else 'Not set'}")
                print(f"   Endpoint: {settings.azure_openai.azure_openai_endpoint[:50]}..." if settings.azure_openai.azure_openai_endpoint else "Not set")
            else:
                print(f"   ❌ Azure OpenAI config missing")
        else:
            print(f"✅ Azure authentication not required for profile: {settings.cloud_profile}")
        
        return True
        
    except Exception as e:
        print(f"❌ Azure auth integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_resource_manager():
    """Test resource manager with centralized settings."""
    print("\n🏗️  Testing Resource Manager Integration")
    print("=" * 50)
    
    try:
        from src.infra.settings import get_settings
        from src.infra.resource_manager import initialize_resources, health_check
        
        settings = get_settings()
        resources = initialize_resources(settings)
        
        print(f"✅ Resources initialized successfully")
        print(f"   Settings Profile: {resources.settings.cloud_profile}")
        print(f"   Chat Client: {'✅' if resources.chat_client else '❌'}")
        print(f"   Embed Client: {'✅' if resources.embed_client else '❌'}")
        print(f"   Search Client: {'✅' if resources.search_client else '❌'}")
        
        # Test health check
        health = health_check()
        print(f"   Overall Health: {health.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Resource manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_structured_logging():
    """Test structured logging setup."""
    print("\n📝 Testing Structured Logging")
    print("=" * 50)
    
    try:
        from src.telemetry.logger import setup_logging, get_logger, Stages
        from src.infra.settings import get_settings
        
        settings = get_settings()
        print(f"✅ Structured logging available")
        print(f"   Enabled: {settings.enable_structured_logging}")
        print(f"   Log Level: {settings.log_level}")
        
        # Test logger creation
        test_logger = get_logger("test_integration")
        test_logger.info("Test log message from integration test")
        print(f"   Logger Created: ✅")
        
        # Test stage definitions
        print(f"   Stage Definitions: {len([attr for attr in dir(Stages) if not attr.startswith('_')])} stages")
        
        return True
        
    except Exception as e:
        print(f"❌ Structured logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("🚀 Centralized Settings Integration Test")
    print("=" * 50)
    
    # Show environment
    profile = os.getenv("CLOUD_PROFILE", "local")
    config_file = os.getenv("UTILITIES_CONFIG", "config.local.ini")
    print(f"Environment: CLOUD_PROFILE={profile}, UTILITIES_CONFIG={config_file}")
    
    # Run tests
    tests = [
        ("Settings Loading", test_settings_loading),
        ("OpenSearch Client", test_opensearch_client), 
        ("Azure Auth Integration", test_azure_auth_integration),
        ("Resource Manager", test_resource_manager),
        ("Structured Logging", test_structured_logging)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed!")
        return 0
    else:
        print("⚠️  Some integration tests failed")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)