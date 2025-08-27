#!/usr/bin/env python3
"""
Test script for the LLM orchestrator with Azure OpenAI.

This verifies that the orchestrator can:
1. Connect to Azure OpenAI in your workspace
2. Plan tool execution
3. Execute searches
4. Generate answers
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_orchestrator():
    """Test the orchestrator with Azure OpenAI."""
    
    # Enable orchestrator
    os.environ["ENABLE_ORCHESTRATOR"] = "true"
    
    # Import after setting env var
    from src.infra.settings import get_settings
    from src.infra.resource_manager import initialize_resources, get_resources
    from src.agent.orchestrator import orchestrated_search
    
    try:
        # Initialize resources (includes Azure OpenAI client)
        logger.info("Initializing resources...")
        settings = get_settings()
        initialize_resources(settings)
        resources = get_resources()
        
        # Verify Azure client is available
        if not resources.chat_client:
            logger.error("Chat client not initialized!")
            return False
            
        logger.info(f"Azure OpenAI client ready: {resources.chat_client}")
        
        # Test queries
        test_queries = [
            "What is CIU?",
            "How do I set up Customer Interaction Utility?",
            "Show me the CIU API documentation",
        ]
        
        for query in test_queries:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing query: {query}")
            logger.info('='*50)
            
            try:
                # Run orchestrated search
                results = await orchestrated_search(
                    query=query,
                    resources=resources,
                    context=None,
                    use_orchestrator=True
                )
                
                # Display results
                logger.info(f"Tool outputs: {len(results.get('tool_outputs', []))} tools executed")
                for tool_output in results.get('tool_outputs', []):
                    logger.info(f"  - {tool_output['tool']}: {tool_output['status']}")
                
                logger.info(f"Search results: {len(results.get('search_results', []))} documents found")
                
                if results.get('final_answer'):
                    logger.info(f"Final answer preview: {results['final_answer'][:200]}...")
                else:
                    logger.warning("No final answer generated")
                    
            except Exception as e:
                logger.error(f"Query failed: {e}", exc_info=True)
                
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return False


async def test_basic_azure_connection():
    """Test basic Azure OpenAI connectivity."""
    
    from src.infra.settings import get_settings
    from src.infra.resource_manager import initialize_resources, get_resources
    
    logger.info("Testing basic Azure OpenAI connection...")
    
    try:
        # Initialize resources
        settings = get_settings()
        initialize_resources(settings)
        resources = get_resources()
        
        if not resources.chat_client:
            logger.error("No chat client available")
            return False
        
        # Try a simple completion
        logger.info("Attempting simple Azure OpenAI call...")
        response = resources.chat_client.chat.completions.create(
            model="gpt-4",  # Will use configured deployment
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Azure connection successful!' if you can read this."}
            ],
            temperature=0,
            max_tokens=50
        )
        
        result = response.choices[0].message.content
        logger.info(f"Azure OpenAI response: {result}")
        
        return "successful" in result.lower()
        
    except Exception as e:
        logger.error(f"Azure connection failed: {e}", exc_info=True)
        return False


async def main():
    """Run all tests."""
    
    logger.info("Starting orchestrator tests...")
    
    # Test basic connectivity first
    if await test_basic_azure_connection():
        logger.info("✓ Azure OpenAI connection successful")
    else:
        logger.error("✗ Azure OpenAI connection failed")
        return
    
    # Test full orchestrator
    if await test_orchestrator():
        logger.info("\n✓ All orchestrator tests passed!")
    else:
        logger.error("\n✗ Some orchestrator tests failed")


if __name__ == "__main__":
    asyncio.run(main())