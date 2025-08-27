#!/usr/bin/env python3
"""
Test script to verify CIU content flows correctly through the pipeline.

Based on logs showing 3,724 chars of context but LLM still saying "No specific steps found".
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.models import Passage
from services.respond import build_context

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_ciu_passages():
    """Create mock CIU passages similar to what would be found in search."""
    passages = [
        Passage(
            doc_id="ciu_onboarding_001", 
            index="test",
            text="Customer Interaction Utility (CIU) onboarding steps: 1. Create a client ID through the utilities intake process. 2. Configure API endpoints for your application. 3. Test connectivity using the provided sandbox environment. For onboarding details, visit the Digital Utilities Intake page.",
            meta={"title": "CIU Onboarding Guide", "utility_name": "Customer Interaction Utility"},
            score=0.95
        ),
        Passage(
            doc_id="ciu_api_002",
            index="test", 
            text="CIU API Documentation: POST /api/v1/interactions - Creates new customer interaction records. Required fields: clientId, customerId, interactionType. Response includes trackingId for follow-up queries.",
            meta={"title": "CIU API Reference", "utility_name": "Customer Interaction Utility"},
            score=0.92
        ),
        Passage(
            doc_id="ciu_fields_003",
            index="test",
            text="CIU interaction data field definitions: clientId (string) - Your registered client identifier, customerId (string) - Target customer ID, interactionType (enum) - Type of interaction (INQUIRY, COMPLAINT, FEEDBACK).",
            meta={"title": "CIU Field Definitions", "utility_name": "Customer Interaction Utility"}, 
            score=0.88
        )
    ]
    return passages


async def test_context_building():
    """Test that CIU content gets properly formatted into context."""
    logger.info("=== Testing Context Building ===")
    
    # Create mock passages
    passages = create_mock_ciu_passages()
    
    # Build context (simulating combine node behavior)
    context = build_context(
        retrieval_results=passages,
        intent={"intent": "info", "confidence": 0.8}  # Mock intent
    )
    
    logger.info(f"Generated context length: {len(context)} chars")
    logger.info(f"Context preview:\n{context[:500]}...")
    
    # Check if CIU content is present
    ciu_keywords = ["Customer Interaction Utility", "CIU", "onboarding", "clientId", "API"]
    found_keywords = [kw for kw in ciu_keywords if kw.lower() in context.lower()]
    
    logger.info(f"Found CIU keywords: {found_keywords}")
    
    return context, len(context)


def test_template_rendering():
    """Test how the answer.jinja template would process CIU context."""
    logger.info("=== Testing Template Rendering ===")
    
    # Read the current template
    from jinja2 import Environment, FileSystemLoader
    template_dir = Path("src/agent/prompts")
    jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = jinja_env.get_template("answer.jinja")
    
    # Mock context with CIU content
    context = """[Source 1: CIU Onboarding Guide]
Customer Interaction Utility (CIU) onboarding steps: 1. Create a client ID through the utilities intake process. 2. Configure API endpoints for your application. 3. Test connectivity using the provided sandbox environment.

[Source 2: CIU API Reference] 
CIU API Documentation: POST /api/v1/interactions - Creates new customer interaction records. Required fields: clientId, customerId, interactionType."""
    
    query = "what is Customer Interaction Utility"
    intent = {"intent": "info", "confidence": 0.8}
    
    # Render template
    prompt = template.render(
        query=query,
        context=context,
        intent=intent,
        chat_history=[]
    )
    
    logger.info(f"Generated prompt length: {len(prompt)} chars")
    logger.info(f"Prompt preview:\n{prompt[:800]}...")
    
    # Check if template focuses on extraction
    if "extract" in prompt.lower() and "present" in prompt.lower():
        logger.info("✓ Template emphasizes extraction over rejection")
    else:
        logger.warning("⚠ Template may still have rejection bias")
        
    return prompt


async def main():
    """Run all context flow tests."""
    print("🔍 Testing CIU Context Flow Through Pipeline")
    print("=" * 60)
    
    try:
        # Test 1: Context building
        context, context_length = await test_context_building()
        
        # Test 2: Template rendering  
        prompt = test_template_rendering()
        
        # Analysis
        print("\n📊 ANALYSIS:")
        print(f"• Context length: {context_length} chars (original logs showed 3,724)")
        print(f"• Prompt length: {len(prompt)} chars")
        print(f"• CIU content present: {'✓' if 'Customer Interaction Utility' in context else '✗'}")
        print(f"• Template updated: {'✓' if 'Extract and present information' in prompt else '✗'}")
        
        if context_length > 3000 and 'Customer Interaction Utility' in context:
            print("\n✅ DIAGNOSIS: Content flow appears correct")
            print("The original issue was likely the template's rejection bias")
            print("With the updated template emphasizing extraction, this should be resolved")
        else:
            print("\n❌ DIAGNOSIS: Content flow issue detected")
            print("Need to investigate further")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))