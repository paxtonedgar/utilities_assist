#!/usr/bin/env python3
"""
Train ONNX intent classifier model for Utilities Assistant.

This script trains a lightweight neural network for intent classification
and exports it to ONNX format for fast inference.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent.intent.onnx_classifier import train_onnx_model, ONNXIntentClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Train and save ONNX intent classifier."""
    logger.info("Training ONNX intent classifier...")
    
    # Train model
    model_path = train_onnx_model()
    
    if model_path:
        logger.info(f"Successfully trained and saved model to {model_path}")
        
        # Test the model
        logger.info("\nTesting the trained model...")
        classifier = ONNXIntentClassifier(model_path)
        
        test_queries = [
            "What is CIU?",
            "How to setup Customer Interaction Utility?",
            "what does etu do",  # Follow-up style
            "Show me CIU API documentation",
            "ETU not working",
        ]
        
        for query in test_queries:
            result = classifier.classify(query)
            logger.info(f"Query: '{query}'")
            logger.info(f"  Intent: {result.intent} (confidence: {result.confidence:.2%})")
            logger.info(f"  Reasoning: {result.reasoning}")
    else:
        logger.error("Failed to train model. Check dependencies.")
        sys.exit(1)

if __name__ == "__main__":
    main()