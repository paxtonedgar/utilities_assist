"""
ONNX-based Intent Classifier for Utilities Assistant

This module provides a neural network-based intent classifier that can:
1. Understand context from previous turns
2. Better classify ambiguous queries
3. Handle utility-specific intents
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ONNXIntentResult:
    """Result from ONNX intent classification."""
    intent: str
    confidence: float
    reasoning: str
    context_aware: bool = False
    features: Dict[str, float] = None


class ONNXIntentClassifier:
    """
    ONNX-based intent classifier with context awareness.
    
    This classifier uses a lightweight neural network to classify intents
    based on both the current query and conversation history.
    """
    
    # Intent categories
    INTENTS = [
        "info",           # What is X? Definition queries
        "procedure",      # How to do X? Step-by-step queries
        "troubleshoot",   # Fix/debug/resolve issues
        "api",           # API documentation/swagger
        "list",          # List/enumerate items
        "utility",       # Utility-specific queries (CIU, ETU, etc)
        "followup",      # Follow-up question using context
    ]
    
    # Utility keywords for feature extraction
    UTILITY_KEYWORDS = {
        "ciu": ["customer interaction utility", "ciu", "interaction"],
        "etu": ["enhanced transaction utility", "etu", "transaction"],
        "csu": ["customer summary utility", "csu", "summary"],
        "au": ["account utility", "au", "account"],
        "pcu": ["product catalog utility", "pcu", "catalog"],
    }
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ONNX classifier.
        
        Args:
            model_path: Path to ONNX model file. If None, will train a new model.
        """
        self.model_path = model_path or self._get_default_model_path()
        self.session = None
        self.vectorizer = None
        self.feature_dim = 512  # Feature vector dimension
        
        # Try to load existing model
        if os.path.exists(self.model_path):
            self._load_model()
        else:
            logger.warning(f"No ONNX model found at {self.model_path}. Will use fallback.")
            
    def _get_default_model_path(self) -> str:
        """Get default path for ONNX model."""
        return str(Path(__file__).parent.parent.parent.parent / "models" / "intent_classifier.onnx")
    
    def _load_model(self):
        """Load ONNX model and vectorizer."""
        try:
            import onnxruntime as ort
            self.session = ort.InferenceSession(self.model_path)
            
            # Load vectorizer config
            vectorizer_path = self.model_path.replace('.onnx', '_vectorizer.json')
            if os.path.exists(vectorizer_path):
                with open(vectorizer_path, 'r') as f:
                    self.vectorizer = json.load(f)
            
            logger.info(f"Loaded ONNX model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            self.session = None
    
    def extract_features(self, query: str, context: Optional[List[str]] = None) -> np.ndarray:
        """
        Extract features from query and context.
        
        Args:
            query: Current user query
            context: Previous conversation turns
            
        Returns:
            Feature vector of shape (feature_dim,)
        """
        features = []
        query_lower = query.lower()
        
        # 1. Query length features
        features.append(len(query.split()))  # Word count
        features.append(len(query))  # Character count
        
        # 2. Question indicators
        question_words = ["what", "how", "why", "when", "where", "who", "which", "does", "can", "is"]
        features.extend([1.0 if word in query_lower else 0.0 for word in question_words])
        
        # 3. Action indicators
        action_words = ["create", "setup", "configure", "install", "build", "implement", "deploy"]
        features.extend([1.0 if word in query_lower else 0.0 for word in action_words])
        
        # 4. Utility-specific features
        for utility, keywords in self.UTILITY_KEYWORDS.items():
            utility_score = sum(1.0 for kw in keywords if kw in query_lower)
            features.append(utility_score)
        
        # 5. API/Technical indicators
        api_words = ["api", "endpoint", "swagger", "rest", "post", "get", "request", "response"]
        features.extend([1.0 if word in query_lower else 0.0 for word in api_words])
        
        # 6. Context features (if available)
        if context:
            # Check if this is a follow-up
            features.append(1.0 if len(query_lower.split()) < 5 else 0.0)  # Short follow-up
            
            # Check for pronouns indicating follow-up
            pronouns = ["it", "this", "that", "they", "them"]
            features.append(sum(1.0 for p in pronouns if p in query_lower))
            
            # Check if previous turn mentioned utilities
            prev_context = " ".join(context[-2:] if len(context) >= 2 else context).lower()
            for utility in self.UTILITY_KEYWORDS:
                features.append(1.0 if utility in prev_context else 0.0)
        else:
            # No context features
            features.extend([0.0] * 8)  # Placeholder for context features
        
        # 7. Troubleshooting indicators
        trouble_words = ["error", "issue", "problem", "fail", "not working", "broken", "fix"]
        features.append(sum(1.0 for word in trouble_words if word in query_lower))
        
        # Pad or truncate to feature_dim
        features = np.array(features, dtype=np.float32)
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)), mode='constant')
        else:
            features = features[:self.feature_dim]
        
        return features.reshape(1, -1)  # Shape: (1, feature_dim)
    
    def classify(self, query: str, context: Optional[List[str]] = None) -> ONNXIntentResult:
        """
        Classify intent using ONNX model.
        
        Args:
            query: User query
            context: Previous conversation turns
            
        Returns:
            ONNXIntentResult with classification
        """
        # Extract features
        features = self.extract_features(query, context)
        
        # If ONNX model is available, use it
        if self.session:
            try:
                # Run inference
                input_name = self.session.get_inputs()[0].name
                output_name = self.session.get_outputs()[0].name
                
                result = self.session.run([output_name], {input_name: features})[0]
                
                # Handle different ONNX output formats
                if result.ndim == 0:  # Scalar output
                    # Single prediction, create probabilities
                    intent_idx = int(result)
                    probs = np.zeros(len(self.INTENTS))
                    probs[intent_idx] = 1.0
                    confidence = 1.0
                elif result.ndim == 1:  # 1D array of probabilities
                    probs = self._softmax(result)
                    intent_idx = np.argmax(probs)
                    confidence = float(probs[intent_idx])
                else:  # 2D array, take first row
                    probs = self._softmax(result[0])
                    intent_idx = np.argmax(probs)
                    confidence = float(probs[intent_idx])
                intent = self.INTENTS[intent_idx]
                
                # Check if this is context-aware
                context_aware = context is not None and len(context) > 0
                
                return ONNXIntentResult(
                    intent=intent,
                    confidence=confidence,
                    reasoning=f"ONNX model prediction (context-aware: {context_aware})",
                    context_aware=context_aware,
                    features={"prob_" + i: float(p) for i, p in zip(self.INTENTS, probs)}
                )
                
            except Exception as e:
                logger.error(f"ONNX inference failed: {e}")
        
        # Fallback to rule-based classification
        return self._rule_based_fallback(query, context)
    
    def _rule_based_fallback(self, query: str, context: Optional[List[str]] = None) -> ONNXIntentResult:
        """
        Rule-based fallback when ONNX model is not available.
        """
        query_lower = query.lower()
        
        # Check for utility queries
        for utility, keywords in self.UTILITY_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return ONNXIntentResult(
                    intent="utility",
                    confidence=0.8,
                    reasoning=f"Detected {utility.upper()} utility query",
                    context_aware=False
                )
        
        # Check for follow-up queries
        if context and len(query_lower.split()) < 5:
            if any(word in query_lower for word in ["it", "this", "that", "what about"]):
                return ONNXIntentResult(
                    intent="followup",
                    confidence=0.7,
                    reasoning="Short follow-up query detected",
                    context_aware=True
                )
        
        # Check for procedures
        if any(word in query_lower for word in ["how to", "setup", "configure", "create"]):
            return ONNXIntentResult(
                intent="procedure",
                confidence=0.75,
                reasoning="Procedure/how-to query detected",
                context_aware=False
            )
        
        # Check for API queries
        if any(word in query_lower for word in ["api", "endpoint", "swagger"]):
            return ONNXIntentResult(
                intent="api",
                confidence=0.8,
                reasoning="API documentation query detected",
                context_aware=False
            )
        
        # Default to info query
        return ONNXIntentResult(
            intent="info",
            confidence=0.6,
            reasoning="Default classification as info query",
            context_aware=False
        )
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax values for array x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


def create_training_data() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create training data for the ONNX model from existing patterns.
    
    Returns:
        X: Feature matrix
        y: Label vector
    """
    # Training examples
    examples = [
        # Info queries
        ("What is CIU?", "info"),
        ("What does Customer Interaction Utility do?", "info"),
        ("Tell me about ETU", "info"),
        ("Explain Enhanced Transaction Utility", "info"),
        
        # Procedure queries
        ("How to setup CIU?", "procedure"),
        ("How do I create client IDs in CIU?", "procedure"),
        ("Steps to onboard to Customer Interaction Utility", "procedure"),
        ("Configure ETU for my application", "procedure"),
        
        # API queries
        ("CIU API documentation", "api"),
        ("Show me the ETU swagger", "api"),
        ("What endpoints does CIU expose?", "api"),
        
        # Utility queries
        ("CIU onboarding runbook", "utility"),
        ("ETU integration guide", "utility"),
        ("Customer Summary Utility fields", "utility"),
        
        # Follow-up queries (with context)
        ("What about ETU?", "followup"),
        ("How does it work?", "followup"),
        ("What are the steps?", "followup"),
        
        # Troubleshooting
        ("CIU connection error", "troubleshoot"),
        ("ETU not working", "troubleshoot"),
        ("Fix authentication issue with CIU", "troubleshoot"),
    ]
    
    classifier = ONNXIntentClassifier()
    X = []
    y = []
    
    for query, label in examples:
        # Extract features (no context for most examples)
        features = classifier.extract_features(query, None)
        X.append(features[0])
        
        # Convert label to index
        label_idx = classifier.INTENTS.index(label)
        y.append(label_idx)
    
    return np.array(X), np.array(y)


def train_onnx_model(save_path: Optional[str] = None):
    """
    Train and save an ONNX intent classification model.
    
    Args:
        save_path: Where to save the model
    """
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split
        import skl2onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        # Get training data
        X, y = create_training_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train MLP classifier
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        clf.fit(X_train, y_train)
        
        # Evaluate
        accuracy = clf.score(X_test, y_test)
        logger.info(f"Model accuracy: {accuracy:.2%}")
        
        # Convert to ONNX
        initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
        onx = convert_sklearn(clf, initial_types=initial_type)
        
        # Save model
        if save_path is None:
            save_path = ONNXIntentClassifier()._get_default_model_path()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(onx.SerializeToString())
        
        logger.info(f"Saved ONNX model to {save_path}")
        
        # Save vectorizer config
        vectorizer_config = {
            "feature_dim": X.shape[1],
            "intents": ONNXIntentClassifier.INTENTS,
            "accuracy": float(accuracy)
        }
        
        vectorizer_path = save_path.replace('.onnx', '_vectorizer.json')
        with open(vectorizer_path, 'w') as f:
            json.dump(vectorizer_config, f, indent=2)
        
        logger.info(f"Saved vectorizer config to {vectorizer_path}")
        
        return save_path
        
    except ImportError as e:
        logger.error(f"Missing dependencies for training: {e}")
        logger.error("Install with: pip install scikit-learn skl2onnx onnxruntime")
        return None


# Singleton instance
_classifier_instance: Optional[ONNXIntentClassifier] = None


def get_onnx_classifier() -> ONNXIntentClassifier:
    """Get singleton ONNX classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ONNXIntentClassifier()
    return _classifier_instance