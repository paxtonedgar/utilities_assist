"""Cross-encoder reranker service using BGE v2-m3 for local relevance scoring.

Provides local cross-encoder reranking after retrieval and before combine/answer.
Takes query and retrieved passages, assigns relevance scores, reorders/filters them,
and exposes scores for confidence/coverage logic.
"""

import logging
import time
import os
from typing import List, Any, Optional, Literal, Tuple
from dataclasses import dataclass
from src.util.cache import ce_token_cache

logger = logging.getLogger(__name__)

# Lazy imports to avoid loading heavy ML dependencies unless reranker is enabled
try:
    import torch
    from sentence_transformers import CrossEncoder

    RERANKER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Reranker dependencies not available: {e}")
    logger.warning(
        "Install with: pip install torch==2.2.2 sentence-transformers==3.0.1"
    )
    RERANKER_AVAILABLE = False
    torch = None
    CrossEncoder = None


@dataclass
class RerankResult:
    """Result of reranking operation."""

    scores: List[float]
    kept_indices: List[int]
    dropped_count: int
    device_used: str
    model_id: str
    took_ms: float
    top_scores: List[float]  # First 3 scores for telemetry
    avg_score: float


class CrossEncodeReranker:
    """Cross-encoder reranker using BGE v2-m3 for local relevance scoring.

    Provides the interface specified in the requirements:
    - ctor: (model_id, device, batch_size)
    - score(query, passages) -> List[float]
    - rerank(query, docs) -> List[Doc]
    """

    def __init__(
        self,
        model_id: str = "BAAI/bge-reranker-v2-m3",
        device: Literal["auto", "mps", "cuda", "cpu"] = "auto",
        batch_size: int = 32,
        max_length: int = 512,
        use_fp16: bool = True,
    ):
        """Initialize cross-encoder reranker.

        Args:
            model_id: Hugging Face model ID
            device: Device preference (auto tries mps -> cuda -> cpu)
            batch_size: Batch size for scoring (8-32 on CPU, 16-64 on GPU)
            max_length: Maximum token length for input truncation
            use_fp16: Use FP16 precision on GPU (not CPU)
        """
        if not RERANKER_AVAILABLE:
            raise RuntimeError(
                "Reranker dependencies not available. "
                "Install with: pip install torch==2.2.2 sentence-transformers==3.0.1"
            )

        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_fp16 = use_fp16

        # Device selection with fallback
        self.device = self._select_device(device)

        # Determine model path - use local model if available
        self.model_path = self._get_model_path(model_id)

        # Set HF_HOME if provided to keep weights out of repo
        if "HF_HOME" in os.environ:
            os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
            os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"]

        # Initialize model
        self.model = None
        self._initialize_model()

        logger.info(
            f"CrossEncodeReranker initialized: model={model_id}, device={self.device}, batch_size={batch_size}"
        )

    def _get_model_path(self, model_id: str) -> str:
        """Get local model path if available, otherwise use HuggingFace ID."""
        if model_id == "BAAI/bge-reranker-v2-m3":
            # Check for local model first
            from pathlib import Path

            local_model_path = (
                Path(__file__).parent.parent.parent / "models" / "bge-reranker-v2-m3"
            )

            if local_model_path.exists():
                model_file = local_model_path / "model.safetensors"
                zip_file = local_model_path / "model-weights.zip"

                # Check if model file exists
                if model_file.exists():
                    logger.info(f"Using local model: {local_model_path}")
                    return str(local_model_path)

                # Check if zip file exists and extract it
                elif zip_file.exists():
                    logger.info(f"Found zipped model, extracting: {zip_file}")
                    self._extract_model_zip(zip_file, local_model_path)
                    if model_file.exists():
                        logger.info(f"Using extracted local model: {local_model_path}")
                        return str(local_model_path)

        # Fallback to HuggingFace ID
        logger.info(f"Using HuggingFace model: {model_id}")
        return model_id

    def _extract_model_zip(self, zip_path, extract_to):
        """Extract model weights from zip file."""
        try:
            import zipfile

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                # Extract only the model.safetensors file
                for file in zip_ref.namelist():
                    if file.endswith("model.safetensors"):
                        # Extract to temporary location first
                        zip_ref.extract(file, extract_to.parent)
                        # Move to the correct location in the model directory
                        extracted_file = extract_to.parent / file
                        target_file = extract_to / "model.safetensors"

                        # Ensure target directory exists
                        target_file.parent.mkdir(parents=True, exist_ok=True)

                        # Move the extracted file to the target location
                        if extracted_file.exists():
                            if target_file.exists():
                                target_file.unlink()  # Remove existing file
                            extracted_file.rename(target_file)
                            logger.info(f"Extracted model weights to: {target_file}")

                            # Clean up any nested directory structure that was created
                            try:
                                nested_dir = extracted_file.parent
                                if (
                                    nested_dir != extract_to.parent
                                    and nested_dir.name in file
                                ):
                                    import shutil

                                    shutil.rmtree(nested_dir, ignore_errors=True)
                            except:
                                pass  # Ignore cleanup errors
                        break
        except Exception as e:
            logger.error(f"Failed to extract model zip: {e}")
            raise

    def _select_device(self, device_pref: str) -> str:
        """Select device with fallback logic."""
        if device_pref == "auto":
            # Try mps -> cuda -> cpu
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        elif device_pref == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif device_pref == "cuda" and torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _initialize_model(self):
        """Initialize the cross-encoder model with device and precision settings."""
        try:
            start_time = time.time()

            # Load model from determined path (local or HuggingFace)
            # Use standard sentence-transformers CrossEncoder loading
            self.model = CrossEncoder(
                self.model_path,
                max_length=self.max_length,
                device=self.device,
                local_files_only=True,  # Force local-only loading
            )

            # Apply FP16 precision on GPU (not CPU)
            if self.use_fp16 and self.device in ["mps", "cuda"]:
                try:
                    if self.device == "mps":
                        # MPS FP16 support
                        self.model.model = self.model.model.half()
                    elif self.device == "cuda":
                        # CUDA FP16 support
                        self.model.model = self.model.model.half()
                    logger.info(f"Applied FP16 precision on {self.device}")
                except Exception as e:
                    logger.warning(f"Failed to apply FP16 on {self.device}: {e}")

            load_time = (time.time() - start_time) * 1000
            logger.info(f"Model loaded in {load_time:.1f}ms on {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize reranker model: {e}")
            raise RuntimeError(f"Reranker model initialization failed: {e}") from e

    def score(self, query: str, passages: List[str]) -> List[float]:
        """Score passages against query using cross-encoder.

        Args:
            query: Search query text
            passages: List of passage texts to score

        Returns:
            List of relevance scores (higher = more relevant)
        """
        if not passages:
            return []

        if self.model is None:
            raise RuntimeError("Reranker model not initialized")

        start_time = time.time()

        try:
            # Create cache key for tokenization caching
            import hashlib

            passage_hash = hashlib.md5("|".join(passages).encode()).hexdigest()[:12]
            cache_key = f"{self.model_id}:{hash(query)}:{passage_hash}"

            # Check tokenization cache
            cached_scores = ce_token_cache.get(cache_key)
            if cached_scores is not None:
                logger.debug(f"CE tokenization cache hit for {len(passages)} passages")
                return cached_scores

            # Create query-passage pairs for cross-encoder
            pairs = [(query, passage) for passage in passages]

            # Score in batches to handle memory efficiently
            all_scores = []
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i : i + self.batch_size]
                batch_scores = self.model.predict(batch)

                # Convert to Python floats for JSON serialization
                if hasattr(batch_scores, "tolist"):
                    batch_scores = batch_scores.tolist()
                elif not isinstance(batch_scores, list):
                    batch_scores = [float(score) for score in batch_scores]

                all_scores.extend(batch_scores)

            score_time = (time.time() - start_time) * 1000

            # Truncate passage text in logs for readability
            truncated_passages = [
                p[:50] + "..." if len(p) > 50 else p for p in passages[:3]
            ]
            logger.debug(
                f"Scored {len(passages)} passages in {score_time:.1f}ms "
                f"(samples: {truncated_passages})"
            )

            # Cache the scores for reuse
            ce_token_cache.put(cache_key, all_scores)

            return all_scores

        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            # Return zero scores as fallback
            return [0.0] * len(passages)

    def rerank(
        self, query: str, docs: List[Any], min_score: float = 0.25, top_k: int = 8
    ) -> Tuple[List[Any], RerankResult]:
        """Rerank documents using cross-encoder scores.

        Args:
            query: Search query text
            docs: List of documents with .text/.content and .meta attributes
            min_score: Minimum score threshold (drop below this)
            top_k: Maximum documents to return

        Returns:
            Tuple of (reranked_docs, rerank_result)
        """
        if not docs:
            return [], RerankResult(
                scores=[],
                kept_indices=[],
                dropped_count=0,
                device_used=self.device,
                model_id=self.model_id,
                took_ms=0.0,
                top_scores=[],
                avg_score=0.0,
            )

        start_time = time.time()

        # Extract text from documents (try .text first, then .content)
        passages = []
        for doc in docs:
            if hasattr(doc, "text") and doc.text:
                passages.append(doc.text)
            elif hasattr(doc, "content") and doc.content:
                passages.append(doc.content)
            else:
                # Fallback to string representation
                passages.append(str(doc))

        # Score all passages
        scores = self.score(query, passages)

        # Create scored document tuples
        scored_docs = [
            (doc, score, idx) for idx, (doc, score) in enumerate(zip(docs, scores))
        ]

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Apply filtering
        kept_docs = []
        kept_indices = []

        for doc, score, original_idx in scored_docs:
            if len(kept_docs) >= top_k:
                break
            if score >= min_score:
                # Set rerank_score attribute on document
                doc.rerank_score = score
                kept_docs.append(doc)
                kept_indices.append(original_idx)

        # Fail-safe: if reranker drops all results, keep top RRF results
        if not kept_docs and docs:
            logger.warning(f"Reranker dropped all {len(docs)} docs (min_score={min_score}), keeping top RRF results")
            # Keep top 3 docs regardless of score to prevent total collapse
            for doc, score, original_idx in scored_docs[:min(3, len(scored_docs))]:
                doc.rerank_score = score
                kept_docs.append(doc)
                kept_indices.append(original_idx)
        
        dropped_count = len(docs) - len(kept_docs)
        took_ms = (time.time() - start_time) * 1000

        # Calculate telemetry metrics
        if scores:
            top_scores = sorted(scores, reverse=True)[:3]
            avg_score = sum(scores) / len(scores)
        else:
            top_scores = []
            avg_score = 0.0

        rerank_result = RerankResult(
            scores=scores,
            kept_indices=kept_indices,
            dropped_count=dropped_count,
            device_used=self.device,
            model_id=self.model_id,
            took_ms=took_ms,
            top_scores=top_scores,
            avg_score=avg_score,
        )

        logger.info(
            f"Reranked {len(docs)} -> {len(kept_docs)} docs "
            f"(dropped: {dropped_count}, avg_score: {avg_score:.3f}, "
            f"took: {took_ms:.1f}ms)"
        )

        return kept_docs, rerank_result


# Singleton instance management
_reranker_instance: Optional[CrossEncodeReranker] = None


def get_reranker(
    model_id: str = "BAAI/bge-reranker-v2-m3",
    device: str = "auto",
    batch_size: int = 32,
    max_length: int = 512,
    use_fp16: bool = True,
    force_reload: bool = False,
) -> Optional[CrossEncodeReranker]:
    """Get singleton reranker instance.

    Args:
        model_id: Hugging Face model ID
        device: Device preference
        batch_size: Batch size for scoring
        max_length: Max token length
        use_fp16: Use FP16 precision on GPU
        force_reload: Force reload of model

    Returns:
        CrossEncodeReranker instance or None if not available
    """
    global _reranker_instance

    if not RERANKER_AVAILABLE:
        logger.warning(
            "Reranker not available - install torch and sentence-transformers"
        )
        return None

    if _reranker_instance is None or force_reload:
        try:
            _reranker_instance = CrossEncodeReranker(
                model_id=model_id,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
                use_fp16=use_fp16,
            )
        except Exception as e:
            logger.error(f"Failed to create reranker instance: {e}")
            return None

    return _reranker_instance


def is_reranker_available() -> bool:
    """Check if reranker dependencies are available."""
    return RERANKER_AVAILABLE
