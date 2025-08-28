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

    def _get_optimal_batch_size(self, num_docs: int) -> int:
        """Get optimal batch size based on document count and device.

        Optimized for our post-swagger fix reality: more documents (8-16 typical).
        Larger batches reduce model loading overhead.
        """
        if num_docs <= 4:
            return num_docs  # Process all at once for very small sets
        elif num_docs <= 16:
            return 16  # Process all 16 at once - reduces batch overhead
        else:
            return min(16, self.batch_size)  # Cap at 16 for larger sets

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
            # Check cache first
            cache_key = self._create_cache_key(query, passages)
            cached_scores = ce_token_cache.get(cache_key)
            if cached_scores is not None:
                logger.debug(f"CE tokenization cache hit for {len(passages)} passages")
                return cached_scores

            # Score all passages in batches
            all_scores = self._score_passages_in_batches(query, passages)
            
            # Log performance and cache results
            self._log_scoring_performance(start_time, len(passages), passages)
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
            return [], self._create_empty_result()

        start_time = time.time()

        # Extract text from documents
        passages, empty_count = self._extract_document_texts(docs)

        # Score all passages
        scores = self.score(query, passages)
        self._log_score_distribution(scores, min_score)

        # Apply adaptive filtering and quality assurance
        kept_docs, kept_indices = self._apply_quality_filtering(
            docs, scores, min_score, top_k, empty_count
        )

        # Build final result
        return self._build_rerank_result(kept_docs, kept_indices, docs, scores, start_time)

    def _create_empty_result(self) -> RerankResult:
        """Create empty result for no input documents."""
        return RerankResult(
            scores=[],
            kept_indices=[],
            dropped_count=0,
            device_used=self.device,
            model_id=self.model_id,
            took_ms=0.0,
            top_scores=[],
            avg_score=0.0,
        )

    def _extract_document_texts(self, docs: List[Any]) -> Tuple[List[str], int]:
        """Extract text from documents with robust fallbacks."""
        passages = []
        empty_count = 0

        for i, doc in enumerate(docs):
            text = self._get_candidate_text(doc)
            if text:
                passages.append(text)
            else:
                passages.append("")  # Keep index alignment
                empty_count += 1
                logger.warning(
                    f"Reranker doc {i}: empty text from {type(doc).__name__} {getattr(doc, 'doc_id', 'unknown')}"
                )

        if empty_count > 0:
            logger.warning(f"Reranker input: {empty_count}/{len(docs)} docs have empty text")

        # Log processing summary
        non_empty_passages = [p for p in passages if p]
        avg_length = sum(len(p) for p in non_empty_passages) / len(non_empty_passages) if non_empty_passages else 0
        logger.info(f"Reranker processing: {len(docs)} docs, {len(non_empty_passages)} with text, avg_length={avg_length:.1f}")

        return passages, empty_count

    def _get_candidate_text(self, candidate: Any) -> str:
        """Extract text from candidate with multiple fallbacks."""
        # Try different text attributes in priority order
        for attr in ("text", "content", "snippet", "body"):
            if hasattr(candidate, attr):
                value = getattr(candidate, attr)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        # Try dict-style access for backwards compatibility
        if isinstance(candidate, dict):
            for key in ("text", "content", "snippet", "body"):
                value = candidate.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

        # Last resort: string representation (avoid short meaningless strings)
        str_repr = str(candidate)
        return str_repr if len(str_repr) > 20 else ""

    def _log_score_distribution(self, scores: List[float], min_score: float) -> None:
        """Log score distribution for debugging."""
        if not scores:
            return

        top_scores = sorted(scores, reverse=True)[:5]
        avg_score = sum(scores) / len(scores)
        above_threshold = sum(1 for s in scores if s >= min_score)

        logger.info(f"Reranker scores: top_5={top_scores}, avg={avg_score:.4f}, min_threshold={min_score}")
        logger.info(f"Reranker threshold filter: {above_threshold}/{len(scores)} docs above {min_score}")

    def _apply_quality_filtering(
        self, docs: List[Any], scores: List[float], min_score: float, top_k: int, empty_count: int
    ) -> Tuple[List[Any], List[int]]:
        """Apply adaptive filtering with quality assurance."""
        # Create scored document tuples and sort by score
        scored_docs = [(doc, score, idx) for idx, (doc, score) in enumerate(zip(docs, scores))]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Calculate adaptive threshold
        effective_min_score = self._calculate_adaptive_threshold(scores, min_score, empty_count, len(docs))

        # Apply primary filtering
        kept_docs, kept_indices = self._apply_threshold_filtering(scored_docs, effective_min_score, top_k)

        # Apply quality assurance (minimum document guarantee)
        return self._ensure_minimum_documents(scored_docs, kept_docs, kept_indices, top_k)

    def _calculate_adaptive_threshold(self, scores: List[float], min_score: float, empty_count: int, total_docs: int) -> float:
        """Calculate adaptive threshold based on input quality."""
        effective_min_score = min_score

        # Lower threshold if too many empty documents
        if empty_count > total_docs * 0.3:
            effective_min_score = min(min_score, 0.005)
            logger.warning(f"High empty text ratio ({empty_count}/{total_docs}), lowering threshold to {effective_min_score}")
        # Lower threshold if all scores are very low
        elif scores and max(scores) < min_score * 2:
            effective_min_score = min(min_score, 0.005)
            logger.warning(f"All scores very low (max={max(scores):.4f}), lowering threshold to {effective_min_score}")

        return effective_min_score

    def _apply_threshold_filtering(self, scored_docs: List, effective_min_score: float, top_k: int) -> Tuple[List[Any], List[int]]:
        """Apply threshold filtering with guaranteed minimum keeps."""
        kept_docs = []
        kept_indices = []

        # Guaranteed keep: always keep top N regardless of threshold (prevents collapse)
        min_keep_count = max(3, len(scored_docs) // 3)  # Keep at least top 33% or 3 docs
        guaranteed_keep = min(min_keep_count, top_k, len(scored_docs))

        for i, (doc, score, original_idx) in enumerate(scored_docs):
            if len(kept_docs) >= top_k:
                break

            # Keep if above threshold OR in guaranteed top percentile
            if score >= effective_min_score or i < guaranteed_keep:
                doc.rerank_score = score
                kept_docs.append(doc)
                kept_indices.append(original_idx)

        return kept_docs, kept_indices

    def _ensure_minimum_documents(self, scored_docs: List, kept_docs: List[Any], kept_indices: List[int], top_k: int) -> Tuple[List[Any], List[int]]:
        """Ensure minimum required documents for context."""
        from src.infra.settings import get_settings
        settings = get_settings()
        min_required_docs = settings.reranker.min_required_docs

        if len(kept_docs) >= min_required_docs:
            return kept_docs, kept_indices

        shortage = min_required_docs - len(kept_docs)
        logger.warning(f"Reranker kept only {len(kept_docs)} docs (need {min_required_docs}), adding {shortage} top results")

        # Add top remaining docs that weren't already kept
        kept_doc_ids = {getattr(doc, "doc_id", id(doc)) for doc in kept_docs}
        added = 0

        for doc, score, original_idx in scored_docs:
            if added >= shortage:
                break

            doc_id = getattr(doc, "doc_id", id(doc))
            if doc_id not in kept_doc_ids:
                doc.rerank_score = score
                kept_docs.append(doc)
                kept_indices.append(original_idx)
                kept_doc_ids.add(doc_id)
                added += 1

        return kept_docs, kept_indices

    def _build_rerank_result(self, kept_docs: List[Any], kept_indices: List[int], original_docs: List[Any], scores: List[float], start_time: float) -> Tuple[List[Any], RerankResult]:
        """Build final rerank result with telemetry."""
        dropped_count = len(original_docs) - len(kept_docs)
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

        logger.info(f"Reranked {len(original_docs)} -> {len(kept_docs)} docs (dropped: {dropped_count}, avg_score: {avg_score:.3f}, took: {took_ms:.1f}ms)")

        return kept_docs, rerank_result

    def _create_cache_key(self, query: str, passages: List[str]) -> str:
        """Create cache key for passage scoring."""
        import hashlib
        passage_hash = hashlib.md5("|".join(passages).encode()).hexdigest()[:12]
        return f"{self.model_id}:{hash(query)}:{passage_hash}"

    def _score_passages_in_batches(self, query: str, passages: List[str]) -> List[float]:
        """Score passages in batches for memory efficiency."""
        # Create query-passage pairs for cross-encoder
        pairs = [(query, passage) for passage in passages]
        
        # Dynamic batch sizing for small document sets
        effective_batch_size = self._get_optimal_batch_size(len(passages))
        
        all_scores = []
        for i in range(0, len(pairs), effective_batch_size):
            batch = pairs[i : i + effective_batch_size]
            batch_scores = self.model.predict(batch)
            
            # Convert to Python floats for JSON serialization
            batch_scores = self._normalize_batch_scores(batch_scores)
            all_scores.extend(batch_scores)
        
        return all_scores

    def _normalize_batch_scores(self, batch_scores) -> List[float]:
        """Normalize batch scores to Python floats for JSON serialization."""
        if hasattr(batch_scores, "tolist"):
            return batch_scores.tolist()
        elif not isinstance(batch_scores, list):
            return [float(score) for score in batch_scores]
        return batch_scores

    def _log_scoring_performance(self, start_time: float, num_passages: int, passages: List[str]) -> None:
        """Log scoring performance metrics and debug information."""
        score_time = (time.time() - start_time) * 1000
        
        # Performance analysis log
        if score_time > 2000:  # Log if >2s
            logger.warning(
                f"Cross-encoder slow: {score_time:.0f}ms for {num_passages} docs on {self.device} (avg: {score_time / num_passages:.0f}ms/doc)"
            )
        
        # Truncate passage text in logs for readability
        truncated_passages = [
            p[:50] + "..." if len(p) > 50 else p for p in passages[:3]
        ]
        logger.debug(
            f"Scored {num_passages} passages in {score_time:.1f}ms "
            f"(samples: {truncated_passages})"
        )


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