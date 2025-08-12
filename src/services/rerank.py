"""Lightweight document reranker with title/section proximity and optional small model scoring."""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import re
import math

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result of reranking operation."""
    documents: List[Dict[str, Any]]
    diagnostics: Dict[str, Any]


class LightweightReranker:
    """Fast, lightweight reranker using simple heuristics and optional small model."""
    
    def __init__(self, 
                 title_boost: float = 2.0,
                 section_boost: float = 1.5,
                 recency_boost: float = 0.1,
                 diversity_threshold: float = 0.8):
        """
        Initialize reranker with configuration.
        
        Args:
            title_boost: Score multiplier for title matches
            section_boost: Score multiplier for section matches  
            recency_boost: Boost per day for recent documents
            diversity_threshold: Similarity threshold for diversity filtering
        """
        self.title_boost = title_boost
        self.section_boost = section_boost
        self.recency_boost = recency_boost
        self.diversity_threshold = diversity_threshold
        
    def rerank(self, 
               documents: List[Dict[str, Any]], 
               query: str,
               max_results: int = 10) -> RerankResult:
        """
        Rerank documents using lightweight scoring.
        
        Args:
            documents: List of documents with scores and metadata
            query: Original search query
            max_results: Maximum number of results to return
            
        Returns:
            RerankResult with reranked documents and diagnostics
        """
        if not documents:
            return RerankResult(documents=[], diagnostics={})
            
        # Extract query terms for proximity matching
        query_terms = self._extract_query_terms(query)
        
        # Score each document
        scored_docs = []
        title_matches = 0
        section_matches = 0
        
        for doc in documents:
            original_score = doc.get('score', 0.0)
            
            # Calculate proximity bonuses
            title_bonus = self._calculate_title_proximity(doc, query_terms)
            section_bonus = self._calculate_section_proximity(doc, query_terms)
            recency_bonus = self._calculate_recency_bonus(doc)
            
            # Combined score
            final_score = (
                original_score * 
                (1 + title_bonus * self.title_boost) *
                (1 + section_bonus * self.section_boost) *
                (1 + recency_bonus)
            )
            
            doc_with_score = doc.copy()
            doc_with_score['rerank_score'] = final_score
            doc_with_score['title_bonus'] = title_bonus
            doc_with_score['section_bonus'] = section_bonus
            doc_with_score['recency_bonus'] = recency_bonus
            
            scored_docs.append(doc_with_score)
            
            if title_bonus > 0:
                title_matches += 1
            if section_bonus > 0:
                section_matches += 1
        
        # Sort by rerank score
        scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        # Apply diversity filtering
        diverse_docs = self._apply_diversity_filter(scored_docs, max_results)
        
        # Prepare diagnostics
        diagnostics = {
            'input_count': len(documents),
            'output_count': len(diverse_docs),
            'title_matches': title_matches,
            'section_matches': section_matches,
            'query_terms': query_terms,
            'avg_boost': sum(d['title_bonus'] + d['section_bonus'] 
                           for d in scored_docs) / len(scored_docs) if scored_docs else 0
        }
        
        return RerankResult(documents=diverse_docs, diagnostics=diagnostics)
    
    def _extract_query_terms(self, query: str) -> Set[str]:
        """Extract meaningful terms from query."""
        # Simple tokenization and filtering
        terms = re.findall(r'\w+', query.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        return {term for term in terms if len(term) > 2 and term not in stop_words}
    
    def _calculate_title_proximity(self, doc: Dict[str, Any], query_terms: Set[str]) -> float:
        """Calculate bonus based on query terms in title."""
        title = doc.get('title', '').lower()
        if not title or not query_terms:
            return 0.0
            
        matches = sum(1 for term in query_terms if term in title)
        return matches / len(query_terms) if query_terms else 0.0
    
    def _calculate_section_proximity(self, doc: Dict[str, Any], query_terms: Set[str]) -> float:
        """Calculate bonus based on query terms in section name."""
        section = doc.get('section', '').lower()
        if not section or not query_terms:
            return 0.0
            
        matches = sum(1 for term in query_terms if term in section)
        return matches / len(query_terms) if query_terms else 0.0
    
    def _calculate_recency_bonus(self, doc: Dict[str, Any]) -> float:
        """Calculate bonus based on document recency."""
        updated_at = doc.get('updated_at')
        if not updated_at:
            return 0.0
            
        try:
            # Simple recency bonus - could be enhanced with actual date parsing
            if '2025' in str(updated_at):
                return self.recency_boost
            elif '2024' in str(updated_at):
                return self.recency_boost * 0.5
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _apply_diversity_filter(self, 
                               docs: List[Dict[str, Any]], 
                               max_results: int) -> List[Dict[str, Any]]:
        """Apply simple diversity filtering to avoid duplicate content."""
        if not docs:
            return []
            
        diverse_docs = [docs[0]]  # Always include top result
        
        for doc in docs[1:]:
            if len(diverse_docs) >= max_results:
                break
                
            # Simple diversity check based on title similarity
            is_diverse = True
            doc_title = doc.get('title', '').lower()
            
            for existing in diverse_docs:
                existing_title = existing.get('title', '').lower()
                if self._calculate_title_similarity(doc_title, existing_title) > self.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_docs.append(doc)
        
        return diverse_docs
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Simple title similarity based on word overlap."""
        if not title1 or not title2:
            return 0.0
            
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


class SmallModelReranker(LightweightReranker):
    """Extended reranker with optional small model scoring."""
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 model_weight: float = 0.3,
                 **kwargs):
        """
        Initialize with optional small model.
        
        Args:
            model_name: Name of small reranking model (e.g., "ms-marco-MiniLM-L-2-v2")
            model_weight: Weight to give model predictions vs heuristic scores
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.model_weight = model_weight
        self.model = None
        
        # Lazy load model if specified
        if model_name:
            self._load_model()
    
    def _load_model(self):
        """Lazy load small reranking model."""
        try:
            from sentence_transformers import SentenceTransformer, util
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded reranking model: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not available, using heuristic-only reranking")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            self.model = None
    
    def rerank(self, 
               documents: List[Dict[str, Any]], 
               query: str,
               max_results: int = 10) -> RerankResult:
        """
        Rerank with optional model scoring.
        """
        # Get base heuristic results
        result = super().rerank(documents, query, max_results)
        
        # Apply model scoring if available
        if self.model and result.documents:
            try:
                model_scores = self._score_with_model(query, result.documents)
                
                # Combine heuristic and model scores
                for doc, model_score in zip(result.documents, model_scores):
                    heuristic_score = doc['rerank_score']
                    combined_score = (
                        (1 - self.model_weight) * heuristic_score + 
                        self.model_weight * model_score
                    )
                    doc['model_score'] = model_score
                    doc['combined_score'] = combined_score
                    doc['rerank_score'] = combined_score
                
                # Re-sort by combined score
                result.documents.sort(key=lambda x: x['rerank_score'], reverse=True)
                result.diagnostics['model_used'] = self.model_name
                result.diagnostics['model_weight'] = self.model_weight
                
            except Exception as e:
                logger.error(f"Model scoring failed: {e}")
                result.diagnostics['model_error'] = str(e)
        
        return result
    
    def _score_with_model(self, query: str, documents: List[Dict[str, Any]]) -> List[float]:
        """Score documents using small model."""
        if not self.model:
            return [0.0] * len(documents)
        
        try:
            from sentence_transformers import util
            
            # Prepare document texts
            doc_texts = []
            for doc in documents:
                text = f"{doc.get('title', '')} {doc.get('section', '')} {doc.get('content', '')}"
                doc_texts.append(text.strip())
            
            # Encode query and documents
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            doc_embeddings = self.model.encode(doc_texts, convert_to_tensor=True)
            
            # Calculate cosine similarities
            similarities = util.cos_sim(query_embedding, doc_embeddings)[0]
            
            return similarities.cpu().tolist()
            
        except Exception as e:
            logger.error(f"Model scoring error: {e}")
            return [0.0] * len(documents)


def create_reranker(config: Optional[Dict[str, Any]] = None) -> LightweightReranker:
    """Factory function to create appropriate reranker."""
    if not config:
        return LightweightReranker()
    
    model_name = config.get('model_name')
    if model_name:
        return SmallModelReranker(
            model_name=model_name,
            model_weight=config.get('model_weight', 0.3),
            title_boost=config.get('title_boost', 2.0),
            section_boost=config.get('section_boost', 1.5),
            recency_boost=config.get('recency_boost', 0.1),
            diversity_threshold=config.get('diversity_threshold', 0.8)
        )
    else:
        return LightweightReranker(
            title_boost=config.get('title_boost', 2.0),
            section_boost=config.get('section_boost', 1.5),
            recency_boost=config.get('recency_boost', 0.1),
            diversity_threshold=config.get('diversity_threshold', 0.8)
        )