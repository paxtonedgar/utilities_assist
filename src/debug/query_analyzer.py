"""Query analyzer for OpenSearch query structure and optimization analysis."""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class QueryComponent:
    """Represents a component of an OpenSearch query."""
    type: str
    field: Optional[str]
    value: Any
    boost: Optional[float] = None
    complexity_score: int = 1
    estimated_cost: str = "low"  # low, medium, high
    

@dataclass
class QueryAnalysis:
    """Results of query analysis."""
    query_type: str
    components: List[QueryComponent]
    complexity_score: int
    estimated_performance: str
    optimization_suggestions: List[str]
    field_usage: Dict[str, int]
    boost_analysis: Dict[str, Any]
    

class QueryAnalyzer:
    """Analyzer for OpenSearch query structure and optimization."""
    
    # Query complexity weights
    COMPLEXITY_WEIGHTS = {
        "term": 1,
        "match": 2,
        "multi_match": 3,
        "bool": 2,
        "dis_max": 2,
        "wildcard": 5,
        "regexp": 8,
        "fuzzy": 4,
        "prefix": 3,
        "range": 2,
        "exists": 1,
        "nested": 6,
        "knn": 4,
        "script": 10,
        "function_score": 7
    }
    
    # Performance cost estimates
    PERFORMANCE_COSTS = {
        "term": "low",
        "match": "low",
        "multi_match": "medium",
        "bool": "medium",
        "dis_max": "medium",
        "wildcard": "high",
        "regexp": "very_high",
        "fuzzy": "high",
        "prefix": "medium",
        "range": "low",
        "exists": "low",
        "nested": "high",
        "knn": "medium",
        "script": "very_high",
        "function_score": "high"
    }
    
    def __init__(self):
        """Initialize query analyzer."""
        pass
    
    def analyze_query(self, query: Dict[str, Any]) -> QueryAnalysis:
        """Analyze an OpenSearch query structure.
        
        Args:
            query: OpenSearch query dictionary
            
        Returns:
            Query analysis results
        """
        logger.info("Analyzing query structure")
        
        # Extract query components
        components = self._extract_query_components(query)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(components)
        
        # Determine query type
        query_type = self._determine_query_type(query)
        
        # Estimate performance
        estimated_performance = self._estimate_performance(components, complexity_score)
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(
            query, components, complexity_score
        )
        
        # Analyze field usage
        field_usage = self._analyze_field_usage(components)
        
        # Analyze boost configuration
        boost_analysis = self._analyze_boost_configuration(components)
        
        return QueryAnalysis(
            query_type=query_type,
            components=components,
            complexity_score=complexity_score,
            estimated_performance=estimated_performance,
            optimization_suggestions=optimization_suggestions,
            field_usage=field_usage,
            boost_analysis=boost_analysis
        )
    
    def compare_queries(
        self,
        queries: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compare multiple queries for performance and complexity.
        
        Args:
            queries: Dictionary of query name to query dict
            
        Returns:
            Comparison analysis
        """
        analyses = {}
        
        # Analyze each query
        for name, query in queries.items():
            analyses[name] = self.analyze_query(query)
        
        # Generate comparison
        comparison = {
            "query_analyses": analyses,
            "complexity_ranking": self._rank_by_complexity(analyses),
            "performance_ranking": self._rank_by_performance(analyses),
            "field_usage_comparison": self._compare_field_usage(analyses),
            "optimization_priorities": self._prioritize_optimizations(analyses)
        }
        
        return comparison
    
    def suggest_query_optimizations(
        self,
        query: Dict[str, Any],
        performance_target: str = "medium"
    ) -> Dict[str, Any]:
        """Suggest specific optimizations for a query.
        
        Args:
            query: OpenSearch query to optimize
            performance_target: Target performance level
            
        Returns:
            Optimization suggestions with modified queries
        """
        analysis = self.analyze_query(query)
        
        suggestions = {
            "original_analysis": analysis,
            "optimizations": [],
            "optimized_queries": {}
        }
        
        # Generate specific optimization strategies
        optimizations = self._generate_specific_optimizations(
            query, analysis, performance_target
        )
        
        for optimization in optimizations:
            suggestions["optimizations"].append(optimization)
            
            # Create optimized query if possible
            if "modified_query" in optimization:
                opt_name = optimization["name"]
                suggestions["optimized_queries"][opt_name] = optimization["modified_query"]
        
        return suggestions
    
    def _extract_query_components(
        self,
        query: Dict[str, Any],
        path: str = "root"
    ) -> List[QueryComponent]:
        """Extract components from query recursively.
        
        Args:
            query: Query dictionary
            path: Current path in query structure
            
        Returns:
            List of query components
        """
        components = []
        
        if isinstance(query, dict):
            for key, value in query.items():
                if key == "query" and isinstance(value, dict):
                    # Recurse into query object
                    components.extend(self._extract_query_components(value, f"{path}.query"))
                elif key in self.COMPLEXITY_WEIGHTS:
                    # This is a query type
                    component = self._create_query_component(key, value, path)
                    components.append(component)
                    
                    # Recurse into nested queries
                    if isinstance(value, dict):
                        components.extend(self._extract_query_components(value, f"{path}.{key}"))
                elif isinstance(value, (dict, list)):
                    # Recurse into nested structures
                    components.extend(self._extract_query_components(value, f"{path}.{key}"))
        elif isinstance(query, list):
            for i, item in enumerate(query):
                components.extend(self._extract_query_components(item, f"{path}[{i}]"))
        
        return components
    
    def _create_query_component(
        self,
        query_type: str,
        query_value: Any,
        path: str
    ) -> QueryComponent:
        """Create a query component from query type and value.
        
        Args:
            query_type: Type of query (e.g., 'match', 'term')
            query_value: Query value/configuration
            path: Path in query structure
            
        Returns:
            QueryComponent instance
        """
        field = None
        value = None
        boost = None
        
        if isinstance(query_value, dict):
            # Extract field and value information
            if len(query_value) == 1 and query_type in ["match", "term", "wildcard", "prefix"]:
                field = list(query_value.keys())[0]
                field_config = query_value[field]
                
                if isinstance(field_config, dict):
                    value = field_config.get("query", field_config.get("value", str(field_config)))
                    boost = field_config.get("boost")
                else:
                    value = field_config
            elif "fields" in query_value:
                # Multi-field query
                field = ",".join(query_value["fields"])
                value = query_value.get("query", "")
                boost = query_value.get("boost")
            else:
                value = str(query_value)
        else:
            value = query_value
        
        return QueryComponent(
            type=query_type,
            field=field,
            value=value,
            boost=boost,
            complexity_score=self.COMPLEXITY_WEIGHTS.get(query_type, 1),
            estimated_cost=self.PERFORMANCE_COSTS.get(query_type, "medium")
        )
    
    def _calculate_complexity_score(self, components: List[QueryComponent]) -> int:
        """Calculate overall complexity score for query.
        
        Args:
            components: List of query components
            
        Returns:
            Complexity score
        """
        base_score = sum(component.complexity_score for component in components)
        
        # Apply multipliers for certain patterns
        multiplier = 1.0
        
        # Nested queries increase complexity
        nested_count = sum(1 for c in components if c.type == "nested")
        if nested_count > 0:
            multiplier *= (1 + nested_count * 0.5)
        
        # Script queries significantly increase complexity
        script_count = sum(1 for c in components if c.type == "script")
        if script_count > 0:
            multiplier *= (1 + script_count * 2)
        
        # Many components increase complexity
        if len(components) > 10:
            multiplier *= 1.3
        elif len(components) > 20:
            multiplier *= 1.6
        
        return int(base_score * multiplier)
    
    def _determine_query_type(self, query: Dict[str, Any]) -> str:
        """Determine the primary type of query.
        
        Args:
            query: Query dictionary
            
        Returns:
            Query type string
        """
        if "query" in query:
            query = query["query"]
        
        if "bool" in query:
            return "boolean"
        elif "dis_max" in query:
            return "disjunction_max"
        elif "multi_match" in query:
            return "multi_match"
        elif "match" in query:
            return "match"
        elif "term" in query:
            return "term"
        elif "knn" in query:
            return "vector_search"
        elif "function_score" in query:
            return "function_score"
        elif "nested" in query:
            return "nested"
        else:
            return "complex"
    
    def _estimate_performance(self, components: List[QueryComponent], complexity_score: int) -> str:
        """Estimate query performance based on components and complexity.
        
        Args:
            components: List of query components
            complexity_score: Overall complexity score
            
        Returns:
            Performance estimate (excellent, good, fair, poor, very_poor)
        """
        # Check for expensive operations
        has_very_high_cost = any(c.estimated_cost == "very_high" for c in components)
        has_high_cost = any(c.estimated_cost == "high" for c in components)
        high_cost_count = sum(1 for c in components if c.estimated_cost in ["high", "very_high"])
        
        if has_very_high_cost or complexity_score > 50:
            return "very_poor"
        elif has_high_cost and high_cost_count > 2 or complexity_score > 30:
            return "poor"
        elif has_high_cost or complexity_score > 20:
            return "fair"
        elif complexity_score > 10:
            return "good"
        else:
            return "excellent"
    
    def _generate_optimization_suggestions(
        self,
        query: Dict[str, Any],
        components: List[QueryComponent],
        complexity_score: int
    ) -> List[str]:
        """Generate optimization suggestions for query.
        
        Args:
            query: Original query
            components: Query components
            complexity_score: Complexity score
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Check for expensive operations
        expensive_ops = [c for c in components if c.estimated_cost in ["high", "very_high"]]
        for op in expensive_ops:
            if op.type == "wildcard":
                suggestions.append(f"Replace wildcard query on '{op.field}' with prefix query if possible")
            elif op.type == "regexp":
                suggestions.append(f"Replace regexp query on '{op.field}' with simpler alternatives")
            elif op.type == "script":
                suggestions.append(f"Consider pre-computing script results or using function_score instead")
            elif op.type == "fuzzy":
                suggestions.append(f"Reduce fuzziness or use match query with fuzziness parameter")
        
        # Check complexity
        if complexity_score > 30:
            suggestions.append("Query is very complex - consider breaking into multiple simpler queries")
            suggestions.append("Use filters instead of queries where exact matches are needed")
        
        # Check for boost optimization
        boosted_components = [c for c in components if c.boost is not None]
        if len(boosted_components) > 5:
            suggestions.append("Many boosted fields detected - review boost values for effectiveness")
        
        # Check for field usage patterns
        field_counts = defaultdict(int)
        for component in components:
            if component.field:
                field_counts[component.field] += 1
        
        repeated_fields = [field for field, count in field_counts.items() if count > 3]
        if repeated_fields:
            suggestions.append(f"Fields {repeated_fields} used multiple times - consider query consolidation")
        
        # Check for boolean query optimization
        bool_components = [c for c in components if c.type == "bool"]
        if len(bool_components) > 2:
            suggestions.append("Multiple boolean queries detected - consider flattening structure")
        
        # General suggestions based on query structure
        if "sort" in query and len(components) > 10:
            suggestions.append("Complex query with sorting - consider using search_after for pagination")
        
        if "_source" not in query:
            suggestions.append("Consider limiting _source fields to reduce response size")
        
        return suggestions
    
    def _analyze_field_usage(self, components: List[QueryComponent]) -> Dict[str, int]:
        """Analyze field usage patterns in query.
        
        Args:
            components: Query components
            
        Returns:
            Field usage counts
        """
        field_usage = defaultdict(int)
        
        for component in components:
            if component.field:
                # Handle multi-field cases
                if "," in component.field:
                    for field in component.field.split(","):
                        field_usage[field.strip()] += 1
                else:
                    field_usage[component.field] += 1
        
        return dict(field_usage)
    
    def _analyze_boost_configuration(self, components: List[QueryComponent]) -> Dict[str, Any]:
        """Analyze boost configuration in query.
        
        Args:
            components: Query components
            
        Returns:
            Boost analysis
        """
        boosted_components = [c for c in components if c.boost is not None]
        
        if not boosted_components:
            return {"has_boosts": False}
        
        boost_values = [c.boost for c in boosted_components]
        boost_fields = [c.field for c in boosted_components if c.field]
        
        analysis = {
            "has_boosts": True,
            "boost_count": len(boosted_components),
            "boost_range": {
                "min": min(boost_values),
                "max": max(boost_values),
                "avg": sum(boost_values) / len(boost_values)
            },
            "boosted_fields": boost_fields,
            "boost_distribution": self._analyze_boost_distribution(boost_values)
        }
        
        # Check for boost optimization opportunities
        if analysis["boost_range"]["max"] > 10:
            analysis["warnings"] = analysis.get("warnings", [])
            analysis["warnings"].append("Very high boost values detected - may skew results")
        
        if len(set(boost_values)) == 1:
            analysis["warnings"] = analysis.get("warnings", [])
            analysis["warnings"].append("All boost values are the same - consider removing boosts")
        
        return analysis
    
    def _analyze_boost_distribution(self, boost_values: List[float]) -> Dict[str, int]:
        """Analyze distribution of boost values.
        
        Args:
            boost_values: List of boost values
            
        Returns:
            Distribution analysis
        """
        distribution = {
            "low (< 2)": 0,
            "medium (2-5)": 0,
            "high (5-10)": 0,
            "very_high (> 10)": 0
        }
        
        for boost in boost_values:
            if boost < 2:
                distribution["low (< 2)"] += 1
            elif boost <= 5:
                distribution["medium (2-5)"] += 1
            elif boost <= 10:
                distribution["high (5-10)"] += 1
            else:
                distribution["very_high (> 10)"] += 1
        
        return distribution
    
    def _rank_by_complexity(self, analyses: Dict[str, QueryAnalysis]) -> List[Tuple[str, int]]:
        """Rank queries by complexity score.
        
        Args:
            analyses: Query analyses
            
        Returns:
            List of (query_name, complexity_score) tuples, sorted by complexity
        """
        return sorted(
            [(name, analysis.complexity_score) for name, analysis in analyses.items()],
            key=lambda x: x[1],
            reverse=True
        )
    
    def _rank_by_performance(self, analyses: Dict[str, QueryAnalysis]) -> List[Tuple[str, str]]:
        """Rank queries by estimated performance.
        
        Args:
            analyses: Query analyses
            
        Returns:
            List of (query_name, performance) tuples, sorted by performance
        """
        performance_order = ["very_poor", "poor", "fair", "good", "excellent"]
        
        return sorted(
            [(name, analysis.estimated_performance) for name, analysis in analyses.items()],
            key=lambda x: performance_order.index(x[1])
        )
    
    def _compare_field_usage(self, analyses: Dict[str, QueryAnalysis]) -> Dict[str, Any]:
        """Compare field usage across queries.
        
        Args:
            analyses: Query analyses
            
        Returns:
            Field usage comparison
        """
        all_fields = set()
        for analysis in analyses.values():
            all_fields.update(analysis.field_usage.keys())
        
        comparison = {
            "total_unique_fields": len(all_fields),
            "field_usage_by_query": {name: analysis.field_usage for name, analysis in analyses.items()},
            "most_used_fields": {},
            "query_field_overlap": {}
        }
        
        # Calculate most used fields across all queries
        field_totals = defaultdict(int)
        for analysis in analyses.values():
            for field, count in analysis.field_usage.items():
                field_totals[field] += count
        
        comparison["most_used_fields"] = dict(
            sorted(field_totals.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        # Calculate field overlap between queries
        query_names = list(analyses.keys())
        for i, query1 in enumerate(query_names):
            for query2 in query_names[i+1:]:
                fields1 = set(analyses[query1].field_usage.keys())
                fields2 = set(analyses[query2].field_usage.keys())
                
                overlap = len(fields1.intersection(fields2))
                total = len(fields1.union(fields2))
                
                comparison["query_field_overlap"][f"{query1}_vs_{query2}"] = {
                    "overlap_count": overlap,
                    "jaccard_similarity": overlap / total if total > 0 else 0
                }
        
        return comparison
    
    def _prioritize_optimizations(self, analyses: Dict[str, QueryAnalysis]) -> List[Dict[str, Any]]:
        """Prioritize optimization opportunities across queries.
        
        Args:
            analyses: Query analyses
            
        Returns:
            List of prioritized optimization opportunities
        """
        priorities = []
        
        for name, analysis in analyses.items():
            if analysis.estimated_performance in ["poor", "very_poor"]:
                priorities.append({
                    "query": name,
                    "priority": "high",
                    "reason": f"Poor performance ({analysis.estimated_performance})",
                    "complexity_score": analysis.complexity_score,
                    "suggestions": analysis.optimization_suggestions[:3]  # Top 3 suggestions
                })
            elif analysis.complexity_score > 20:
                priorities.append({
                    "query": name,
                    "priority": "medium",
                    "reason": f"High complexity (score: {analysis.complexity_score})",
                    "complexity_score": analysis.complexity_score,
                    "suggestions": analysis.optimization_suggestions[:2]
                })
        
        # Sort by priority and complexity
        priority_order = {"high": 3, "medium": 2, "low": 1}
        priorities.sort(
            key=lambda x: (priority_order[x["priority"]], x["complexity_score"]),
            reverse=True
        )
        
        return priorities
    
    def _generate_specific_optimizations(
        self,
        query: Dict[str, Any],
        analysis: QueryAnalysis,
        performance_target: str
    ) -> List[Dict[str, Any]]:
        """Generate specific optimization strategies with modified queries.
        
        Args:
            query: Original query
            analysis: Query analysis
            performance_target: Target performance level
            
        Returns:
            List of specific optimization strategies
        """
        optimizations = []
        
        # Optimization 1: Convert wildcards to prefix queries
        wildcard_components = [c for c in analysis.components if c.type == "wildcard"]
        if wildcard_components:
            optimizations.append({
                "name": "wildcard_to_prefix",
                "description": "Convert wildcard queries to prefix queries for better performance",
                "impact": "high",
                "fields_affected": [c.field for c in wildcard_components if c.field]
            })
        
        # Optimization 2: Add filters for exact matches
        term_components = [c for c in analysis.components if c.type == "term"]
        if term_components:
            optimizations.append({
                "name": "term_to_filter",
                "description": "Move term queries to filter context for better caching",
                "impact": "medium",
                "fields_affected": [c.field for c in term_components if c.field]
            })
        
        # Optimization 3: Reduce boost complexity
        if analysis.boost_analysis.get("has_boosts") and len(analysis.boost_analysis.get("boosted_fields", [])) > 5:
            optimizations.append({
                "name": "simplify_boosts",
                "description": "Reduce number of boosted fields to improve performance",
                "impact": "medium",
                "current_boost_count": analysis.boost_analysis["boost_count"]
            })
        
        # Optimization 4: Query structure simplification
        if analysis.complexity_score > 25:
            optimizations.append({
                "name": "simplify_structure",
                "description": "Simplify query structure to reduce complexity",
                "impact": "high",
                "current_complexity": analysis.complexity_score,
                "target_complexity": 15
            })
        
        return optimizations