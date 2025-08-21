#!/usr/bin/env python3
"""Example usage of OpenSearch debugging tools."""

import time
from datetime import timedelta

from .debug_client import OpenSearchDebugClient
from .query_analyzer import QueryAnalyzer
from .performance_monitor import PerformanceMonitor, PerformanceAlert
from ..infra.opensearch_client import OpenSearchClient


def setup_debug_environment():
    """Set up debugging environment with sample data."""
    print("Setting up OpenSearch debugging environment...")

    # Initialize clients
    opensearch_client = OpenSearchClient()
    debug_client = OpenSearchDebugClient(opensearch_client)
    analyzer = QueryAnalyzer()
    monitor = PerformanceMonitor()

    return debug_client, analyzer, monitor


def example_explain_query():
    """Demonstrate query explanation functionality."""
    print("\n=== Query Explanation Example ===")

    debug_client, _, _ = setup_debug_environment()

    # Example query
    query_text = "machine learning algorithms"
    doc_id = "example_doc_1"  # Replace with actual document ID

    try:
        print(f"Explaining why document '{doc_id}' matches query: '{query_text}'")

        result = debug_client.explain_query(
            query_text=query_text,
            index_name="utilities_assist",
            search_strategy="enhanced_rrf",
            doc_id=doc_id,
        )

        print("\nExplanation Summary:")
        print(f"Document matches: {result['explanation']['match']}")
        print(f"Score: {result['explanation']['value']}")
        print(f"Description: {result['explanation']['description']}")

        # Show detailed breakdown
        if "details" in result["explanation"]:
            print("\nDetailed Breakdown:")
            for detail in result["explanation"]["details"]:
                print(f"  - {detail['description']}: {detail['value']}")

        return result

    except Exception as e:
        print(f"Error explaining query: {e}")
        return None


def example_profile_query():
    """Demonstrate query profiling functionality."""
    print("\n=== Query Profiling Example ===")

    debug_client, _, _ = setup_debug_environment()

    # Example query
    query_text = "neural networks deep learning"

    try:
        print(f"Profiling query: '{query_text}'")

        result = debug_client.profile_query(
            query_text=query_text,
            index_name="utilities_assist",
            search_strategy="enhanced_rrf",
            include_explain=True,
        )

        # Extract performance metrics
        search_result = result["search_result"]
        profile_data = result["profile_analysis"]

        print("\nQuery Performance:")
        print(f"Total time: {search_result.get('took', 0)}ms")
        print(
            f"Total hits: {search_result.get('hits', {}).get('total', {}).get('value', 0)}"
        )
        print(f"Max score: {search_result.get('hits', {}).get('max_score', 0)}")

        print("\nProfile Summary:")
        print(f"Total shards: {profile_data['summary']['total_shards']}")
        print(
            f"Average query time: {profile_data['summary']['avg_query_time_ms']:.2f}ms"
        )
        print(f"Slowest component: {profile_data['summary']['slowest_component']}")

        if profile_data["bottlenecks"]:
            print("\nBottlenecks:")
            for bottleneck in profile_data["bottlenecks"]:
                print(f"  - {bottleneck}")

        if profile_data["optimization_suggestions"]:
            print("\nOptimization Suggestions:")
            for suggestion in profile_data["optimization_suggestions"]:
                print(f"  - {suggestion}")

        return result

    except Exception as e:
        print(f"Error profiling query: {e}")
        return None


def example_compare_strategies():
    """Demonstrate strategy comparison functionality."""
    print("\n=== Strategy Comparison Example ===")

    debug_client, _, _ = setup_debug_environment()

    # Example query
    query_text = "artificial intelligence research"
    strategies = ["enhanced_rrf", "bm25"]

    try:
        print(f"Comparing strategies {strategies} for query: '{query_text}'")

        result = debug_client.debug_search_strategy(
            query_text=query_text,
            index_name="utilities_assist",
            strategies=strategies,
            include_profile=True,
            include_explain=False,
        )

        print("\nStrategy Comparison Results:")

        for strategy, data in result["strategy_results"].items():
            search_result = data["search_result"]
            hits = search_result.get("hits", {}).get("total", {}).get("value", 0)
            took_ms = search_result.get("took", 0)
            max_score = search_result.get("hits", {}).get("max_score", 0)

            print(f"\n{strategy.upper()}:")
            print(f"  Hits: {hits}")
            print(f"  Time: {took_ms}ms")
            print(f"  Max Score: {max_score:.4f}")

            if "profile_analysis" in data:
                profile = data["profile_analysis"]
                print(
                    f"  Avg Query Time: {profile['summary']['avg_query_time_ms']:.2f}ms"
                )
                print(f"  Slowest Component: {profile['summary']['slowest_component']}")

        # Show comparison summary
        comparison = result["comparison"]
        print("\nComparison Summary:")
        print(
            f"Fastest strategy: {comparison['fastest_strategy']} ({comparison['performance_comparison']['fastest_time_ms']}ms)"
        )
        print(
            f"Most relevant: {comparison['most_relevant_strategy']} (score: {comparison['relevance_comparison']['highest_max_score']:.4f})"
        )
        print(
            f"Most hits: {comparison['most_comprehensive_strategy']} ({comparison['relevance_comparison']['most_hits']} hits)"
        )

        return result

    except Exception as e:
        print(f"Error comparing strategies: {e}")
        return None


def example_query_analysis():
    """Demonstrate query analysis functionality."""
    print("\n=== Query Analysis Example ===")

    _, analyzer, _ = setup_debug_environment()

    # Example complex query
    complex_query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": "machine learning",
                            "fields": ["title^4", "content^2", "tags^3"],
                            "type": "best_fields",
                        }
                    }
                ],
                "should": [
                    {"term": {"category": "AI"}},
                    {"range": {"publish_date": {"gte": "2020-01-01"}}},
                ],
                "filter": [{"term": {"status": "published"}}],
            }
        },
        "sort": [{"_score": {"order": "desc"}}, {"publish_date": {"order": "desc"}}],
        "size": 20,
    }

    try:
        print("Analyzing complex query structure...")

        analysis = analyzer.analyze_query(complex_query)

        print("\nQuery Analysis Results:")
        print(f"Complexity Score: {analysis.complexity_score}/10")
        print(f"Performance Score: {analysis.performance_score}/10")
        print(f"Total Components: {len(analysis.components)}")

        print("\nQuery Components:")
        for component in analysis.components:
            print(
                f"  - {component.type}: {component.description} (weight: {component.weight})"
            )

        print("\nField Usage:")
        for field, count in analysis.field_usage.items():
            print(f"  - {field}: {count} occurrences")

        print("\nBoost Analysis:")
        for field, boost in analysis.boost_analysis.items():
            print(f"  - {field}: {boost}x boost")

        if analysis.optimization_suggestions:
            print("\nOptimization Suggestions:")
            for suggestion in analysis.optimization_suggestions:
                print(f"  - {suggestion}")

        # Compare with simpler query
        simple_query = {"query": {"match": {"content": "machine learning"}}}

        simple_analysis = analyzer.analyze_query(simple_query)

        print("\n=== Comparison with Simple Query ===")
        comparison = analyzer.compare_queries(complex_query, simple_query)

        print(f"Complexity difference: {comparison['complexity_difference']:.2f}")
        print(f"Performance difference: {comparison['performance_difference']:.2f}")
        print(f"Recommendation: {comparison['recommendation']}")

        return analysis

    except Exception as e:
        print(f"Error analyzing query: {e}")
        return None


def example_performance_monitoring():
    """Demonstrate performance monitoring functionality."""
    print("\n=== Performance Monitoring Example ===")

    debug_client, _, monitor = setup_debug_environment()

    # Set up alert callback
    def alert_handler(alert: PerformanceAlert):
        print(f"\n🚨 PERFORMANCE ALERT: {alert.message}")
        print(f"   Severity: {alert.severity}")
        print(f"   Current Value: {alert.current_value}")
        print(f"   Threshold: {alert.threshold_value}")

    monitor.add_alert_callback(alert_handler)

    # Set custom thresholds
    monitor.set_threshold(
        metric_name="query_duration_ms",
        warning_threshold=500,
        critical_threshold=2000,
        comparison_operator="gt",
    )

    # Simulate query execution with monitoring
    queries = [
        "machine learning",
        "deep learning neural networks",
        "artificial intelligence research",
        "natural language processing",
        "computer vision algorithms",
    ]

    try:
        print("Executing queries with performance monitoring...")

        for i, query_text in enumerate(queries):
            print(f"\nExecuting query {i + 1}: '{query_text}'")

            start_time = time.time()

            # Execute query
            result = debug_client.profile_query(
                query_text=query_text,
                index_name="utilities_assist",
                search_strategy="enhanced_rrf",
                include_explain=False,
            )

            # Record performance metrics
            monitor.record_query_performance(
                query_result=result["search_result"],
                query_text=query_text,
                search_strategy="enhanced_rrf",
                start_time=start_time,
                query_id=f"example_{i + 1}",
                index_name="utilities_assist",
            )

            # Add some artificial delay
            time.sleep(0.5)

        # Get performance summary
        print("\n=== Performance Summary ===")
        summary = monitor.get_metrics_summary(time_window=timedelta(minutes=5))

        print(f"Total metrics recorded: {summary['total_metrics']}")

        for metric_name, stats in summary["metrics"].items():
            print(f"\n{metric_name}:")
            print(f"  Count: {stats['count']}")
            print(f"  Average: {stats['mean']:.2f}")
            print(f"  Min: {stats['min']:.2f}")
            print(f"  Max: {stats['max']:.2f}")
            print(f"  P95: {stats['percentiles']['p95']:.2f}")

        # Get performance trends
        print("\n=== Performance Trends ===")
        trends = monitor.get_performance_trends(
            metric_name="query_duration_ms",
            time_window=timedelta(minutes=5),
            bucket_size=timedelta(seconds=30),
        )

        print(f"Trend direction: {trends['trend']['direction']}")
        print(f"Trend magnitude: {trends['trend']['magnitude']:.2f}")
        print(f"Time buckets: {len(trends['buckets'])}")

        # Show recent alerts
        recent_alerts = monitor.get_recent_alerts(time_window=timedelta(minutes=5))

        if recent_alerts:
            print(f"\n=== Recent Alerts ({len(recent_alerts)}) ===")
            for alert in recent_alerts[:3]:  # Show last 3 alerts
                print(f"  - {alert.timestamp.strftime('%H:%M:%S')}: {alert.message}")
        else:
            print("\n=== No Recent Alerts ===")

        return summary

    except Exception as e:
        print(f"Error during performance monitoring: {e}")
        return None


def example_slow_query_analysis():
    """Demonstrate slow query analysis functionality."""
    print("\n=== Slow Query Analysis Example ===")

    debug_client, _, _ = setup_debug_environment()

    try:
        print("Analyzing slow queries from the last 24 hours...")

        result = debug_client.analyze_slow_queries(
            index_name="utilities_assist", threshold_ms=1000, time_window_hours=24
        )

        print("\nSlow Query Analysis Results:")
        print(f"Analysis period: {result['analysis_period']}")
        print(f"Threshold: {result['threshold_ms']}ms")
        print(f"Total slow queries found: {result['total_slow_queries']}")

        if result["slow_queries"]:
            print("\nTop Slow Queries:")
            for i, query in enumerate(result["slow_queries"][:5], 1):
                print(f"  {i}. Query: {query['query_text'][:50]}...")
                print(f"     Time: {query['execution_time_ms']}ms")
                print(f"     Strategy: {query['search_strategy']}")
                print(f"     Hits: {query['total_hits']}")

        if result["patterns"]:
            print("\nCommon Patterns:")
            for pattern in result["patterns"]:
                print(f"  - {pattern}")

        if result["recommendations"]:
            print("\nRecommendations:")
            for recommendation in result["recommendations"]:
                print(f"  - {recommendation}")

        return result

    except Exception as e:
        print(f"Error analyzing slow queries: {e}")
        return None


def run_comprehensive_debug_example():
    """Run a comprehensive debugging example."""
    print("🔍 OpenSearch Debugging Tools - Comprehensive Example")
    print("=" * 60)

    try:
        # Run all examples
        explain_result = example_explain_query()
        profile_result = example_profile_query()
        comparison_result = example_compare_strategies()
        analysis_result = example_query_analysis()
        monitoring_result = example_performance_monitoring()
        slow_query_result = example_slow_query_analysis()

        print("\n" + "=" * 60)
        print("🎉 All debugging examples completed successfully!")
        print("\nSummary of capabilities demonstrated:")
        print("✅ Query explanation and scoring analysis")
        print("✅ Query performance profiling")
        print("✅ Search strategy comparison")
        print("✅ Query structure analysis and optimization")
        print("✅ Real-time performance monitoring")
        print("✅ Slow query detection and analysis")

        return {
            "explain": explain_result,
            "profile": profile_result,
            "comparison": comparison_result,
            "analysis": analysis_result,
            "monitoring": monitoring_result,
            "slow_queries": slow_query_result,
        }

    except Exception as e:
        print(f"\n❌ Error running comprehensive example: {e}")
        return None


if __name__ == "__main__":
    # Run the comprehensive example
    results = run_comprehensive_debug_example()

    if results:
        print("\n📊 Example completed. Check the output above for detailed results.")
        print("\n💡 To use these tools in your own code:")
        print(
            "   from src.debug import OpenSearchDebugClient, QueryAnalyzer, PerformanceMonitor"
        )
        print("\n🔧 To use the CLI tools:")
        print("   python -m src.debug.cli explain 'your query' --doc-id doc123")
        print("   python -m src.debug.cli profile 'your query' --strategy enhanced_rrf")
        print("   python -m src.debug.cli monitor 'your query' --duration 60")
    else:
        print("\n❌ Example failed. Please check your OpenSearch configuration.")
