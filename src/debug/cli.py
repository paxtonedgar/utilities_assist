#!/usr/bin/env python3
"""Command-line interface for OpenSearch debugging tools."""

import argparse
import json
import sys
import time
from datetime import timedelta

from .debug_client import OpenSearchDebugClient
from .query_analyzer import QueryAnalyzer
from .performance_monitor import PerformanceMonitor
from ..infra.opensearch_client import OpenSearchClient


def create_debug_client() -> OpenSearchDebugClient:
    """Create and return a debug client instance."""
    # Initialize OpenSearch client
    opensearch_client = OpenSearchClient()
    return OpenSearchDebugClient(opensearch_client)


def explain_query_command(args) -> None:
    """Handle explain query command."""
    debug_client = create_debug_client()

    try:
        result = debug_client.explain_query(
            query_text=args.query,
            index_name=args.index,
            search_strategy=args.strategy,
            doc_id=args.doc_id,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Explanation saved to {args.output}")
        else:
            print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(f"Error explaining query: {e}", file=sys.stderr)
        sys.exit(1)


def profile_query_command(args) -> None:
    """Handle profile query command."""
    debug_client = create_debug_client()

    try:
        result = debug_client.profile_query(
            query_text=args.query,
            index_name=args.index,
            search_strategy=args.strategy,
            include_explain=args.include_explain,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Profile saved to {args.output}")
        else:
            print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(f"Error profiling query: {e}", file=sys.stderr)
        sys.exit(1)


def debug_strategies_command(args) -> None:
    """Handle debug strategies command."""
    debug_client = create_debug_client()

    strategies = args.strategies.split(",")

    try:
        result = debug_client.debug_search_strategy(
            query_text=args.query,
            index_name=args.index,
            strategies=strategies,
            include_profile=args.include_profile,
            include_explain=args.include_explain,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Strategy comparison saved to {args.output}")
        else:
            print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(f"Error debugging strategies: {e}", file=sys.stderr)
        sys.exit(1)


def analyze_query_command(args) -> None:
    """Handle analyze query command."""
    analyzer = QueryAnalyzer()

    try:
        # Parse query if it's a JSON string
        if args.query.strip().startswith("{"):
            query = json.loads(args.query)
        else:
            # Simple text query - convert to basic match query
            query = {"query": {"match": {"_all": args.query}}}

        analysis = analyzer.analyze_query(query)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(analysis.__dict__, f, indent=2, default=str)
            print(f"Query analysis saved to {args.output}")
        else:
            print(f"Query Complexity: {analysis.complexity_score}")
            print(f"Performance Score: {analysis.performance_score}")
            print(f"\nComponents ({len(analysis.components)}):")
            for comp in analysis.components:
                print(f"  - {comp.type}: {comp.description} (weight: {comp.weight})")

            if analysis.optimization_suggestions:
                print("\nOptimization Suggestions:")
                for suggestion in analysis.optimization_suggestions:
                    print(f"  - {suggestion}")

            if analysis.field_usage:
                print("\nField Usage:")
                for field, count in analysis.field_usage.items():
                    print(f"  - {field}: {count}")

            if analysis.boost_analysis:
                print("\nBoost Analysis:")
                for field, boost in analysis.boost_analysis.items():
                    print(f"  - {field}: {boost}")

    except Exception as e:
        print(f"Error analyzing query: {e}", file=sys.stderr)
        sys.exit(1)


def monitor_performance_command(args) -> None:
    """Handle performance monitoring command."""
    monitor = PerformanceMonitor()
    debug_client = create_debug_client()

    print(f"Starting performance monitoring for {args.duration} seconds...")
    print(f"Query: {args.query}")
    print(f"Strategy: {args.strategy}")
    print(f"Interval: {args.interval} seconds")
    print("Press Ctrl+C to stop\n")

    try:
        start_time = time.time()
        query_count = 0

        while time.time() - start_time < args.duration:
            query_start = time.time()

            try:
                # Execute query with profiling
                result = debug_client.profile_query(
                    query_text=args.query,
                    index_name=args.index,
                    search_strategy=args.strategy,
                    include_explain=False,
                )

                # Record performance metrics
                monitor.record_query_performance(
                    query_result=result.get("search_result", {}),
                    query_text=args.query,
                    search_strategy=args.strategy,
                    start_time=query_start,
                    query_id=f"monitor_{query_count}",
                    index_name=args.index,
                )

                query_count += 1

                if args.verbose:
                    took_ms = result.get("search_result", {}).get("took", 0)
                    hits = (
                        result.get("search_result", {})
                        .get("hits", {})
                        .get("total", {})
                        .get("value", 0)
                    )
                    print(f"Query {query_count}: {took_ms}ms, {hits} hits")

            except Exception as e:
                print(f"Query {query_count} failed: {e}")

            time.sleep(args.interval)

        # Print summary
        print(f"\nMonitoring completed. Executed {query_count} queries.")

        summary = monitor.get_metrics_summary(
            time_window=timedelta(seconds=args.duration + 10)
        )

        print("\nPerformance Summary:")
        print(json.dumps(summary, indent=2, default=str))

        # Save results if requested
        if args.output:
            monitor.export_metrics(args.output, format="json")
            print(f"\nDetailed metrics saved to {args.output}")

    except KeyboardInterrupt:
        print(f"\nMonitoring stopped. Executed {query_count} queries.")
    except Exception as e:
        print(f"Error during monitoring: {e}", file=sys.stderr)
        sys.exit(1)


def analyze_slow_queries_command(args) -> None:
    """Handle slow query analysis command."""
    debug_client = create_debug_client()

    try:
        result = debug_client.analyze_slow_queries(
            index_name=args.index,
            threshold_ms=args.threshold,
            time_window_hours=args.time_window,
        )

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Slow query analysis saved to {args.output}")
        else:
            print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(f"Error analyzing slow queries: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OpenSearch debugging and performance analysis tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explain why a document matches a query
  %(prog)s explain "machine learning" --doc-id doc123
  
  # Profile query performance
  %(prog)s profile "neural networks" --strategy enhanced_rrf
  
  # Compare multiple search strategies
  %(prog)s debug-strategies "AI research" --strategies "enhanced_rrf,bm25"
  
  # Analyze query structure
  %(prog)s analyze-query '{"query": {"match": {"title": "python"}}}'
  
  # Monitor performance in real-time
  %(prog)s monitor "data science" --duration 60 --interval 5
  
  # Analyze slow queries
  %(prog)s slow-queries --threshold 1000 --time-window 24
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Explain command
    explain_parser = subparsers.add_parser("explain", help="Explain query scoring")
    explain_parser.add_argument("query", help="Query text")
    explain_parser.add_argument(
        "--doc-id", required=True, help="Document ID to explain"
    )
    explain_parser.add_argument(
        "--index", default="utilities_assist", help="Index name"
    )
    explain_parser.add_argument(
        "--strategy",
        default="enhanced_rrf",
        choices=["enhanced_rrf", "bm25", "knn"],
        help="Search strategy",
    )
    explain_parser.add_argument("--output", help="Output file path")
    explain_parser.set_defaults(func=explain_query_command)

    # Profile command
    profile_parser = subparsers.add_parser("profile", help="Profile query performance")
    profile_parser.add_argument("query", help="Query text")
    profile_parser.add_argument(
        "--index", default="utilities_assist", help="Index name"
    )
    profile_parser.add_argument(
        "--strategy",
        default="enhanced_rrf",
        choices=["enhanced_rrf", "bm25", "knn"],
        help="Search strategy",
    )
    profile_parser.add_argument(
        "--include-explain", action="store_true", help="Include explain information"
    )
    profile_parser.add_argument("--output", help="Output file path")
    profile_parser.set_defaults(func=profile_query_command)

    # Debug strategies command
    debug_parser = subparsers.add_parser(
        "debug-strategies", help="Compare search strategies"
    )
    debug_parser.add_argument("query", help="Query text")
    debug_parser.add_argument(
        "--strategies",
        default="enhanced_rrf,bm25",
        help="Comma-separated list of strategies",
    )
    debug_parser.add_argument("--index", default="utilities_assist", help="Index name")
    debug_parser.add_argument(
        "--include-profile", action="store_true", help="Include profiling information"
    )
    debug_parser.add_argument(
        "--include-explain", action="store_true", help="Include explain information"
    )
    debug_parser.add_argument("--output", help="Output file path")
    debug_parser.set_defaults(func=debug_strategies_command)

    # Analyze query command
    analyze_parser = subparsers.add_parser(
        "analyze-query", help="Analyze query structure"
    )
    analyze_parser.add_argument("query", help="Query text or JSON query")
    analyze_parser.add_argument("--output", help="Output file path")
    analyze_parser.set_defaults(func=analyze_query_command)

    # Monitor performance command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor query performance")
    monitor_parser.add_argument("query", help="Query text")
    monitor_parser.add_argument(
        "--duration", type=int, default=60, help="Monitoring duration in seconds"
    )
    monitor_parser.add_argument(
        "--interval", type=float, default=5.0, help="Query interval in seconds"
    )
    monitor_parser.add_argument(
        "--index", default="utilities_assist", help="Index name"
    )
    monitor_parser.add_argument(
        "--strategy",
        default="enhanced_rrf",
        choices=["enhanced_rrf", "bm25", "knn"],
        help="Search strategy",
    )
    monitor_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    monitor_parser.add_argument("--output", help="Output file for detailed metrics")
    monitor_parser.set_defaults(func=monitor_performance_command)

    # Slow queries command
    slow_parser = subparsers.add_parser("slow-queries", help="Analyze slow queries")
    slow_parser.add_argument(
        "--threshold",
        type=int,
        default=1000,
        help="Slow query threshold in milliseconds",
    )
    slow_parser.add_argument(
        "--time-window", type=int, default=24, help="Time window in hours"
    )
    slow_parser.add_argument("--index", default="utilities_assist", help="Index name")
    slow_parser.add_argument("--output", help="Output file path")
    slow_parser.set_defaults(func=analyze_slow_queries_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Call the appropriate command function
    args.func(args)


if __name__ == "__main__":
    main()
