"""Performance monitoring for OpenSearch queries and system metrics."""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Represents a performance metric measurement."""

    timestamp: datetime
    metric_name: str
    value: float
    query_id: Optional[str] = None
    query_text: Optional[str] = None
    search_strategy: Optional[str] = None
    index_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceAlert:
    """Represents a performance alert."""

    alert_id: str
    timestamp: datetime
    severity: str  # low, medium, high, critical
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    query_context: Optional[Dict[str, Any]] = None


@dataclass
class PerformanceThreshold:
    """Represents a performance threshold configuration."""

    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison_operator: str  # gt, lt, gte, lte
    window_size: int = 10  # Number of measurements to consider
    alert_cooldown: int = 300  # Seconds between alerts


class PerformanceMonitor:
    """Monitor and track OpenSearch query performance metrics."""

    def __init__(self, max_metrics_history: int = 10000):
        """Initialize performance monitor.

        Args:
            max_metrics_history: Maximum number of metrics to keep in memory
        """
        self.max_metrics_history = max_metrics_history
        self.metrics_history: deque = deque(maxlen=max_metrics_history)
        self.thresholds: Dict[str, PerformanceThreshold] = {}
        self.alerts: List[PerformanceAlert] = []
        self.alert_callbacks: List[Callable[[PerformanceAlert], None]] = []
        self.last_alert_times: Dict[str, datetime] = {}
        self._lock = Lock()

        # Set up default thresholds
        self._setup_default_thresholds()

    def _setup_default_thresholds(self):
        """Set up default performance thresholds."""
        default_thresholds = [
            PerformanceThreshold(
                metric_name="query_duration_ms",
                warning_threshold=1000,
                critical_threshold=5000,
                comparison_operator="gt",
            ),
            PerformanceThreshold(
                metric_name="total_hits",
                warning_threshold=100000,
                critical_threshold=500000,
                comparison_operator="gt",
            ),
            PerformanceThreshold(
                metric_name="shard_failures",
                warning_threshold=1,
                critical_threshold=5,
                comparison_operator="gte",
            ),
            PerformanceThreshold(
                metric_name="memory_usage_mb",
                warning_threshold=1000,
                critical_threshold=2000,
                comparison_operator="gt",
            ),
            PerformanceThreshold(
                metric_name="cpu_usage_percent",
                warning_threshold=80,
                critical_threshold=95,
                comparison_operator="gt",
            ),
        ]

        for threshold in default_thresholds:
            self.thresholds[threshold.metric_name] = threshold

    def record_metric(
        self,
        metric_name: str,
        value: float,
        query_id: Optional[str] = None,
        query_text: Optional[str] = None,
        search_strategy: Optional[str] = None,
        index_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a performance metric.

        Args:
            metric_name: Name of the metric
            value: Metric value
            query_id: Optional query identifier
            query_text: Optional query text
            search_strategy: Optional search strategy
            index_name: Optional index name
            metadata: Optional additional metadata
        """
        with self._lock:
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_name=metric_name,
                value=value,
                query_id=query_id,
                query_text=query_text,
                search_strategy=search_strategy,
                index_name=index_name,
                metadata=metadata or {},
            )

            self.metrics_history.append(metric)

            # Check thresholds and generate alerts
            self._check_thresholds(metric)

            logger.debug(f"Recorded metric {metric_name}: {value}")

    def record_query_performance(
        self,
        query_result: Dict[str, Any],
        query_text: str,
        search_strategy: str,
        start_time: float,
        query_id: Optional[str] = None,
        index_name: Optional[str] = None,
    ) -> None:
        """Record performance metrics from a query result.

        Args:
            query_result: OpenSearch query result
            query_text: Query text
            search_strategy: Search strategy used
            start_time: Query start time (from time.time())
            query_id: Optional query identifier
            index_name: Optional index name
        """
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        # Extract metrics from query result
        took_ms = query_result.get("took", 0)
        total_hits = query_result.get("hits", {}).get("total", {}).get("value", 0)
        max_score = query_result.get("hits", {}).get("max_score", 0)
        shard_info = query_result.get("_shards", {})

        # Record basic performance metrics
        base_metadata = {
            "opensearch_took_ms": took_ms,
            "total_shards": shard_info.get("total", 0),
            "successful_shards": shard_info.get("successful", 0),
            "max_score": max_score,
        }

        metrics_to_record = [
            ("query_duration_ms", duration_ms),
            ("opensearch_took_ms", took_ms),
            ("total_hits", total_hits),
            ("shard_failures", shard_info.get("failed", 0)),
            ("shard_skipped", shard_info.get("skipped", 0)),
        ]

        for metric_name, value in metrics_to_record:
            self.record_metric(
                metric_name=metric_name,
                value=value,
                query_id=query_id,
                query_text=query_text,
                search_strategy=search_strategy,
                index_name=index_name,
                metadata=base_metadata,
            )

        # Record profile metrics if available
        if "profile" in query_result:
            self._record_profile_metrics(
                query_result["profile"],
                query_id,
                query_text,
                search_strategy,
                index_name,
                base_metadata,
            )

    def _record_profile_metrics(
        self,
        profile_data: Dict[str, Any],
        query_id: Optional[str],
        query_text: str,
        search_strategy: str,
        index_name: Optional[str],
        base_metadata: Dict[str, Any],
    ) -> None:
        """Record metrics from OpenSearch profile data.

        Args:
            profile_data: Profile data from OpenSearch
            query_id: Query identifier
            query_text: Query text
            search_strategy: Search strategy
            index_name: Index name
            base_metadata: Base metadata
        """
        shards = profile_data.get("shards", [])

        for shard in shards:
            shard_id = shard.get("id", "unknown")
            searches = shard.get("searches", [])

            for search in searches:
                query_info = search.get("query", [])

                # Record query component timings
                for component in query_info:
                    component_type = component.get("type", "unknown")
                    time_nanos = component.get("time_in_nanos", 0)
                    time_ms = time_nanos / 1_000_000

                    metadata = {
                        **base_metadata,
                        "shard_id": shard_id,
                        "component_type": component_type,
                        "time_nanos": time_nanos,
                    }

                    self.record_metric(
                        metric_name=f"query_component_{component_type}_ms",
                        value=time_ms,
                        query_id=query_id,
                        query_text=query_text,
                        search_strategy=search_strategy,
                        index_name=index_name,
                        metadata=metadata,
                    )

                # Record collector metrics
                collector_info = search.get("collector", [])
                for collector in collector_info:
                    collector_name = collector.get("name", "unknown")
                    collector_time_nanos = collector.get("time_in_nanos", 0)
                    collector_time_ms = collector_time_nanos / 1_000_000

                    metadata = {
                        **base_metadata,
                        "shard_id": shard_id,
                        "collector_name": collector_name,
                        "time_nanos": collector_time_nanos,
                    }

                    self.record_metric(
                        metric_name=f"collector_{collector_name}_ms",
                        value=collector_time_ms,
                        query_id=query_id,
                        query_text=query_text,
                        search_strategy=search_strategy,
                        index_name=index_name,
                        metadata=metadata,
                    )

    def get_metrics_summary(
        self,
        metric_name: Optional[str] = None,
        time_window: Optional[timedelta] = None,
        search_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get summary statistics for metrics.

        Args:
            metric_name: Optional metric name filter
            time_window: Optional time window filter
            search_strategy: Optional search strategy filter

        Returns:
            Summary statistics
        """
        with self._lock:
            # Filter metrics
            filtered_metrics = list(self.metrics_history)

            if metric_name:
                filtered_metrics = [
                    m for m in filtered_metrics if m.metric_name == metric_name
                ]

            if time_window:
                cutoff_time = datetime.now() - time_window
                filtered_metrics = [
                    m for m in filtered_metrics if m.timestamp >= cutoff_time
                ]

            if search_strategy:
                filtered_metrics = [
                    m for m in filtered_metrics if m.search_strategy == search_strategy
                ]

            if not filtered_metrics:
                return {"message": "No metrics found matching criteria"}

            # Group metrics by name
            metrics_by_name = defaultdict(list)
            for metric in filtered_metrics:
                metrics_by_name[metric.metric_name].append(metric.value)

            # Calculate statistics
            summary = {
                "total_metrics": len(filtered_metrics),
                "time_range": {
                    "start": min(m.timestamp for m in filtered_metrics).isoformat(),
                    "end": max(m.timestamp for m in filtered_metrics).isoformat(),
                },
                "metrics": {},
            }

            for name, values in metrics_by_name.items():
                if values:
                    summary["metrics"][name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                        "percentiles": {
                            "p50": statistics.median(values),
                            "p90": self._percentile(values, 0.9),
                            "p95": self._percentile(values, 0.95),
                            "p99": self._percentile(values, 0.99),
                        },
                    }

            return summary

    def get_performance_trends(
        self,
        metric_name: str,
        time_window: timedelta = timedelta(hours=1),
        bucket_size: timedelta = timedelta(minutes=5),
    ) -> Dict[str, Any]:
        """Get performance trends over time.

        Args:
            metric_name: Metric name to analyze
            time_window: Time window to analyze
            bucket_size: Size of time buckets for aggregation

        Returns:
            Trend analysis
        """
        with self._lock:
            cutoff_time = datetime.now() - time_window

            # Filter metrics
            filtered_metrics = [
                m
                for m in self.metrics_history
                if m.metric_name == metric_name and m.timestamp >= cutoff_time
            ]

            if not filtered_metrics:
                return {"message": f"No metrics found for {metric_name} in time window"}

            # Create time buckets
            start_time = min(m.timestamp for m in filtered_metrics)
            end_time = max(m.timestamp for m in filtered_metrics)

            buckets = []
            current_time = start_time

            while current_time <= end_time:
                bucket_end = current_time + bucket_size
                bucket_metrics = [
                    m
                    for m in filtered_metrics
                    if current_time <= m.timestamp < bucket_end
                ]

                if bucket_metrics:
                    values = [m.value for m in bucket_metrics]
                    buckets.append(
                        {
                            "timestamp": current_time.isoformat(),
                            "count": len(values),
                            "min": min(values),
                            "max": max(values),
                            "mean": statistics.mean(values),
                            "median": statistics.median(values),
                        }
                    )

                current_time = bucket_end

            # Calculate trend
            if len(buckets) >= 2:
                recent_avg = statistics.mean(
                    [b["mean"] for b in buckets[-3:]]
                )  # Last 3 buckets
                earlier_avg = statistics.mean(
                    [b["mean"] for b in buckets[:3]]
                )  # First 3 buckets
                trend_direction = (
                    "improving" if recent_avg < earlier_avg else "degrading"
                )
                trend_magnitude = (
                    abs(recent_avg - earlier_avg) / earlier_avg
                    if earlier_avg > 0
                    else 0
                )
            else:
                trend_direction = "stable"
                trend_magnitude = 0

            return {
                "metric_name": metric_name,
                "time_window": str(time_window),
                "bucket_size": str(bucket_size),
                "buckets": buckets,
                "trend": {"direction": trend_direction, "magnitude": trend_magnitude},
                "summary": {
                    "total_measurements": len(filtered_metrics),
                    "time_buckets": len(buckets),
                },
            }

    def set_threshold(
        self,
        metric_name: str,
        warning_threshold: float,
        critical_threshold: float,
        comparison_operator: str = "gt",
        window_size: int = 10,
        alert_cooldown: int = 300,
    ) -> None:
        """Set performance threshold for a metric.

        Args:
            metric_name: Name of the metric
            warning_threshold: Warning threshold value
            critical_threshold: Critical threshold value
            comparison_operator: Comparison operator (gt, lt, gte, lte)
            window_size: Number of measurements to consider
            alert_cooldown: Seconds between alerts
        """
        threshold = PerformanceThreshold(
            metric_name=metric_name,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            comparison_operator=comparison_operator,
            window_size=window_size,
            alert_cooldown=alert_cooldown,
        )

        self.thresholds[metric_name] = threshold
        logger.info(
            f"Set threshold for {metric_name}: warning={warning_threshold}, critical={critical_threshold}"
        )

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]) -> None:
        """Add callback function to be called when alerts are generated.

        Args:
            callback: Function to call with PerformanceAlert
        """
        self.alert_callbacks.append(callback)

    def _check_thresholds(self, metric: PerformanceMetric) -> None:
        """Check if metric violates any thresholds.

        Args:
            metric: Performance metric to check
        """
        threshold = self.thresholds.get(metric.metric_name)
        if not threshold:
            return

        # Check cooldown
        last_alert_time = self.last_alert_times.get(metric.metric_name)
        if last_alert_time:
            time_since_last = (datetime.now() - last_alert_time).total_seconds()
            if time_since_last < threshold.alert_cooldown:
                return

        # Get recent metrics for window analysis
        recent_metrics = [
            m
            for m in list(self.metrics_history)[-threshold.window_size :]
            if m.metric_name == metric.metric_name
        ]

        if len(recent_metrics) < threshold.window_size:
            return  # Not enough data

        # Calculate average value in window
        avg_value = statistics.mean([m.value for m in recent_metrics])

        # Check thresholds
        severity = None
        threshold_value = None

        if self._compare_value(
            avg_value, threshold.critical_threshold, threshold.comparison_operator
        ):
            severity = "critical"
            threshold_value = threshold.critical_threshold
        elif self._compare_value(
            avg_value, threshold.warning_threshold, threshold.comparison_operator
        ):
            severity = "warning"
            threshold_value = threshold.warning_threshold

        if severity:
            alert = PerformanceAlert(
                alert_id=f"{metric.metric_name}_{int(time.time())}",
                timestamp=datetime.now(),
                severity=severity,
                metric_name=metric.metric_name,
                current_value=avg_value,
                threshold_value=threshold_value,
                message=f"{metric.metric_name} {severity} threshold exceeded: {avg_value:.2f} {threshold.comparison_operator} {threshold_value}",
                query_context={
                    "query_id": metric.query_id,
                    "query_text": metric.query_text,
                    "search_strategy": metric.search_strategy,
                    "index_name": metric.index_name,
                }
                if metric.query_id
                else None,
            )

            self.alerts.append(alert)
            self.last_alert_times[metric.metric_name] = datetime.now()

            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error calling alert callback: {e}")

            logger.warning(f"Performance alert: {alert.message}")

    def _compare_value(self, value: float, threshold: float, operator: str) -> bool:
        """Compare value against threshold using operator.

        Args:
            value: Value to compare
            threshold: Threshold value
            operator: Comparison operator

        Returns:
            True if condition is met
        """
        if operator == "gt":
            return value > threshold
        elif operator == "gte":
            return value >= threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "lte":
            return value <= threshold
        else:
            return False

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values.

        Args:
            values: List of values
            percentile: Percentile (0.0 to 1.0)

        Returns:
            Percentile value
        """
        sorted_values = sorted(values)
        index = int(percentile * (len(sorted_values) - 1))
        return sorted_values[index]

    def get_recent_alerts(
        self, severity: Optional[str] = None, time_window: Optional[timedelta] = None
    ) -> List[PerformanceAlert]:
        """Get recent performance alerts.

        Args:
            severity: Optional severity filter
            time_window: Optional time window filter

        Returns:
            List of alerts
        """
        alerts = self.alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if time_window:
            cutoff_time = datetime.now() - time_window
            alerts = [a for a in alerts if a.timestamp >= cutoff_time]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def export_metrics(
        self,
        output_path: str,
        time_window: Optional[timedelta] = None,
        format: str = "json",
    ) -> None:
        """Export metrics to file.

        Args:
            output_path: Path to save metrics
            time_window: Optional time window filter
            format: Export format (json, csv)
        """
        with self._lock:
            metrics = list(self.metrics_history)

            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]

            if format == "json":
                data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "metrics_count": len(metrics),
                    "metrics": [asdict(m) for m in metrics],
                }

                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)

            elif format == "csv":
                import csv

                with open(output_path, "w", newline="") as f:
                    if metrics:
                        fieldnames = list(asdict(metrics[0]).keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()

                        for metric in metrics:
                            writer.writerow(asdict(metric))

            logger.info(f"Exported {len(metrics)} metrics to {output_path}")

    def clear_old_metrics(self, older_than: timedelta) -> int:
        """Clear metrics older than specified time.

        Args:
            older_than: Time threshold

        Returns:
            Number of metrics cleared
        """
        with self._lock:
            cutoff_time = datetime.now() - older_than
            original_count = len(self.metrics_history)

            # Filter out old metrics
            self.metrics_history = deque(
                [m for m in self.metrics_history if m.timestamp >= cutoff_time],
                maxlen=self.max_metrics_history,
            )

            cleared_count = original_count - len(self.metrics_history)

            if cleared_count > 0:
                logger.info(f"Cleared {cleared_count} old metrics")

            return cleared_count
