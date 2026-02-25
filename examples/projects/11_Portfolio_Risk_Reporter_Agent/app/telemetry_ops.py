"""Telemetry helpers for tier-3+ examples."""

from afk.observability import (
    RuntimeTelemetryCollector,
    project_run_metrics_from_collector,
    project_run_metrics_from_result,
)


def build_collector() -> RuntimeTelemetryCollector:
    """Create runtime telemetry collector sink."""
    return RuntimeTelemetryCollector()


def project_metrics(*, collector: RuntimeTelemetryCollector, result) -> tuple[dict, dict]:
    """Project metrics from collector and result into dict form."""
    collector_metrics = project_run_metrics_from_collector(collector).to_dict()
    result_metrics = project_run_metrics_from_result(result).to_dict()
    return collector_metrics, result_metrics
