"""Tests for observability projectors and runtime collector integration."""

from __future__ import annotations

import time

import pytest

from afk.core.telemetry import TelemetryEvent
from afk.observability import contracts
from afk.observability.collectors.runtime import RuntimeTelemetryCollector
from afk.observability.projectors.run_metrics import (
    _counter_total,
    _to_float,
    _to_int,
    _to_str,
    project_run_metrics_from_collector,
    run_metrics_schema_version,
)

# ---------------------------------------------------------------------------
# run_metrics_schema_version
# ---------------------------------------------------------------------------


class TestRunMetricsSchemaVersion:
    """Schema version identifier is stable and well-typed."""

    def test_returns_string(self):
        assert isinstance(run_metrics_schema_version(), str)

    def test_value_is_run_metrics_v1(self):
        assert run_metrics_schema_version() == "run_metrics.v1"


# ---------------------------------------------------------------------------
# project_run_metrics_from_collector -- empty collector
# ---------------------------------------------------------------------------


class TestProjectRunMetricsEmpty:
    """An empty collector produces safe zero/empty defaults."""

    def test_empty_collector_defaults(self):
        collector = RuntimeTelemetryCollector()
        metrics = project_run_metrics_from_collector(collector)

        assert metrics.llm_calls == 0
        assert metrics.tool_calls == 0
        assert metrics.errors == []
        assert metrics.llm_latencies_ms == []
        assert metrics.tool_latencies_ms == {}
        assert metrics.run_id == ""
        assert metrics.state == "unknown"
        assert metrics.total_duration_s >= 0.0


# ---------------------------------------------------------------------------
# project_run_metrics_from_collector -- counter data
# ---------------------------------------------------------------------------


class TestProjectRunMetricsCounters:
    """Counter increments are aggregated into llm_calls / tool_calls."""

    def test_llm_calls_counted(self):
        collector = RuntimeTelemetryCollector()
        collector.increment_counter(contracts.METRIC_AGENT_LLM_CALLS_TOTAL, 1)
        collector.increment_counter(contracts.METRIC_AGENT_LLM_CALLS_TOTAL, 1)

        metrics = project_run_metrics_from_collector(collector)
        assert metrics.llm_calls == 2

    def test_tool_calls_counted(self):
        collector = RuntimeTelemetryCollector()
        collector.increment_counter(contracts.METRIC_AGENT_TOOL_CALLS_TOTAL, 1)
        collector.increment_counter(contracts.METRIC_AGENT_TOOL_CALLS_TOTAL, 1)
        collector.increment_counter(contracts.METRIC_AGENT_TOOL_CALLS_TOTAL, 1)

        metrics = project_run_metrics_from_collector(collector)
        assert metrics.tool_calls == 3

    def test_mixed_counters(self):
        collector = RuntimeTelemetryCollector()
        collector.increment_counter(contracts.METRIC_AGENT_LLM_CALLS_TOTAL, 1)
        collector.increment_counter(contracts.METRIC_AGENT_TOOL_CALLS_TOTAL, 2)

        metrics = project_run_metrics_from_collector(collector)
        assert metrics.llm_calls == 1
        assert metrics.tool_calls == 2

    def test_increment_by_larger_value(self):
        collector = RuntimeTelemetryCollector()
        collector.increment_counter(contracts.METRIC_AGENT_LLM_CALLS_TOTAL, 5)

        metrics = project_run_metrics_from_collector(collector)
        assert metrics.llm_calls == 5


# ---------------------------------------------------------------------------
# project_run_metrics_from_collector -- histogram data
# ---------------------------------------------------------------------------


class TestProjectRunMetricsHistograms:
    """Histogram recordings populate latency lists."""

    def test_llm_latencies_populated(self):
        collector = RuntimeTelemetryCollector()
        collector.record_histogram(contracts.METRIC_AGENT_LLM_LATENCY_MS, 150.0)
        collector.record_histogram(contracts.METRIC_AGENT_LLM_LATENCY_MS, 200.0)

        metrics = project_run_metrics_from_collector(collector)
        assert metrics.llm_latencies_ms == [150.0, 200.0]

    def test_tool_latencies_populated(self):
        collector = RuntimeTelemetryCollector()
        collector.record_histogram(
            contracts.METRIC_AGENT_TOOL_CALL_LATENCY_MS,
            50.0,
            attributes={"tool_name": "search"},
        )
        collector.record_histogram(
            contracts.METRIC_AGENT_TOOL_CALL_LATENCY_MS,
            75.0,
            attributes={"tool_name": "search"},
        )

        metrics = project_run_metrics_from_collector(collector)
        assert "search" in metrics.tool_latencies_ms
        assert metrics.tool_latencies_ms["search"] == [50.0, 75.0]

    def test_tool_latencies_without_tool_name_defaults_to_unknown(self):
        collector = RuntimeTelemetryCollector()
        collector.record_histogram(contracts.METRIC_AGENT_TOOL_CALL_LATENCY_MS, 30.0)

        metrics = project_run_metrics_from_collector(collector)
        assert "unknown" in metrics.tool_latencies_ms
        assert metrics.tool_latencies_ms["unknown"] == [30.0]


# ---------------------------------------------------------------------------
# project_run_metrics_from_collector -- agent.run span
# ---------------------------------------------------------------------------


class TestProjectRunMetricsRunSpan:
    """A completed agent.run span extracts run metadata."""

    def test_run_span_extracts_attributes(self):
        collector = RuntimeTelemetryCollector()
        span = collector.start_span(
            contracts.SPAN_AGENT_RUN,
            attributes={
                "run_id": "r1",
                "agent_name": "test_agent",
                "state": "completed",
                "steps": 3,
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "total_cost_usd": 0.01,
            },
        )
        collector.end_span(span, status="ok")

        metrics = project_run_metrics_from_collector(collector)

        assert metrics.run_id == "r1"
        assert metrics.agent_name == "test_agent"
        assert metrics.state == "completed"
        assert metrics.steps == 3
        assert metrics.input_tokens == 100
        assert metrics.output_tokens == 50
        assert metrics.total_tokens == 150
        assert metrics.estimated_cost_usd == pytest.approx(0.01)

    def test_run_span_duration_overrides_wall_clock(self):
        collector = RuntimeTelemetryCollector()
        span = collector.start_span(
            contracts.SPAN_AGENT_RUN,
            attributes={"run_id": "r2", "state": "completed"},
        )
        collector.end_span(span, status="ok")

        metrics = project_run_metrics_from_collector(collector)
        # duration_ms is computed from end-start, which should be >= 0
        assert metrics.total_duration_s >= 0.0


# ---------------------------------------------------------------------------
# project_run_metrics_from_collector -- error spans
# ---------------------------------------------------------------------------


class TestProjectRunMetricsErrors:
    """Error spans and failed run events populate the errors list."""

    def test_error_span_populates_errors(self):
        collector = RuntimeTelemetryCollector()
        span = collector.start_span("some.operation")
        collector.end_span(span, status="error", error="connection refused")

        metrics = project_run_metrics_from_collector(collector)
        assert "connection refused" in metrics.errors

    def test_run_failed_event_populates_errors(self):
        collector = RuntimeTelemetryCollector()
        collector.record_event(
            TelemetryEvent(
                name=contracts.AGENT_RUN_EVENT,
                timestamp_ms=1000,
                attributes={
                    "event_type": "run_failed",
                    "message": "something went wrong",
                },
            )
        )

        metrics = project_run_metrics_from_collector(collector)
        assert "something went wrong" in metrics.errors

    def test_non_failed_event_does_not_add_error(self):
        collector = RuntimeTelemetryCollector()
        collector.record_event(
            TelemetryEvent(
                name=contracts.AGENT_RUN_EVENT,
                timestamp_ms=1000,
                attributes={"event_type": "run_completed", "message": "done"},
            )
        )

        metrics = project_run_metrics_from_collector(collector)
        assert metrics.errors == []

    def test_error_span_without_message_not_added(self):
        collector = RuntimeTelemetryCollector()
        span = collector.start_span("op")
        collector.end_span(span, status="error", error=None)

        metrics = project_run_metrics_from_collector(collector)
        assert metrics.errors == []

    def test_both_error_sources_combined(self):
        collector = RuntimeTelemetryCollector()

        span = collector.start_span("failing.op")
        collector.end_span(span, status="error", error="span error")

        collector.record_event(
            TelemetryEvent(
                name=contracts.AGENT_RUN_EVENT,
                timestamp_ms=2000,
                attributes={"event_type": "run_failed", "message": "event error"},
            )
        )

        metrics = project_run_metrics_from_collector(collector)
        assert "span error" in metrics.errors
        assert "event error" in metrics.errors
        assert len(metrics.errors) == 2


# ---------------------------------------------------------------------------
# project_run_metrics_from_collector -- multiple run spans
# ---------------------------------------------------------------------------


class TestProjectRunMetricsMultipleSpans:
    """When multiple agent.run spans exist, the latest (by ended_at_ms) wins."""

    def test_latest_run_span_used(self):
        collector = RuntimeTelemetryCollector()

        # First span -- ended earlier
        span1 = collector.start_span(
            contracts.SPAN_AGENT_RUN,
            attributes={"run_id": "old_run", "state": "failed"},
        )
        collector.end_span(span1, status="error", error="first span fail")

        # Small delay to ensure ended_at_ms differs
        time.sleep(0.01)

        # Second span -- ended later, should be selected
        span2 = collector.start_span(
            contracts.SPAN_AGENT_RUN,
            attributes={"run_id": "new_run", "state": "completed"},
        )
        collector.end_span(span2, status="ok")

        metrics = project_run_metrics_from_collector(collector)
        assert metrics.run_id == "new_run"
        assert metrics.state == "completed"

    def test_non_run_spans_ignored_for_selection(self):
        collector = RuntimeTelemetryCollector()

        # Non-run span should be ignored for run attribute extraction
        other_span = collector.start_span(
            "agent.llm.call",
            attributes={"run_id": "wrong", "state": "wrong"},
        )
        collector.end_span(other_span, status="ok")

        metrics = project_run_metrics_from_collector(collector)
        # No agent.run span means run_id stays default
        assert metrics.run_id == ""


# ---------------------------------------------------------------------------
# _to_int helper
# ---------------------------------------------------------------------------


class TestToInt:
    """Conversion helper _to_int handles diverse input types."""

    def test_int_returns_int(self):
        assert _to_int(42) == 42

    def test_float_returns_int(self):
        assert _to_int(3.9) == 3

    def test_string_numeric_returns_int(self):
        assert _to_int("42") == 42

    def test_string_non_numeric_returns_default(self):
        assert _to_int("abc") == 0

    def test_string_non_numeric_custom_default(self):
        assert _to_int("abc", default=-1) == -1

    def test_bool_true_returns_1(self):
        assert _to_int(True) == 1

    def test_bool_false_returns_0(self):
        assert _to_int(False) == 0

    def test_none_returns_default(self):
        assert _to_int(None) == 0

    def test_none_custom_default(self):
        assert _to_int(None, default=99) == 99


# ---------------------------------------------------------------------------
# _to_float helper
# ---------------------------------------------------------------------------


class TestToFloat:
    """Conversion helper _to_float handles diverse input types."""

    def test_int_returns_float(self):
        result = _to_float(5)
        assert result == 5.0
        assert isinstance(result, float)

    def test_float_returns_float(self):
        result = _to_float(3.14)
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_string_numeric_returns_float(self):
        result = _to_float("3.14")
        assert result == pytest.approx(3.14)

    def test_string_non_numeric_returns_default_none(self):
        assert _to_float("abc") is None

    def test_string_non_numeric_custom_default(self):
        assert _to_float("abc", default=0.0) == 0.0

    def test_bool_true_returns_float(self):
        result = _to_float(True)
        assert result == 1.0
        assert isinstance(result, float)

    def test_bool_false_returns_float(self):
        result = _to_float(False)
        assert result == 0.0
        assert isinstance(result, float)

    def test_none_returns_default_none(self):
        assert _to_float(None) is None

    def test_none_custom_default(self):
        assert _to_float(None, default=1.5) == 1.5


# ---------------------------------------------------------------------------
# _to_str helper
# ---------------------------------------------------------------------------


class TestToStr:
    """Conversion helper _to_str handles diverse input types."""

    def test_string_returned_as_is(self):
        assert _to_str("hello") == "hello"

    def test_none_returns_default_empty(self):
        assert _to_str(None) == ""

    def test_none_custom_default(self):
        assert _to_str(None, default="N/A") == "N/A"

    def test_int_returns_str(self):
        assert _to_str(42) == "42"

    def test_float_returns_str(self):
        assert _to_str(3.14) == "3.14"

    def test_empty_string_returned(self):
        assert _to_str("") == ""


# ---------------------------------------------------------------------------
# _counter_total helper
# ---------------------------------------------------------------------------


class TestCounterTotal:
    """_counter_total sums matching counter rows by name."""

    def test_empty_rows_returns_zero(self):
        assert _counter_total([], "any.metric") == 0

    def test_matching_name_summed(self):
        rows = [
            {"name": "agent.llm.calls.total", "value": 1},
            {"name": "agent.llm.calls.total", "value": 2},
        ]
        assert _counter_total(rows, "agent.llm.calls.total") == 3

    def test_non_matching_names_ignored(self):
        rows = [
            {"name": "agent.llm.calls.total", "value": 1},
            {"name": "agent.tool.calls.total", "value": 5},
        ]
        assert _counter_total(rows, "agent.llm.calls.total") == 1

    def test_no_matching_rows_returns_zero(self):
        rows = [
            {"name": "agent.tool.calls.total", "value": 5},
        ]
        assert _counter_total(rows, "agent.llm.calls.total") == 0

    def test_mixed_rows(self):
        rows = [
            {"name": "agent.llm.calls.total", "value": 3},
            {"name": "agent.tool.calls.total", "value": 2},
            {"name": "agent.llm.calls.total", "value": 7},
            {"name": "other.metric", "value": 99},
        ]
        assert _counter_total(rows, "agent.llm.calls.total") == 10
        assert _counter_total(rows, "agent.tool.calls.total") == 2
        assert _counter_total(rows, "other.metric") == 99


# ---------------------------------------------------------------------------
# Integration -- full collector pipeline
# ---------------------------------------------------------------------------


class TestFullCollectorPipeline:
    """End-to-end: feed diverse data into collector and project once."""

    def test_full_pipeline(self):
        collector = RuntimeTelemetryCollector()

        # Counters
        collector.increment_counter(contracts.METRIC_AGENT_LLM_CALLS_TOTAL, 1)
        collector.increment_counter(contracts.METRIC_AGENT_LLM_CALLS_TOTAL, 1)
        collector.increment_counter(contracts.METRIC_AGENT_TOOL_CALLS_TOTAL, 1)

        # Histograms
        collector.record_histogram(contracts.METRIC_AGENT_LLM_LATENCY_MS, 150.0)
        collector.record_histogram(contracts.METRIC_AGENT_LLM_LATENCY_MS, 250.0)
        collector.record_histogram(
            contracts.METRIC_AGENT_TOOL_CALL_LATENCY_MS,
            80.0,
            attributes={"tool_name": "calculator"},
        )

        # Run span
        span = collector.start_span(
            contracts.SPAN_AGENT_RUN,
            attributes={
                "run_id": "integration_run",
                "agent_name": "integration_agent",
                "state": "completed",
                "steps": 5,
                "input_tokens": 500,
                "output_tokens": 200,
                "total_tokens": 700,
                "total_cost_usd": 0.05,
            },
        )
        collector.end_span(span, status="ok")

        # Run failed event
        collector.record_event(
            TelemetryEvent(
                name=contracts.AGENT_RUN_EVENT,
                timestamp_ms=int(time.time() * 1000),
                attributes={
                    "event_type": "run_failed",
                    "message": "partial failure noted",
                },
            )
        )

        metrics = project_run_metrics_from_collector(collector)

        assert metrics.llm_calls == 2
        assert metrics.tool_calls == 1
        assert metrics.llm_latencies_ms == [150.0, 250.0]
        assert metrics.tool_latencies_ms == {"calculator": [80.0]}
        assert metrics.run_id == "integration_run"
        assert metrics.agent_name == "integration_agent"
        assert metrics.state == "completed"
        assert metrics.steps == 5
        assert metrics.input_tokens == 500
        assert metrics.output_tokens == 200
        assert metrics.total_tokens == 700
        assert metrics.estimated_cost_usd == pytest.approx(0.05)
        assert "partial failure noted" in metrics.errors
        assert metrics.total_duration_s >= 0.0
