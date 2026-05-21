"""Tests for observability backends, collectors, registry, and exporters."""

from __future__ import annotations

import io
import json
import time

import pytest

from afk.core.telemetry import TelemetryEvent, TelemetrySpan
from afk.observability.backends import registry as _reg
from afk.observability.backends.null import NullTelemetryBackend, NullTelemetrySink
from afk.observability.backends.registry import (
    TelemetryBackendError,
    create_telemetry_sink,
    get_telemetry_backend,
    list_telemetry_backends,
    register_telemetry_backend,
)
from afk.observability.collectors.runtime import RuntimeTelemetryCollector
from afk.observability.exporters.console import ConsoleRunMetricsExporter
from afk.observability.exporters.json import JSONRunMetricsExporter
from afk.observability.exporters.jsonl import JSONLRunMetricsExporter
from afk.observability.models import RunMetrics
from afk.observability.projectors.run_metrics import run_metrics_schema_version

# =======================================================================
# NullTelemetrySink
# =======================================================================


class TestNullTelemetrySink:
    def test_record_event_does_not_raise(self):
        sink = NullTelemetrySink()
        event = TelemetryEvent(
            name="test.event",
            timestamp_ms=int(time.time() * 1000),
            attributes={"key": "value"},
        )
        sink.record_event(event)  # should not raise

    def test_start_span_returns_none(self):
        sink = NullTelemetrySink()
        result = sink.start_span("my.span", attributes={"foo": "bar"})
        assert result is None

    def test_end_span_does_not_raise_with_none_span(self):
        sink = NullTelemetrySink()
        sink.end_span(None, status="ok")  # should not raise

    def test_end_span_does_not_raise_with_real_span(self):
        sink = NullTelemetrySink()
        span = TelemetrySpan(
            name="test.span",
            started_at_ms=int(time.time() * 1000),
        )
        sink.end_span(span, status="ok", error=None, attributes={"a": 1})

    def test_increment_counter_does_not_raise(self):
        sink = NullTelemetrySink()
        sink.increment_counter("my.counter", 5, attributes={"region": "us"})

    def test_record_histogram_does_not_raise(self):
        sink = NullTelemetrySink()
        sink.record_histogram("my.histogram", 42.5, attributes={"unit": "ms"})


# =======================================================================
# NullTelemetryBackend
# =======================================================================


class TestNullTelemetryBackend:
    def test_backend_id_is_null(self):
        backend = NullTelemetryBackend()
        assert backend.backend_id == "null"

    def test_create_sink_returns_null_telemetry_sink(self):
        backend = NullTelemetryBackend()
        sink = backend.create_sink()
        assert isinstance(sink, NullTelemetrySink)

    def test_create_sink_with_config(self):
        backend = NullTelemetryBackend()
        sink = backend.create_sink(config={"some_key": "some_value"})
        assert isinstance(sink, NullTelemetrySink)


# =======================================================================
# RuntimeTelemetryCollector
# =======================================================================


class TestRuntimeTelemetryCollectorInit:
    def test_constructor_initializes_empty_events(self):
        collector = RuntimeTelemetryCollector()
        assert collector.events() == []

    def test_constructor_initializes_empty_spans(self):
        collector = RuntimeTelemetryCollector()
        assert collector.spans() == []

    def test_constructor_initializes_empty_counters(self):
        collector = RuntimeTelemetryCollector()
        assert collector.counters() == []

    def test_constructor_initializes_empty_histograms(self):
        collector = RuntimeTelemetryCollector()
        assert collector.histograms() == []


class TestRuntimeTelemetryCollectorRecordEvent:
    def test_record_event_stores_event(self):
        collector = RuntimeTelemetryCollector()
        event = TelemetryEvent(
            name="test.event",
            timestamp_ms=1000,
            attributes={"key": "val"},
        )
        collector.record_event(event)
        events = collector.events()
        assert len(events) == 1
        assert events[0].name == "test.event"
        assert events[0].timestamp_ms == 1000
        assert events[0].attributes == {"key": "val"}

    def test_record_multiple_events(self):
        collector = RuntimeTelemetryCollector()
        for i in range(3):
            collector.record_event(
                TelemetryEvent(name=f"event.{i}", timestamp_ms=i * 100)
            )
        assert len(collector.events()) == 3


class TestRuntimeTelemetryCollectorStartSpan:
    def test_start_span_returns_telemetry_span(self):
        collector = RuntimeTelemetryCollector()
        span = collector.start_span("my.span")
        assert isinstance(span, TelemetrySpan)

    def test_start_span_has_correct_name(self):
        collector = RuntimeTelemetryCollector()
        span = collector.start_span("my.span")
        assert span.name == "my.span"

    def test_start_span_has_started_at_ms(self):
        collector = RuntimeTelemetryCollector()
        before_ms = int(time.time() * 1000)
        span = collector.start_span("my.span")
        after_ms = int(time.time() * 1000)
        assert before_ms <= span.started_at_ms <= after_ms

    def test_start_span_with_attributes(self):
        collector = RuntimeTelemetryCollector()
        span = collector.start_span("my.span", attributes={"agent": "test"})
        assert span.attributes == {"agent": "test"}

    def test_start_span_without_attributes(self):
        collector = RuntimeTelemetryCollector()
        span = collector.start_span("my.span")
        assert span.attributes == {}


class TestRuntimeTelemetryCollectorEndSpan:
    def test_end_span_records_span_with_duration(self):
        collector = RuntimeTelemetryCollector()
        span = collector.start_span("my.span")
        collector.end_span(span, status="ok")
        spans = collector.spans()
        assert len(spans) == 1
        record = spans[0]
        assert record["name"] == "my.span"
        assert "duration_ms" in record
        assert record["duration_ms"] >= 0
        assert record["status"] == "ok"

    def test_end_span_records_error(self):
        collector = RuntimeTelemetryCollector()
        span = collector.start_span("my.span")
        collector.end_span(span, status="error", error="something broke")
        record = collector.spans()[0]
        assert record["error"] == "something broke"
        assert record["status"] == "error"

    def test_end_span_merges_attributes(self):
        collector = RuntimeTelemetryCollector()
        span = collector.start_span("my.span", attributes={"a": 1})
        collector.end_span(span, status="ok", attributes={"b": 2})
        record = collector.spans()[0]
        assert record["attributes"]["a"] == 1
        assert record["attributes"]["b"] == 2

    def test_end_span_with_none_span_is_noop(self):
        collector = RuntimeTelemetryCollector()
        collector.end_span(None, status="ok")
        assert collector.spans() == []


class TestRuntimeTelemetryCollectorCounters:
    def test_increment_counter_records_counter(self):
        collector = RuntimeTelemetryCollector()
        collector.increment_counter("my.counter", 5, attributes={"region": "us"})
        counters = collector.counters()
        assert len(counters) == 1
        assert counters[0]["name"] == "my.counter"
        assert counters[0]["value"] == 5
        assert counters[0]["attributes"] == {"region": "us"}

    def test_increment_counter_default_value(self):
        collector = RuntimeTelemetryCollector()
        collector.increment_counter("my.counter")
        assert collector.counters()[0]["value"] == 1


class TestRuntimeTelemetryCollectorHistograms:
    def test_record_histogram_stores_float_value(self):
        collector = RuntimeTelemetryCollector()
        collector.record_histogram("latency", 42.5, attributes={"unit": "ms"})
        histograms = collector.histograms()
        assert len(histograms) == 1
        assert histograms[0]["name"] == "latency"
        assert histograms[0]["value"] == 42.5
        assert isinstance(histograms[0]["value"], float)
        assert histograms[0]["attributes"] == {"unit": "ms"}

    def test_record_histogram_int_value_cast_to_float(self):
        collector = RuntimeTelemetryCollector()
        collector.record_histogram("latency", 100)
        assert isinstance(collector.histograms()[0]["value"], float)
        assert collector.histograms()[0]["value"] == 100.0


class TestRuntimeTelemetryCollectorAccessors:
    def test_events_returns_copy(self):
        collector = RuntimeTelemetryCollector()
        event = TelemetryEvent(name="e", timestamp_ms=0)
        collector.record_event(event)
        copy1 = collector.events()
        copy2 = collector.events()
        assert copy1 == copy2
        assert copy1 is not copy2

    def test_spans_returns_copy(self):
        collector = RuntimeTelemetryCollector()
        span = collector.start_span("s")
        collector.end_span(span, status="ok")
        copy1 = collector.spans()
        copy2 = collector.spans()
        assert copy1 == copy2
        assert copy1 is not copy2

    def test_counters_returns_copy(self):
        collector = RuntimeTelemetryCollector()
        collector.increment_counter("c")
        copy1 = collector.counters()
        copy2 = collector.counters()
        assert copy1 == copy2
        assert copy1 is not copy2

    def test_histograms_returns_copy(self):
        collector = RuntimeTelemetryCollector()
        collector.record_histogram("h", 1.0)
        copy1 = collector.histograms()
        copy2 = collector.histograms()
        assert copy1 == copy2
        assert copy1 is not copy2

    def test_started_at_returns_float_timestamp(self):
        before = time.time()
        collector = RuntimeTelemetryCollector()
        after = time.time()
        assert isinstance(collector.started_at(), float)
        assert before <= collector.started_at() <= after


class TestRuntimeTelemetryCollectorReset:
    def test_reset_clears_all_records(self):
        collector = RuntimeTelemetryCollector()
        collector.record_event(TelemetryEvent(name="e", timestamp_ms=0))
        span = collector.start_span("s")
        collector.end_span(span, status="ok")
        collector.increment_counter("c")
        collector.record_histogram("h", 1.0)

        collector.reset()

        assert collector.events() == []
        assert collector.spans() == []
        assert collector.counters() == []
        assert collector.histograms() == []

    def test_reset_updates_started_at(self):
        collector = RuntimeTelemetryCollector()
        original_started_at = collector.started_at()
        time.sleep(0.01)
        collector.reset()
        assert collector.started_at() >= original_started_at


# =======================================================================
# TelemetryBackendRegistry
# =======================================================================


@pytest.fixture(autouse=False)
def _clean_registry():
    """Ensure clean registry state before and after each registry test."""
    with _reg._LOCK:
        saved = dict(_reg._BACKENDS)
        _reg._BACKENDS.clear()
    yield
    with _reg._LOCK:
        _reg._BACKENDS.clear()
        _reg._BACKENDS.update(saved)


class TestRegisterTelemetryBackend:
    def test_register_by_backend_id(self, _clean_registry):
        backend = NullTelemetryBackend()
        register_telemetry_backend(backend)
        assert "null" in list_telemetry_backends()

    def test_register_with_empty_id_raises(self, _clean_registry):
        class EmptyIdBackend:
            backend_id = "   "

            def create_sink(self, *, config=None):
                return NullTelemetrySink()

        with pytest.raises(TelemetryBackendError):
            register_telemetry_backend(EmptyIdBackend())


class TestGetTelemetryBackend:
    def test_returns_registered_backend(self, _clean_registry):
        backend = NullTelemetryBackend()
        register_telemetry_backend(backend)
        resolved = get_telemetry_backend("null")
        assert resolved is backend

    def test_raises_for_unknown_backend(self, _clean_registry):
        with pytest.raises(TelemetryBackendError, match="Unknown telemetry backend"):
            get_telemetry_backend("nonexistent")


class TestListTelemetryBackends:
    def test_returns_sorted_ids(self, _clean_registry):
        class BackendA:
            backend_id = "beta"

            def create_sink(self, *, config=None):
                return NullTelemetrySink()

        class BackendB:
            backend_id = "alpha"

            def create_sink(self, *, config=None):
                return NullTelemetrySink()

        register_telemetry_backend(BackendA())
        register_telemetry_backend(BackendB())
        ids = list_telemetry_backends()
        assert ids == ["alpha", "beta"]

    def test_returns_empty_when_no_backends(self, _clean_registry):
        assert list_telemetry_backends() == []


class TestCreateTelemetrySink:
    def test_none_defaults_to_null(self, _clean_registry):
        register_telemetry_backend(NullTelemetryBackend())
        sink = create_telemetry_sink(None)
        assert isinstance(sink, NullTelemetrySink)

    def test_string_id_resolves(self, _clean_registry):
        register_telemetry_backend(NullTelemetryBackend())
        sink = create_telemetry_sink("null")
        assert isinstance(sink, NullTelemetrySink)

    def test_sink_instance_returned_directly(self, _clean_registry):
        my_sink = NullTelemetrySink()
        result = create_telemetry_sink(my_sink)
        assert result is my_sink


# =======================================================================
# ConsoleRunMetricsExporter
# =======================================================================


def _make_metrics(**overrides) -> RunMetrics:
    """Helper to build RunMetrics with sensible defaults for testing."""
    defaults = dict(
        run_id="abcdef1234567890abcdef",
        agent_name="test-agent",
        state="completed",
        total_duration_s=1.5,
        llm_calls=3,
        tool_calls=2,
        steps=4,
    )
    defaults.update(overrides)
    return RunMetrics(**defaults)


class TestConsoleRunMetricsExporter:
    def test_export_writes_status_duration_calls(self):
        buf = io.StringIO()
        exporter = ConsoleRunMetricsExporter(output=buf, color=False)
        metrics = _make_metrics()
        exporter.export(metrics)
        output = buf.getvalue()
        assert "SUCCESS" in output
        assert "1.50s" in output
        assert "3" in output  # llm_calls
        assert "2" in output  # tool_calls

    def test_color_true_applies_ansi_codes(self):
        buf = io.StringIO()
        exporter = ConsoleRunMetricsExporter(output=buf, color=True)
        metrics = _make_metrics()
        exporter.export(metrics)
        output = buf.getvalue()
        # ANSI escape code prefix
        assert "\033[" in output

    def test_color_false_no_ansi_codes(self):
        buf = io.StringIO()
        exporter = ConsoleRunMetricsExporter(output=buf, color=False)
        metrics = _make_metrics()
        exporter.export(metrics)
        output = buf.getvalue()
        assert "\033[" not in output

    def test_errors_section_printed(self):
        buf = io.StringIO()
        exporter = ConsoleRunMetricsExporter(output=buf, color=False)
        metrics = _make_metrics(
            state="failed",
            errors=["something went wrong", "another error"],
        )
        exporter.export(metrics)
        output = buf.getvalue()
        assert "FAILED" in output
        assert "Errors" in output
        assert "something went wrong" in output
        assert "another error" in output

    def test_agent_name_in_output(self):
        buf = io.StringIO()
        exporter = ConsoleRunMetricsExporter(output=buf, color=False)
        metrics = _make_metrics(agent_name="my-agent")
        exporter.export(metrics)
        output = buf.getvalue()
        assert "my-agent" in output

    def test_cost_in_output(self):
        buf = io.StringIO()
        exporter = ConsoleRunMetricsExporter(output=buf, color=False)
        metrics = _make_metrics(estimated_cost_usd=0.0123)
        exporter.export(metrics)
        output = buf.getvalue()
        assert "$0.0123" in output


# =======================================================================
# JSONRunMetricsExporter
# =======================================================================


class TestJSONRunMetricsExporter:
    def test_export_to_file(self, tmp_path):
        out_file = tmp_path / "metrics.json"
        exporter = JSONRunMetricsExporter(path=out_file, indent=2)
        metrics = _make_metrics()
        exporter.export(metrics)

        assert out_file.exists()
        data = json.loads(out_file.read_text("utf-8"))
        assert data["schema_version"] == "run_metrics.v1"
        assert "metrics" in data
        assert data["metrics"]["agent_name"] == "test-agent"

    def test_export_without_path_prints_to_stdout(self, monkeypatch, capsys):
        exporter = JSONRunMetricsExporter(path=None)
        metrics = _make_metrics()
        exporter.export(metrics)

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["schema_version"] == "run_metrics.v1"
        assert "metrics" in data

    def test_last_output_property(self):
        exporter = JSONRunMetricsExporter(path=None)
        assert exporter.last_output is None
        metrics = _make_metrics()
        exporter.export(metrics)
        assert exporter.last_output is not None
        data = json.loads(exporter.last_output)
        assert data["schema_version"] == "run_metrics.v1"

    def test_indent_parameter(self, tmp_path):
        out_file = tmp_path / "metrics.json"
        exporter_compact = JSONRunMetricsExporter(path=out_file, indent=0)
        exporter_compact.export(_make_metrics())
        compact_text = out_file.read_text("utf-8")

        out_file2 = tmp_path / "metrics2.json"
        exporter_pretty = JSONRunMetricsExporter(path=out_file2, indent=4)
        exporter_pretty.export(_make_metrics())
        pretty_text = out_file2.read_text("utf-8")

        # Pretty-printed JSON should be longer due to extra whitespace
        assert len(pretty_text) > len(compact_text)

    def test_reported_at_is_present(self, tmp_path):
        out_file = tmp_path / "metrics.json"
        exporter = JSONRunMetricsExporter(path=out_file)
        exporter.export(_make_metrics())
        data = json.loads(out_file.read_text("utf-8"))
        assert "reported_at" in data
        assert isinstance(data["reported_at"], float)


# =======================================================================
# JSONLRunMetricsExporter
# =======================================================================


class TestJSONLRunMetricsExporter:
    def test_export_appends_jsonl_line(self, tmp_path):
        out_file = tmp_path / "metrics.jsonl"
        exporter = JSONLRunMetricsExporter(path=out_file)
        exporter.export(_make_metrics())

        lines = out_file.read_text("utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["schema_version"] == "run_metrics.v1"
        assert "metrics" in data

    def test_multiple_exports_append_multiple_lines(self, tmp_path):
        out_file = tmp_path / "metrics.jsonl"
        exporter = JSONLRunMetricsExporter(path=out_file)
        exporter.export(_make_metrics(agent_name="agent-1"))
        exporter.export(_make_metrics(agent_name="agent-2"))
        exporter.export(_make_metrics(agent_name="agent-3"))

        lines = out_file.read_text("utf-8").strip().split("\n")
        assert len(lines) == 3
        agents = [json.loads(line)["metrics"]["agent_name"] for line in lines]
        assert agents == ["agent-1", "agent-2", "agent-3"]

    def test_read_all_returns_all_records(self, tmp_path):
        out_file = tmp_path / "metrics.jsonl"
        exporter = JSONLRunMetricsExporter(path=out_file)
        exporter.export(_make_metrics(agent_name="a"))
        exporter.export(_make_metrics(agent_name="b"))

        records = exporter.read_all()
        assert len(records) == 2
        assert records[0]["metrics"]["agent_name"] == "a"
        assert records[1]["metrics"]["agent_name"] == "b"

    def test_read_all_returns_empty_list_when_file_does_not_exist(self, tmp_path):
        out_file = tmp_path / "nonexistent.jsonl"
        exporter = JSONLRunMetricsExporter(path=out_file)
        assert exporter.read_all() == []

    def test_each_line_has_schema_version(self, tmp_path):
        out_file = tmp_path / "metrics.jsonl"
        exporter = JSONLRunMetricsExporter(path=out_file)
        exporter.export(_make_metrics())
        exporter.export(_make_metrics())

        records = exporter.read_all()
        for record in records:
            assert record["schema_version"] == "run_metrics.v1"


# =======================================================================
# run_metrics_schema_version
# =======================================================================


class TestRunMetricsSchemaVersion:
    def test_returns_run_metrics_v1(self):
        assert run_metrics_schema_version() == "run_metrics.v1"

    def test_returns_string(self):
        assert isinstance(run_metrics_schema_version(), str)
