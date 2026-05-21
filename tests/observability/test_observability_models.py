"""Tests for observability data models (RunMetrics)."""

from __future__ import annotations

from afk.observability.models import RunMetrics

# -----------------------------------------------------------------------
# RunMetrics default values
# -----------------------------------------------------------------------


class TestRunMetricsDefaults:
    def test_default_run_id(self):
        m = RunMetrics()
        assert m.run_id == ""

    def test_default_agent_name(self):
        m = RunMetrics()
        assert m.agent_name == ""

    def test_default_state(self):
        m = RunMetrics()
        assert m.state == "unknown"

    def test_default_total_duration_s(self):
        m = RunMetrics()
        assert m.total_duration_s == 0.0

    def test_default_llm_calls(self):
        m = RunMetrics()
        assert m.llm_calls == 0

    def test_default_tool_calls(self):
        m = RunMetrics()
        assert m.tool_calls == 0

    def test_default_input_tokens(self):
        m = RunMetrics()
        assert m.input_tokens == 0

    def test_default_output_tokens(self):
        m = RunMetrics()
        assert m.output_tokens == 0

    def test_default_total_tokens(self):
        m = RunMetrics()
        assert m.total_tokens == 0

    def test_default_estimated_cost_usd(self):
        m = RunMetrics()
        assert m.estimated_cost_usd is None

    def test_default_steps(self):
        m = RunMetrics()
        assert m.steps == 0

    def test_default_errors(self):
        m = RunMetrics()
        assert m.errors == []

    def test_default_tool_latencies_ms(self):
        m = RunMetrics()
        assert m.tool_latencies_ms == {}

    def test_default_llm_latencies_ms(self):
        m = RunMetrics()
        assert m.llm_latencies_ms == []


# -----------------------------------------------------------------------
# RunMetrics.success
# -----------------------------------------------------------------------


class TestRunMetricsSuccess:
    def test_success_true_when_completed(self):
        m = RunMetrics(state="completed")
        assert m.success is True

    def test_success_false_when_failed(self):
        m = RunMetrics(state="failed")
        assert m.success is False

    def test_success_false_when_unknown(self):
        m = RunMetrics(state="unknown")
        assert m.success is False

    def test_success_false_when_running(self):
        m = RunMetrics(state="running")
        assert m.success is False

    def test_success_false_when_empty_string(self):
        m = RunMetrics(state="")
        assert m.success is False


# -----------------------------------------------------------------------
# RunMetrics.avg_llm_latency_ms
# -----------------------------------------------------------------------


class TestRunMetricsAvgLlmLatency:
    def test_with_values(self):
        m = RunMetrics(llm_latencies_ms=[100.0, 200.0, 300.0])
        assert m.avg_llm_latency_ms == 200.0

    def test_single_value(self):
        m = RunMetrics(llm_latencies_ms=[42.5])
        assert m.avg_llm_latency_ms == 42.5

    def test_returns_none_when_empty(self):
        m = RunMetrics(llm_latencies_ms=[])
        assert m.avg_llm_latency_ms is None

    def test_returns_none_on_default(self):
        m = RunMetrics()
        assert m.avg_llm_latency_ms is None

    def test_floating_point_precision(self):
        m = RunMetrics(llm_latencies_ms=[10.0, 20.0, 30.0])
        result = m.avg_llm_latency_ms
        assert result is not None
        assert abs(result - 20.0) < 1e-9


# -----------------------------------------------------------------------
# RunMetrics.avg_tool_latency_ms
# -----------------------------------------------------------------------


class TestRunMetricsAvgToolLatency:
    def test_with_values_single_tool(self):
        m = RunMetrics(tool_latencies_ms={"search": [50.0, 150.0]})
        assert m.avg_tool_latency_ms == 100.0

    def test_with_values_multiple_tools(self):
        m = RunMetrics(
            tool_latencies_ms={
                "search": [100.0, 200.0],
                "fetch": [300.0, 400.0],
            }
        )
        # All values: [100, 200, 300, 400] -> mean = 250.0
        assert m.avg_tool_latency_ms == 250.0

    def test_returns_none_when_empty(self):
        m = RunMetrics(tool_latencies_ms={})
        assert m.avg_tool_latency_ms is None

    def test_returns_none_on_default(self):
        m = RunMetrics()
        assert m.avg_tool_latency_ms is None

    def test_with_empty_value_lists(self):
        m = RunMetrics(tool_latencies_ms={"search": [], "fetch": []})
        assert m.avg_tool_latency_ms is None

    def test_mixed_empty_and_nonempty_lists(self):
        m = RunMetrics(
            tool_latencies_ms={
                "search": [],
                "fetch": [60.0, 80.0],
            }
        )
        assert m.avg_tool_latency_ms == 70.0


# -----------------------------------------------------------------------
# RunMetrics.to_dict()
# -----------------------------------------------------------------------


class TestRunMetricsToDict:
    def test_contains_all_expected_keys(self):
        m = RunMetrics(
            run_id="run-123",
            agent_name="agent-1",
            state="completed",
            total_duration_s=12.3456,
            llm_calls=5,
            tool_calls=3,
            input_tokens=100,
            output_tokens=200,
            total_tokens=300,
            estimated_cost_usd=0.05,
            steps=2,
            errors=["oops"],
            llm_latencies_ms=[50.0, 60.0],
            tool_latencies_ms={"search": [30.0, 40.0]},
        )
        d = m.to_dict()
        expected_keys = {
            "run_id",
            "agent_name",
            "state",
            "success",
            "total_duration_s",
            "llm_calls",
            "tool_calls",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "estimated_cost_usd",
            "steps",
            "errors",
            "avg_llm_latency_ms",
            "avg_tool_latency_ms",
        }
        assert set(d.keys()) == expected_keys

    def test_rounds_total_duration_s_to_three_decimals(self):
        m = RunMetrics(total_duration_s=1.23456789)
        d = m.to_dict()
        assert d["total_duration_s"] == 1.235

    def test_rounds_avg_llm_latency_ms_to_one_decimal(self):
        m = RunMetrics(llm_latencies_ms=[10.123, 20.456, 30.789])
        d = m.to_dict()
        # avg = (10.123 + 20.456 + 30.789) / 3 = 20.456
        assert d["avg_llm_latency_ms"] == 20.5

    def test_rounds_avg_tool_latency_ms_to_one_decimal(self):
        m = RunMetrics(tool_latencies_ms={"t1": [10.333, 20.666]})
        d = m.to_dict()
        # avg = (10.333 + 20.666) / 2 = 15.4995
        assert d["avg_tool_latency_ms"] == 15.5

    def test_none_latencies_when_empty(self):
        m = RunMetrics()
        d = m.to_dict()
        assert d["avg_llm_latency_ms"] is None
        assert d["avg_tool_latency_ms"] is None

    def test_success_field_in_dict(self):
        m = RunMetrics(state="completed")
        d = m.to_dict()
        assert d["success"] is True

    def test_success_false_in_dict(self):
        m = RunMetrics(state="failed")
        d = m.to_dict()
        assert d["success"] is False

    def test_errors_is_copy(self):
        original_errors = ["err1", "err2"]
        m = RunMetrics(errors=original_errors)
        d = m.to_dict()
        # The dict errors list should be equal but not the same object
        assert d["errors"] == ["err1", "err2"]
        assert d["errors"] is not original_errors

    def test_scalar_values_match(self):
        m = RunMetrics(
            run_id="abc",
            agent_name="test-agent",
            state="completed",
            llm_calls=10,
            tool_calls=5,
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
            estimated_cost_usd=0.123,
            steps=4,
        )
        d = m.to_dict()
        assert d["run_id"] == "abc"
        assert d["agent_name"] == "test-agent"
        assert d["state"] == "completed"
        assert d["llm_calls"] == 10
        assert d["tool_calls"] == 5
        assert d["input_tokens"] == 1000
        assert d["output_tokens"] == 500
        assert d["total_tokens"] == 1500
        assert d["estimated_cost_usd"] == 0.123
        assert d["steps"] == 4
