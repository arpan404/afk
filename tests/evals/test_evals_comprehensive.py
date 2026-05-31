"""
Comprehensive tests for the afk.evals sub-package.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from afk.agents.types.policy import AgentRunEvent
from afk.evals.budgets import EvalBudget, evaluate_budget
from afk.evals.datasets import (
    _as_json_obj,
    _as_tags,
    _json_cast,
    load_eval_cases_json,
)
from afk.evals.golden import compare_event_types, write_golden_trace
from afk.evals.models import (
    EvalAssertionResult,
    EvalCase,
    EvalCaseResult,
    EvalSuiteConfig,
    EvalSuiteResult,
)
from afk.observability.models import RunMetrics

# ======================================================================
# Fake agent for EvalCase construction (avoids real Agent dependencies)
# ======================================================================


class _FakeAgent:
    """Minimal agent stub satisfying BaseAgent duck-typing for EvalCase."""

    name = "test"
    instructions = "stub"


def _resolver(name: str):
    """Agent resolver returning a _FakeAgent for any name."""
    return _FakeAgent()


# ======================================================================
# EvalCase
# ======================================================================


class TestEvalCase:
    """Tests for EvalCase dataclass defaults."""

    def test_defaults(self):
        agent = _FakeAgent()
        case = EvalCase(name="c1", agent=agent)
        assert case.name == "c1"
        assert case.agent is agent
        assert case.user_message is None
        assert case.context == {}
        assert case.thread_id is None
        assert case.tags == ()
        assert case.budget is None

    def test_custom_values(self):
        agent = _FakeAgent()
        budget = EvalBudget(max_duration_s=10.0)
        case = EvalCase(
            name="c2",
            agent=agent,
            user_message="hello",
            context={"key": "val"},
            thread_id="tid",
            tags=("a", "b"),
            budget=budget,
        )
        assert case.user_message == "hello"
        assert case.context == {"key": "val"}
        assert case.thread_id == "tid"
        assert case.tags == ("a", "b")
        assert case.budget is budget


# ======================================================================
# EvalAssertionResult
# ======================================================================


class TestEvalAssertionResult:
    """Tests for EvalAssertionResult dataclass."""

    def test_required_fields(self):
        r = EvalAssertionResult(name="assert1", passed=True)
        assert r.name == "assert1"
        assert r.passed is True

    def test_defaults(self):
        r = EvalAssertionResult(name="a", passed=False)
        assert r.details is None
        assert r.score is None

    def test_custom_values(self):
        r = EvalAssertionResult(
            name="a", passed=True, details="all good", score=0.95
        )
        assert r.details == "all good"
        assert r.score == 0.95


# ======================================================================
# EvalCaseResult
# ======================================================================


class TestEvalCaseResult:
    """Tests for EvalCaseResult dataclass."""

    def test_required_fields(self):
        r = EvalCaseResult(
            case="c1",
            state="completed",
            final_text="done",
            run_id="r1",
            thread_id="t1",
        )
        assert r.case == "c1"
        assert r.state == "completed"
        assert r.final_text == "done"
        assert r.run_id == "r1"
        assert r.thread_id == "t1"

    def test_defaults(self):
        r = EvalCaseResult(
            case="c1",
            state="completed",
            final_text="done",
            run_id="r1",
            thread_id="t1",
        )
        assert r.event_types == []
        assert isinstance(r.metrics, RunMetrics)
        assert r.assertions == []
        assert r.budget_violations == []
        assert r.passed is True

    def test_custom_values(self):
        assertion = EvalAssertionResult(name="a1", passed=False)
        metrics = RunMetrics(run_id="r1", total_duration_s=5.0)
        r = EvalCaseResult(
            case="c1",
            state="failed",
            final_text="err",
            run_id="r1",
            thread_id="t1",
            event_types=["run_started", "run_failed"],
            metrics=metrics,
            assertions=[assertion],
            budget_violations=["over budget"],
            passed=False,
        )
        assert r.event_types == ["run_started", "run_failed"]
        assert r.metrics is metrics
        assert len(r.assertions) == 1
        assert r.budget_violations == ["over budget"]
        assert r.passed is False


# ======================================================================
# EvalSuiteConfig
# ======================================================================


class TestEvalSuiteConfig:
    """Tests for EvalSuiteConfig dataclass."""

    def test_defaults(self):
        cfg = EvalSuiteConfig()
        assert cfg.execution_mode == "adaptive"
        assert cfg.max_concurrency == 4
        assert cfg.fail_fast is False
        assert cfg.assertions == ()
        assert cfg.scorers == ()
        assert cfg.budget is None


# ======================================================================
# EvalSuiteResult
# ======================================================================


class TestEvalSuiteResult:
    """Tests for EvalSuiteResult computed properties."""

    def _make_case_result(self, *, passed: bool = True) -> EvalCaseResult:
        return EvalCaseResult(
            case="c",
            state="completed" if passed else "failed",
            final_text="text",
            run_id="r",
            thread_id="t",
            passed=passed,
        )

    def test_total_counts_results(self):
        suite = EvalSuiteResult(
            results=[self._make_case_result(), self._make_case_result()],
            execution_mode="sequential",
        )
        assert suite.total == 2

    def test_passed_counts_passed_results(self):
        suite = EvalSuiteResult(
            results=[
                self._make_case_result(passed=True),
                self._make_case_result(passed=False),
                self._make_case_result(passed=True),
            ],
            execution_mode="parallel",
        )
        assert suite.passed == 2

    def test_failed_counts_failed_results(self):
        suite = EvalSuiteResult(
            results=[
                self._make_case_result(passed=True),
                self._make_case_result(passed=False),
                self._make_case_result(passed=False),
            ],
            execution_mode="adaptive",
        )
        assert suite.failed == 2

    def test_empty_results(self):
        suite = EvalSuiteResult(results=[], execution_mode="adaptive")
        assert suite.total == 0
        assert suite.passed == 0
        assert suite.failed == 0


# ======================================================================
# EvalBudget
# ======================================================================


class TestEvalBudget:
    """Tests for EvalBudget dataclass."""

    def test_defaults(self):
        budget = EvalBudget()
        assert budget.max_duration_s is None
        assert budget.max_total_tokens is None
        assert budget.max_total_cost_usd is None

    def test_custom_values(self):
        budget = EvalBudget(
            max_duration_s=60.0, max_total_tokens=1000, max_total_cost_usd=0.50
        )
        assert budget.max_duration_s == 60.0
        assert budget.max_total_tokens == 1000
        assert budget.max_total_cost_usd == 0.50


# ======================================================================
# evaluate_budget
# ======================================================================


class TestEvaluateBudget:
    """Tests for the evaluate_budget function."""

    def test_none_budget_returns_empty_list(self):
        metrics = RunMetrics()
        assert evaluate_budget(metrics, None) == []

    def test_budget_with_no_limits_returns_empty_list(self):
        metrics = RunMetrics(total_duration_s=100.0, total_tokens=5000)
        budget = EvalBudget()
        assert evaluate_budget(metrics, budget) == []

    def test_duration_exceeded_returns_violation(self):
        metrics = RunMetrics(total_duration_s=15.0)
        budget = EvalBudget(max_duration_s=10.0)
        violations = evaluate_budget(metrics, budget)
        assert len(violations) == 1
        assert "duration_s" in violations[0]
        assert "max_duration_s" in violations[0]

    def test_token_budget_exceeded_returns_violation(self):
        metrics = RunMetrics(total_tokens=2000)
        budget = EvalBudget(max_total_tokens=1000)
        violations = evaluate_budget(metrics, budget)
        assert len(violations) == 1
        assert "total_tokens" in violations[0]
        assert "max_total_tokens" in violations[0]

    def test_cost_budget_exceeded_returns_violation(self):
        metrics = RunMetrics(estimated_cost_usd=1.50)
        budget = EvalBudget(max_total_cost_usd=1.00)
        violations = evaluate_budget(metrics, budget)
        assert len(violations) == 1
        assert "total_cost_usd" in violations[0]
        assert "max_total_cost_usd" in violations[0]

    def test_multiple_violations_returned(self):
        metrics = RunMetrics(
            total_duration_s=20.0,
            total_tokens=5000,
            estimated_cost_usd=5.0,
        )
        budget = EvalBudget(
            max_duration_s=10.0,
            max_total_tokens=1000,
            max_total_cost_usd=1.0,
        )
        violations = evaluate_budget(metrics, budget)
        assert len(violations) == 3

    def test_budget_not_exceeded_returns_empty(self):
        metrics = RunMetrics(
            total_duration_s=5.0,
            total_tokens=500,
            estimated_cost_usd=0.10,
        )
        budget = EvalBudget(
            max_duration_s=10.0,
            max_total_tokens=1000,
            max_total_cost_usd=1.0,
        )
        assert evaluate_budget(metrics, budget) == []

    def test_estimated_cost_none_with_cost_budget_no_violation(self):
        """When estimated_cost_usd is None, cost budget check is skipped."""
        metrics = RunMetrics(estimated_cost_usd=None)
        budget = EvalBudget(max_total_cost_usd=1.0)
        assert evaluate_budget(metrics, budget) == []


# ======================================================================
# load_eval_cases_json
# ======================================================================


class TestLoadEvalCasesJson:
    """Tests for load_eval_cases_json dataset loader."""

    def test_valid_json_loads_correctly(self, tmp_path: Path):
        data = [
            {"name": "case1", "agent": "test", "user_message": "hi"},
            {"name": "case2", "agent": "test"},
        ]
        p = tmp_path / "cases.json"
        p.write_text(json.dumps(data))
        cases = load_eval_cases_json(p, agent_resolver=_resolver)
        assert len(cases) == 2
        assert cases[0].name == "case1"
        assert cases[0].user_message == "hi"
        assert cases[1].name == "case2"
        assert cases[1].user_message is None

    def test_non_list_json_raises(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"not": "a list"}))
        with pytest.raises(ValueError, match="list"):
            load_eval_cases_json(p, agent_resolver=_resolver)

    def test_row_without_name_raises(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps([{"agent": "test"}]))
        with pytest.raises(ValueError, match="name"):
            load_eval_cases_json(p, agent_resolver=_resolver)

    def test_row_without_agent_raises(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps([{"name": "c1"}]))
        with pytest.raises(ValueError, match="agent"):
            load_eval_cases_json(p, agent_resolver=_resolver)

    def test_non_dict_row_raises(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(["not a dict"]))
        with pytest.raises(ValueError, match="object"):
            load_eval_cases_json(p, agent_resolver=_resolver)

    def test_tags_parsed_as_tuple(self, tmp_path: Path):
        data = [{"name": "c1", "agent": "test", "tags": ["alpha", "beta"]}]
        p = tmp_path / "cases.json"
        p.write_text(json.dumps(data))
        cases = load_eval_cases_json(p, agent_resolver=_resolver)
        assert cases[0].tags == ("alpha", "beta")

    def test_non_string_tag_raises(self, tmp_path: Path):
        data = [{"name": "c1", "agent": "test", "tags": [123]}]
        p = tmp_path / "cases.json"
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="strings"):
            load_eval_cases_json(p, agent_resolver=_resolver)

    def test_context_parsed_as_dict(self, tmp_path: Path):
        data = [{"name": "c1", "agent": "test", "context": {"foo": "bar"}}]
        p = tmp_path / "cases.json"
        p.write_text(json.dumps(data))
        cases = load_eval_cases_json(p, agent_resolver=_resolver)
        assert cases[0].context == {"foo": "bar"}

    def test_non_dict_context_raises(self, tmp_path: Path):
        data = [{"name": "c1", "agent": "test", "context": "not_a_dict"}]
        p = tmp_path / "cases.json"
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="object"):
            load_eval_cases_json(p, agent_resolver=_resolver)


# ======================================================================
# compare_event_types
# ======================================================================


class TestCompareEventTypes:
    """Tests for the compare_event_types golden helper."""

    def test_matching_lists(self):
        ok, msg = compare_event_types(["a", "b"], ["a", "b"])
        assert ok is True
        assert msg is None

    def test_different_lists(self):
        ok, msg = compare_event_types(["a", "b"], ["a", "c"])
        assert ok is False
        assert msg is not None
        assert "expected" in msg
        assert "observed" in msg

    def test_both_empty(self):
        ok, msg = compare_event_types([], [])
        assert ok is True
        assert msg is None


# ======================================================================
# write_golden_trace
# ======================================================================


class TestWriteGoldenTrace:
    """Tests for the write_golden_trace golden helper."""

    def test_writes_json_file(self, tmp_path: Path):
        event = AgentRunEvent(
            type="run_started",
            run_id="r1",
            thread_id="t1",
            state="running",
            step=0,
            message="started",
        )
        out_path = tmp_path / "golden.json"
        write_golden_trace(out_path, [event])
        assert out_path.exists()
        rows = json.loads(out_path.read_text(encoding="utf-8"))
        assert isinstance(rows, list)
        assert len(rows) == 1

    def test_row_fields(self, tmp_path: Path):
        event = AgentRunEvent(
            type="run_started",
            run_id="r1",
            thread_id="t1",
            state="running",
            step=0,
            message="started",
            data={"info": "val"},
        )
        out_path = tmp_path / "golden.json"
        write_golden_trace(out_path, [event])
        rows = json.loads(out_path.read_text(encoding="utf-8"))
        row = rows[0]
        assert row["type"] == "run_started"
        assert row["state"] == "running"
        assert row["step"] == 0
        assert row["message"] == "started"
        assert row["data"] == {"info": "val"}

    def test_multiple_events(self, tmp_path: Path):
        events = [
            AgentRunEvent(
                type="run_started",
                run_id="r1",
                thread_id="t1",
                state="running",
                step=0,
                message="start",
            ),
            AgentRunEvent(
                type="run_completed",
                run_id="r1",
                thread_id="t1",
                state="completed",
                step=1,
                message="done",
            ),
        ]
        out_path = tmp_path / "golden.json"
        write_golden_trace(out_path, events)
        rows = json.loads(out_path.read_text(encoding="utf-8"))
        assert len(rows) == 2
        assert rows[0]["type"] == "run_started"
        assert rows[1]["type"] == "run_completed"


# ======================================================================
# _as_json_obj, _as_tags, _json_cast helpers
# ======================================================================


class TestAsJsonObj:
    """Tests for the _as_json_obj helper."""

    def test_none_returns_empty_dict(self):
        assert _as_json_obj(None) == {}

    def test_dict_passes_through(self):
        assert _as_json_obj({"key": "val"}) == {"key": "val"}

    def test_non_dict_raises(self):
        with pytest.raises(ValueError):
            _as_json_obj(42)


class TestAsTags:
    """Tests for the _as_tags helper."""

    def test_none_returns_empty_tuple(self):
        assert _as_tags(None) == ()

    def test_list_of_strings(self):
        assert _as_tags(["a", "b"]) == ("a", "b")

    def test_non_list_raises(self):
        with pytest.raises(ValueError):
            _as_tags(42)

    def test_non_string_tag_raises(self):
        with pytest.raises(ValueError, match="strings"):
            _as_tags([1])


class TestJsonCast:
    """Tests for the _json_cast helper."""

    def test_none_identity(self):
        assert _json_cast(None) is None

    def test_str_identity(self):
        assert _json_cast("hello") == "hello"

    def test_int_identity(self):
        assert _json_cast(42) == 42

    def test_float_identity(self):
        assert _json_cast(3.14) == 3.14

    def test_bool_identity(self):
        assert _json_cast(True) is True
        assert _json_cast(False) is False

    def test_list_recursive(self):
        result = _json_cast([1, "two", [3]])
        assert result == [1, "two", [3]]

    def test_dict_recursive_with_str_keys(self):
        result = _json_cast({1: "a", "b": 2})
        assert result == {"1": "a", "b": 2}

    def test_object_stringified(self):
        obj = object()
        result = _json_cast(obj)
        assert result == str(obj)

    def test_nested_complex(self):
        result = _json_cast({"items": [1, {"nested": True}]})
        assert result == {"items": [1, {"nested": True}]}
