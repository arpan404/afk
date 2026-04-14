"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Conversation-level metrics for agent run telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """
    Aggregated metrics for a complete agent run.

    Attributes:
        run_id: Run identifier.
        thread_id: Thread identifier.
        total_llm_calls: Number of LLM API calls made.
        total_tool_calls: Number of tool executions.
        total_subagent_calls: Number of subagent invocations.
        total_cost_usd: Aggregated cost in USD.
        total_steps: Total steps executed.
        success_rate: Ratio of successful to total operations (0.0 to 1.0).
        avg_llm_latency_ms: Average LLM response latency in milliseconds.
        avg_tool_latency_ms: Average tool execution latency in milliseconds.
        p50_llm_latency_ms: 50th percentile LLM latency.
        p90_llm_latency_ms: 90th percentile LLM latency.
        p99_llm_latency_ms: 99th percentile LLM latency.
        error_count: Number of errors encountered.
        circuit_breaker_trips: Number of circuit breaker trips.
        cache_hit_rate: Ratio of cached responses to total LLM calls.
        total_input_tokens: Total prompt tokens consumed.
        total_output_tokens: Total completion tokens produced.
        wall_time_s: Total wall-clock time for the run.
        state: Terminal agent state.
        degraded: Whether run ended in degraded state.
    """

    run_id: str
    thread_id: str
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_subagent_calls: int = 0
    total_cost_usd: float = 0.0
    total_steps: int = 0
    success_rate: float = 1.0
    avg_llm_latency_ms: float = 0.0
    avg_tool_latency_ms: float = 0.0
    p50_llm_latency_ms: float = 0.0
    p90_llm_latency_ms: float = 0.0
    p99_llm_latency_ms: float = 0.0
    error_count: int = 0
    circuit_breaker_trips: int = 0
    cache_hit_rate: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    wall_time_s: float = 0.0
    state: str = "completed"
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_subagent_calls": self.total_subagent_calls,
            "total_cost_usd": self.total_cost_usd,
            "total_steps": self.total_steps,
            "success_rate": self.success_rate,
            "avg_llm_latency_ms": self.avg_llm_latency_ms,
            "avg_tool_latency_ms": self.avg_tool_latency_ms,
            "p50_llm_latency_ms": self.p50_llm_latency_ms,
            "p90_llm_latency_ms": self.p90_llm_latency_ms,
            "p99_llm_latency_ms": self.p99_llm_latency_ms,
            "error_count": self.error_count,
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "cache_hit_rate": self.cache_hit_rate,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "wall_time_s": self.wall_time_s,
            "state": self.state,
            "degraded": self.degraded,
        }


@dataclass
class MetricsCollector:
    """
    In-memory metrics collector for tracking run telemetry.

    Accumulates latencies, counts, and costs during a run and produces
    a final RunMetrics snapshot at completion.
    """

    run_id: str
    thread_id: str

    _llm_latencies: list[float] = field(default_factory=list)
    _tool_latencies: list[float] = field(default_factory=list)
    _llm_calls: int = 0
    _tool_calls: int = 0
    _subagent_calls: int = 0
    _errors: int = 0
    _circuit_trips: int = 0
    _cache_hits: int = 0
    _total_cost: float = 0.0
    _input_tokens: int = 0
    _output_tokens: int = 0
    _start_time_s: float | None = None
    _end_time_s: float | None = None
    _steps: int = 0
    _degraded: bool = False
    _terminal_state: str = "completed"

    def record_llm_call(self, latency_ms: float, cached: bool = False) -> None:
        """Record a single LLM call with its latency."""
        self._llm_latencies.append(latency_ms)
        self._llm_calls += 1
        if cached:
            self._cache_hits += 1

    def record_tool_call(self, latency_ms: float) -> None:
        """Record a single tool call with its latency."""
        self._tool_latencies.append(latency_ms)
        self._tool_calls += 1

    def record_subagent_call(self) -> None:
        """Record a subagent invocation."""
        self._subagent_calls += 1

    def record_error(self) -> None:
        """Record an error occurrence."""
        self._errors += 1

    def record_circuit_trip(self) -> None:
        """Record a circuit breaker trip."""
        self._circuit_trips += 1

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage from LLM response."""
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens

    def record_cost(self, cost_usd: float) -> None:
        """Record cost in USD."""
        self._total_cost += cost_usd

    def start(self) -> None:
        """Mark run start time."""
        import time

        self._start_time_s = time.time()

    def end(self, state: str = "completed", degraded: bool = False) -> None:
        """Mark run end time and terminal state."""
        import time

        self._end_time_s = time.time()
        self._terminal_state = state
        self._degraded = degraded

    def record_step(self) -> None:
        """Increment step count."""
        self._steps += 1

    def to_run_metrics(self) -> RunMetrics:
        """
        Compute final metrics from accumulated data.

        Returns:
            RunMetrics snapshot with all aggregations.
        """
        import statistics

        wall_time = 0.0
        if self._start_time_s and self._end_time_s:
            wall_time = self._end_time_s - self._start_time_s

        # Calculate latency percentiles
        llm_latencies = sorted(self._llm_latencies) if self._llm_latencies else [0.0]
        total_ops = self._llm_calls + self._tool_calls
        success_rate = (
            1.0 - (self._errors / total_ops) if total_ops > 0 else 1.0
        )

        def percentile(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            idx = int(len(data) * p / 100)
            idx = min(idx, len(data) - 1)
            return data[idx]

        return RunMetrics(
            run_id=self.run_id,
            thread_id=self.thread_id,
            total_llm_calls=self._llm_calls,
            total_tool_calls=self._tool_calls,
            total_subagent_calls=self._subagent_calls,
            total_cost_usd=round(self._total_cost, 6),
            total_steps=self._steps,
            success_rate=round(success_rate, 4),
            avg_llm_latency_ms=round(statistics.mean(llm_latencies), 2),
            avg_tool_latency_ms=round(
                statistics.mean(self._tool_latencies) if self._tool_latencies else 0.0, 2
            ),
            p50_llm_latency_ms=round(percentile(llm_latencies, 50), 2),
            p90_llm_latency_ms=round(percentile(llm_latencies, 90), 2),
            p99_llm_latency_ms=round(percentile(llm_latencies, 99), 2),
            error_count=self._errors,
            circuit_breaker_trips=self._circuit_trips,
            cache_hit_rate=round(
                self._cache_hits / self._llm_calls if self._llm_calls > 0 else 0.0, 4
            ),
            total_input_tokens=self._input_tokens,
            total_output_tokens=self._output_tokens,
            wall_time_s=round(wall_time, 3),
            state=self._terminal_state,
            degraded=self._degraded,
        )