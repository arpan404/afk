"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Agent evaluation harness for benchmark testing.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..types import AgentResult
from ..core.base import BaseAgent


@dataclass
class EvalTask:
    """
    A single evaluation task for an agent.

    Attributes:
        id: Unique task identifier.
        name: Human-readable task name.
        description: Task description.
        prompt: Input prompt for the agent.
        expected_outcome: Description of expected outcome.
        validation_fn: Optional function to validate result.
        timeout_s: Maximum allowed execution time.
        tags: Tags for categorization.
        metadata: Additional task metadata.
    """

    id: str
    name: str
    description: str
    prompt: str
    expected_outcome: str = ""
    validation_fn: Callable[[AgentResult], bool] | None = None
    timeout_s: float = 300.0
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """
    Result of evaluating a single task.

    Attributes:
        task_id: Task that was evaluated.
        success: Whether task succeeded.
        score: Task-specific score (0.0 to 1.0).
        latency_s: Execution time in seconds.
        result: AgentResult from execution.
        error: Error message if failed.
        validation_details: Details from validation function.
    """

    task_id: str
    success: bool
    score: float = 0.0
    latency_s: float = 0.0
    result: AgentResult | None = None
    error: str | None = None
    validation_details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """
    Aggregated benchmark results.

    Attributes:
        benchmark_name: Name of the benchmark.
        total_tasks: Total number of tasks.
        passed: Number of tasks that passed.
        failed: Number of tasks that failed.
        avg_score: Average score across all tasks.
        avg_latency_s: Average latency in seconds.
        total_time_s: Total benchmark time.
        task_results: Individual task results.
        metadata: Additional benchmark metadata.
    """

    benchmark_name: str
    total_tasks: int = 0
    passed: int = 0
    failed: int = 0
    avg_score: float = 0.0
    avg_latency_s: float = 0.0
    total_time_s: float = 0.0
    task_results: list[EvalResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "total_tasks": self.total_tasks,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "avg_latency_s": self.avg_latency_s,
            "total_time_s": self.total_time_s,
            "task_results": [
                {
                    "task_id": r.task_id,
                    "success": r.success,
                    "score": r.score,
                    "latency_s": r.latency_s,
                    "error": r.error,
                }
                for r in self.task_results
            ],
            "metadata": self.metadata,
        }

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        return self.passed / self.total_tasks if self.total_tasks > 0 else 0.0


@dataclass
class AgentEvaluator:
    """
    Evaluation harness for agent benchmarks.

    Runs a set of tasks against an agent and produces
    aggregated metrics and results.
    """

    def __init__(
        self,
        benchmark_name: str = "agent-benchmark",
        max_concurrency: int = 4,
        default_timeout_s: float = 300.0,
    ) -> None:
        """
        Initialize evaluator.

        Args:
            benchmark_name: Name for this benchmark.
            max_concurrency: Maximum parallel task execution.
            default_timeout_s: Default task timeout.
        """
        self._benchmark_name = benchmark_name
        self._max_concurrency = max_concurrency
        self._default_timeout_s = default_timeout_s
        self._tasks: list[EvalTask] = []

    def add_task(self, task: EvalTask) -> None:
        """Add a task to the benchmark."""
        self._tasks.append(task)

    def add_tasks(self, tasks: list[EvalTask]) -> None:
        """Add multiple tasks to the benchmark."""
        self._tasks.extend(tasks)

    async def run(
        self,
        agent: BaseAgent,
        runner: Any,
        *,
        concurrency: int | None = None,
        early_stop_on_failure: bool = False,
    ) -> BenchmarkResult:
        """
        Run benchmark against an agent.

        Args:
            agent: Agent to evaluate.
            runner: Runner instance for execution.
            concurrency: Override max concurrency.
            early_stop_on_failure: Stop on first failure.

        Returns:
            BenchmarkResult with aggregated metrics.
        """
        max_conc = concurrency or self._max_concurrency
        start_time = time.time()
        results: list[EvalResult] = []

        # Create semaphore for concurrency control
        sem = asyncio.Semaphore(max_conc)

        async def run_task(task: EvalTask) -> EvalResult:
            async with sem:
                return await self._run_single_task(agent, runner, task)

        # Run tasks with concurrency control
        coroutines = [run_task(task) for task in self._tasks]
        task_results = await asyncio.gather(*coroutines, return_exceptions=True)

        for i, result in enumerate(task_results):
            if isinstance(result, Exception):
                results.append(
                    EvalResult(
                        task_id=self._tasks[i].id,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                results.append(result)

        total_time = time.time() - start_time
        passed = sum(1 for r in results if r.success)
        failed = len(results) - passed
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0
        avg_latency = sum(r.latency_s for r in results) / len(results) if results else 0.0

        return BenchmarkResult(
            benchmark_name=self._benchmark_name,
            total_tasks=len(results),
            passed=passed,
            failed=failed,
            avg_score=avg_score,
            avg_latency_s=avg_latency,
            total_time_s=total_time,
            task_results=results,
            metadata={
                "concurrency": max_conc,
                "timestamp": int(time.time() * 1000),
            },
        )

    async def _run_single_task(
        self,
        agent: BaseAgent,
        runner: Any,
        task: EvalTask,
    ) -> EvalResult:
        """Run a single evaluation task."""
        start_time = time.time()
        timeout = task.timeout_s or self._default_timeout_s

        try:
            # Run agent with timeout
            result = await asyncio.wait_for(
                runner.run(
                    agent,
                    user_message=task.prompt,
                    thread_id=f"eval-{task.id}",
                ),
                timeout=timeout,
            )

            latency = time.time() - start_time

            # Validate result
            validation_details = {}
            success = False
            score = 0.0

            if task.validation_fn:
                try:
                    success = task.validation_fn(result)
                    validation_details["validation_passed"] = success
                    score = 1.0 if success else 0.0
                except Exception as e:
                    validation_details["validation_error"] = str(e)
                    success = False
                    score = 0.0
            else:
                # Default: check if result has non-empty text
                success = bool(result and result.final_text)
                score = 1.0 if success else 0.0

            return EvalResult(
                task_id=task.id,
                success=success,
                score=score,
                latency_s=latency,
                result=result,
                validation_details=validation_details,
            )

        except asyncio.TimeoutError:
            return EvalResult(
                task_id=task.id,
                success=False,
                score=0.0,
                latency_s=timeout,
                error=f"Task timed out after {timeout}s",
            )
        except Exception as e:
            return EvalResult(
                task_id=task.id,
                success=False,
                score=0.0,
                latency_s=time.time() - start_time,
                error=str(e),
            )


def load_benchmark(name: str) -> list[EvalTask]:
    """
    Load a named benchmark dataset.

    Args:
        name: Benchmark name (e.g., "agent-bench-v1").

    Returns:
        List of EvalTask objects.
    """
    # This is a placeholder for loading actual benchmarks
    # In production, this would load from files, databases, etc.
    return []


def create_safety_eval_task(
    task_id: str,
    prompt: str,
    forbidden_patterns: list[str],
) -> EvalTask:
    """
    Create a safety evaluation task.

    Args:
        task_id: Task identifier.
        prompt: Input prompt.
        forbidden_patterns: Patterns that should NOT appear in output.

    Returns:
        EvalTask configured for safety checking.
    """

    def validate_safety(result: AgentResult) -> bool:
        if not result or not result.final_text:
            return False
        text = result.final_text.lower()
        for pattern in forbidden_patterns:
            if pattern.lower() in text:
                return False
        return True

    return EvalTask(
        id=task_id,
        name=f"safety-{task_id}",
        description=f"Safety evaluation: prevent {forbidden_patterns}",
        prompt=prompt,
        validation_fn=validate_safety,
        tags=["safety", "security"],
    )


def create_correctness_eval_task(
    task_id: str,
    prompt: str,
    expected_keywords: list[str],
) -> EvalTask:
    """
    Create a correctness evaluation task.

    Args:
        task_id: Task identifier.
        prompt: Input prompt.
        expected_keywords: Keywords that should appear in output.

    Returns:
        EvalTask configured for correctness checking.
    """

    def validate_correctness(result: AgentResult) -> bool:
        if not result or not result.final_text:
            return False
        text = result.final_text.lower()
        return all(kw.lower() in text for kw in expected_keywords)

    return EvalTask(
        id=task_id,
        name=f"correctness-{task_id}",
        description=f"Correctness evaluation: requires {expected_keywords}",
        prompt=prompt,
        validation_fn=validate_correctness,
        tags=["correctness", "quality"],
    )