"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Goal decomposition planner using LLM-assisted HTN-style planning.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..llms.types import JSONValue


@dataclass(frozen=True, slots=True)
class DecomposedTask:
    """
    A single task from goal decomposition.

    Attributes:
        id: Unique task identifier.
        description: Human-readable task description.
        dependencies: Task IDs that must complete before this task.
        estimated_cost: Estimated LLM cost in USD.
        estimated_duration_s: Estimated duration in seconds.
        optional: Whether task failure should block completion.
        metadata: Additional task metadata.
    """

    id: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_duration_s: float = 0.0
    optional: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecompositionResult:
    """
    Result of goal decomposition into tasks.

    Attributes:
        original_goal: The goal that was decomposed.
        tasks: Ordered list of decomposed tasks (topologically sorted).
        estimated_total_cost: Sum of all task estimated costs.
        estimated_total_duration_s: Sum of all task durations.
        execution_order: Task IDs in execution order.
        parallel_groups: Tasks that can run in parallel.
        metadata: Additional decomposition metadata.
    """

    original_goal: str
    tasks: list[DecomposedTask] = field(default_factory=list)
    estimated_total_cost: float = 0.0
    estimated_total_duration_s: float = 0.0
    execution_order: list[str] = field(default_factory=list)
    parallel_groups: list[list[str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_delegation_plan(self) -> dict[str, Any]:
        """
        Convert decomposition result to a delegation plan dict.

        Returns:
            Dictionary suitable for passing to DelegationPlan.from_dict().
        """
        edges = []
        for task in self.tasks:
            for dep in task.dependencies:
                edges.append({"from": dep, "to": task.id})

        return {
            "nodes": [
                {
                    "node_id": task.id,
                    "target_agent": task.metadata.get("agent", "default"),
                    "description": task.description,
                    "optional": task.optional,
                    "timeout_s": task.estimated_duration_s,
                }
                for task in self.tasks
            ],
            "edges": edges,
            "execution_order": self.execution_order,
            "parallel_groups": self.parallel_groups,
        }


@dataclass
class GoalDecomposer:
    """
    LLM-assisted goal decomposition planner.

    Takes a high-level goal and decomposes it into a DAG of tasks
    using HTN-style planning with LLM assistance.
    """

    def __init__(self, llm_client: Any | None = None, max_depth: int = 5) -> None:
        """
        Initialize goal decomposer.

        Args:
            llm_client: LLM client for decomposition assistance.
            max_depth: Maximum decomposition depth.
        """
        self._llm = llm_client
        self._max_depth = max_depth

    async def decompose(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
    ) -> DecompositionResult:
        """
        Decompose a goal into tasks.

        Args:
            goal: High-level goal to decompose.
            context: Optional context for decomposition.

        Returns:
            DecompositionResult with tasks and execution plan.
        """
        ctx = context or {}

        if self._llm:
            tasks = await self._llm_decompose(goal, ctx)
        else:
            tasks = self._rule_based_decompose(goal, ctx)

        # Compute execution order via topological sort
        execution_order = self._topological_sort(tasks)

        # Identify parallel groups
        parallel_groups = self._identify_parallel_groups(tasks, execution_order)

        # Compute estimates
        total_cost = sum(t.estimated_cost for t in tasks)
        total_duration = sum(t.estimated_duration_s for t in tasks)

        return DecompositionResult(
            original_goal=goal,
            tasks=tasks,
            estimated_total_cost=total_cost,
            estimated_total_duration_s=total_duration,
            execution_order=execution_order,
            parallel_groups=parallel_groups,
            metadata={"decomposer": "goal_decomposer", "depth": self._max_depth},
        )

    async def _llm_decompose(
        self, goal: str, context: dict[str, Any]
    ) -> list[DecomposedTask]:
        """Use LLM to decompose goal into tasks."""
        prompt = f"""Decompose the following goal into specific, actionable tasks.

Goal: {goal}

Context: {context}

Return a JSON list of tasks, each with:
- id: unique identifier (e.g., "task-1")
- description: what this task entails
- dependencies: list of task IDs this depends on
- estimated_cost: approximate USD cost
- estimated_duration_s: approximate seconds to complete
- optional: whether task failure should block completion

Format: JSON array of task objects."""

        try:
            response = await self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=context.get("model", "gpt-4.1-mini"),
            )
            # Parse response and convert to DecomposedTask objects
            import json

            task_data = json.loads(response.text)
            tasks = []
            for item in task_data:
                tasks.append(
                    DecomposedTask(
                        id=item.get("id", f"task-{len(tasks)+1}"),
                        description=item.get("description", ""),
                        dependencies=item.get("dependencies", []),
                        estimated_cost=float(item.get("estimated_cost", 0.0)),
                        estimated_duration_s=float(item.get("estimated_duration_s", 60.0)),
                        optional=bool(item.get("optional", False)),
                        metadata=item.get("metadata", {}),
                    )
                )
            return tasks
        except Exception:
            # Fallback to rule-based on LLM failure
            return self._rule_based_decompose(goal, context)

    def _rule_based_decompose(
        self, goal: str, context: dict[str, Any]
    ) -> list[DecomposedTask]:
        """Simple rule-based decomposition fallback."""
        tasks = []

        # Simple heuristic: split by common conjunctions
        import re

        segments = re.split(r"\s+(?:and|then|also|plus|additionally)\s+", goal, flags=re.IGNORECASE)

        for i, segment in enumerate(segments):
            segment = segment.strip()
            if not segment:
                continue

            task_id = f"task-{i+1}"
            deps = [f"task-{j}" for j in range(1, i + 1)]

            tasks.append(
                DecomposedTask(
                    id=task_id,
                    description=segment,
                    dependencies=deps,
                    estimated_cost=0.01 * (i + 1),
                    estimated_duration_s=30.0 * (i + 1),
                    metadata={"source": "rule_based"},
                )
            )

        return tasks

    def _topological_sort(self, tasks: list[DecomposedTask]) -> list[str]:
        """Compute topological sort of tasks."""
        # Build adjacency list
        in_degree: dict[str, int] = {t.id: 0 for t in tasks}
        adj: dict[str, list[str]] = {t.id: [] for t in tasks}

        for task in tasks:
            for dep in task.dependencies:
                if dep in adj:
                    adj[dep].append(task.id)
                    in_degree[task.id] += 1

        # Kahn's algorithm
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            # Sort for deterministic ordering
            queue.sort()
            tid = queue.pop(0)
            result.append(tid)

            for neighbor in adj[tid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return result

    def _identify_parallel_groups(
        self, tasks: list[DecomposedTask], execution_order: list[str]
    ) -> list[list[str]]:
        """Identify tasks that can execute in parallel."""
        task_map = {t.id: t for t in tasks}
        completed = set()
        groups = []

        remaining = list(execution_order)
        while remaining:
            # Find tasks with all dependencies satisfied
            ready = []
            for tid in remaining:
                task = task_map[tid]
                if all(dep in completed for dep in task.dependencies):
                    ready.append(tid)

            if not ready:
                # Cycle detected, just add remaining
                groups.append(remaining)
                break

            groups.append(ready)
            completed.update(ready)
            remaining = [t for t in remaining if t not in ready]

        return groups

    async def refine_plan(
        self, plan: DecompositionResult, feedback: str
    ) -> DecompositionResult:
        """
        Refine an existing plan based on feedback.

        Args:
            plan: Existing plan to refine.
            feedback: Feedback for improvement.

        Returns:
            New refined DecompositionResult.
        """
        # Simple refinement: re-decompose with context
        context = {
            "original_goal": plan.original_goal,
            "feedback": feedback,
            "current_tasks": [t.description for t in plan.tasks],
        }
        return await self.decompose(plan.original_goal, context)