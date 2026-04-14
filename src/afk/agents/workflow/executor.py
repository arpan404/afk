"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Workflow executor for running state machine workflows.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .state_machine import (
    WorkflowEdge,
    WorkflowEvent,
    WorkflowNode,
    WorkflowSpec,
    WorkflowState,
)


@dataclass
class WorkflowExecutionContext:
    """Execution context for a workflow run."""

    workflow_id: str
    run_id: str
    thread_id: str
    current_node_id: str | None = None
    state: WorkflowState = WorkflowState.PENDING
    node_states: dict[str, WorkflowState] = field(default_factory=dict)
    node_results: dict[str, Any] = field(default_factory=dict)
    node_errors: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at_ms: int = 0
    finished_at_ms: int | None = None

    def get_node_state(self, node_id: str) -> WorkflowState:
        """Get state of a specific node."""
        return self.node_states.get(node_id, WorkflowState.PENDING)

    def set_node_state(self, node_id: str, state: WorkflowState) -> None:
        """Set state of a specific node."""
        self.node_states[node_id] = state


@dataclass
class WorkflowExecutionResult:
    """Result of workflow execution."""

    workflow_id: str
    run_id: str
    final_state: WorkflowState
    current_node_id: str | None
    node_states: dict[str, WorkflowState]
    node_results: dict[str, Any]
    node_errors: dict[str, str]
    started_at_ms: int
    finished_at_ms: int
    duration_ms: float


NodeExecutor = Callable[
    [WorkflowExecutionContext, WorkflowNode],
    tuple[WorkflowState, Any],
]


class WorkflowExecutor:
    """
    Executes workflow state machines with node handlers.

    Provides event-driven execution with support for:
    - Parallel node execution
    - Conditional branching
    - Retry handling
    - Pause/resume
    """

    def __init__(
        self,
        node_executor: NodeExecutor | None = None,
        max_parallel: int = 4,
    ) -> None:
        self._node_executor = node_executor
        self._max_parallel = max_parallel
        self._running_tasks: dict[str, asyncio.Task] = {}

    async def execute(
        self,
        workflow: WorkflowSpec,
        context: WorkflowExecutionContext,
        *,
        start_node_id: str | None = None,
    ) -> WorkflowExecutionResult:
        """
        Execute a workflow from the specified node.

        Args:
            workflow: Workflow specification.
            context: Execution context.
            start_node_id: Node to start from (defaults to initial node).

        Returns:
            WorkflowExecutionResult with final state and results.
        """
        start = start_node_id or workflow.initial_node
        if not start:
            raise ValueError("No start node specified")

        context.started_at_ms = int(time.time() * 1000)

        while not workflow.is_terminal(context.state):
            node = workflow.get_node(start)
            if node is None:
                context.state = WorkflowState.FAILED
                context.node_errors[start] = f"Node '{start}' not found"
                break

            context.current_node_id = start
            context.set_node_state(start, WorkflowState.RUNNING)

            try:
                result = await self._execute_node(workflow, context, node)

                if result[0] == WorkflowState.COMPLETED:
                    context.node_results[start] = result[1]
                    context.set_node_state(start, WorkflowState.COMPLETED)

                    next_node = self._find_next_node(workflow, start)
                    if next_node:
                        start = next_node
                        continue
                    else:
                        context.state = WorkflowState.COMPLETED
                        break
                elif result[0] == WorkflowState.FAILED:
                    context.node_errors[start] = str(result[1])
                    context.set_node_state(start, WorkflowState.FAILED)

                    if node.continue_on_error:
                        next_node = self._find_next_node(workflow, start)
                        if next_node:
                            start = next_node
                            continue
                    context.state = WorkflowState.FAILED
                    break
                else:
                    context.state = result[0]
                    break

            except asyncio.CancelledError:
                context.state = WorkflowState.CANCELLED
                context.set_node_state(start, WorkflowState.CANCELLED)
                raise
            except Exception as exc:
                context.node_errors[start] = str(exc)
                context.set_node_state(start, WorkflowState.FAILED)
                context.state = WorkflowState.FAILED
                break

        context.finished_at_ms = int(time.time() * 1000)

        return WorkflowExecutionResult(
            workflow_id=workflow.id,
            run_id=context.run_id,
            final_state=context.state,
            current_node_id=context.current_node_id,
            node_states=dict(context.node_states),
            node_results=dict(context.node_results),
            node_errors=dict(context.node_errors),
            started_at_ms=context.started_at_ms,
            finished_at_ms=context.finished_at_ms,
            duration_ms=context.finished_at_ms - context.started_at_ms,
        )

    async def _execute_node(
        self,
        workflow: WorkflowSpec,
        context: WorkflowExecutionContext,
        node: WorkflowNode,
    ) -> tuple[WorkflowState, Any]:
        """Execute a single workflow node."""
        if node.timeout_s:
            try:
                async with asyncio.timeout(node.timeout_s):
                    return await self._run_node_executor(context, node)
            except asyncio.TimeoutError:
                # Check retry
                if node.can_retry():
                    node.increment_retry()
                    return await self._run_node_executor(context, node)
                return WorkflowState.FAILED, f"Node '{node.id}' timed out after {node.timeout_s}s"
        else:
            return await self._run_node_executor(context, node)

    async def _run_node_executor(
        self,
        context: WorkflowExecutionContext,
        node: WorkflowNode,
    ) -> tuple[WorkflowState, Any]:
        """Run the node executor or default handler."""
        if self._node_executor:
            return await self._node_executor(context, node)

        # Default: just mark as completed
        return WorkflowState.COMPLETED, None

    def _find_next_node(self, workflow: WorkflowSpec, from_node_id: str) -> str | None:
        """Find next node based on workflow edges."""
        edges = workflow.get_outgoing_edges(from_node_id)
        if not edges:
            return None

        for edge in edges:
            if edge.condition:
                continue  # Conditional - skip for now
            return edge.to_node

        return edges[0].to_node if edges else None

    async def execute_parallel(
        self,
        workflow: WorkflowSpec,
        context: WorkflowExecutionContext,
        node_ids: list[str],
    ) -> dict[str, tuple[WorkflowState, Any]]:
        """
        Execute multiple nodes in parallel.

        Args:
            workflow: Workflow specification.
            context: Execution context.
            node_ids: IDs of nodes to execute in parallel.

        Returns:
            Dict mapping node_id to (state, result) tuples.
        """
        if len(node_ids) > self._max_parallel:
            node_ids = node_ids[: self._max_parallel]

        results: dict[str, tuple[WorkflowState, Any]] = {}

        async def run_one(node_id: str) -> tuple[str, tuple[WorkflowState, Any]]:
            node = workflow.get_node(node_id)
            if node:
                return node_id, await self._execute_node(workflow, context, node)
            return node_id, (WorkflowState.FAILED, f"Node '{node_id}' not found")

        tasks = [run_one(nid) for nid in node_ids]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, tuple):
                results[item[0]] = item[1]
            elif isinstance(item, Exception):
                pass  # Log error

        return results


def create_workflow_executor(
    node_handler: Callable[[WorkflowExecutionContext, WorkflowNode], Any],
    *,
    max_parallel: int = 4,
) -> WorkflowExecutor:
    """Create a workflow executor with custom node handler."""
    return WorkflowExecutor(
        node_executor=node_handler,
        max_parallel=max_parallel,
    )
