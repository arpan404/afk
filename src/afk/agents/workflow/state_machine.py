"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

State machine workflow builder for complex multi-step agentic workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class WorkflowState(str, Enum):
    """States in a workflow state machine."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class WorkflowEvent(str, Enum):
    """Events that transition workflow state."""

    START = "start"
    STEP_COMPLETE = "step_complete"
    STEP_FAILED = "step_failed"
    TOOL_APPROVAL_PENDING = "tool_approval_pending"
    TOOL_APPROVED = "tool_approved"
    TOOL_DENIED = "tool_denied"
    USER_INPUT_REQUIRED = "user_input_required"
    USER_INPUT_RECEIVED = "user_input_received"
    EXTERNAL_CALLBACK = "external_callback"
    TIMEOUT = "timeout"
    CANCEL = "cancel"
    PAUSE = "pause"
    RESUME = "resume"
    RETRY = "retry"


@dataclass(frozen=True, slots=True)
class WorkflowTransition:
    """A single transition in the workflow state machine."""

    from_state: WorkflowState
    event: WorkflowEvent
    to_state: WorkflowState
    action: str | None = None
    condition: str | None = None


@dataclass
class WorkflowNode:
    """
    A single node in the workflow state machine.

    Represents a step/task in the workflow with entry/exit handlers.
    """

    id: str
    name: str
    state: WorkflowState = WorkflowState.PENDING
    max_retries: int = 3
    timeout_s: float | None = None
    on_enter: str | None = None
    on_exit: str | None = None
    on_error: str | None = None
    continue_on_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def can_retry(self) -> bool:
        """Check if this node can be retried."""
        current_retries = self.metadata.get("_retries", 0)
        return current_retries < self.max_retries

    def increment_retry(self) -> None:
        """Increment retry counter."""
        self.metadata["_retries"] = self.metadata.get("_retries", 0) + 1


@dataclass
class WorkflowEdge:
    """
    An edge connecting workflow nodes.

    Represents a dependency or transition between nodes.
    """

    from_node: str
    to_node: str
    condition: str | None = None
    on_error: str | None = None


@dataclass
class WorkflowSpec:
    """
    Complete workflow specification with state machine definition.
    """

    id: str
    name: str
    description: str = ""
    nodes: dict[str, WorkflowNode] = field(default_factory=dict)
    edges: list[WorkflowEdge] = field(default_factory=list)
    initial_node: str | None = None
    final_states: list[WorkflowState] = field(
        default_factory=lambda: [WorkflowState.COMPLETED, WorkflowState.FAILED]
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_node(self, node_id: str) -> WorkflowNode | None:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_outgoing_edges(self, node_id: str) -> list[WorkflowEdge]:
        """Get all outgoing edges from a node."""
        return [e for e in self.edges if e.from_node == node_id]

    def get_incoming_edges(self, node_id: str) -> list[WorkflowEdge]:
        """Get all incoming edges to a node."""
        return [e for e in self.edges if e.to_node == node_id]

    def is_terminal(self, state: WorkflowState) -> bool:
        """Check if state is terminal."""
        return state in self.final_states


class WorkflowBuilder:
    """
    Builder for constructing workflow state machines.

    Provides fluent API for defining workflows with nodes and edges.
    """

    def __init__(self, workflow_id: str, name: str) -> None:
        self._spec = WorkflowSpec(id=workflow_id, name=name)
        self._node_ids: set[str] = set()

    def add_node(
        self,
        node_id: str,
        name: str,
        *,
        timeout_s: float | None = None,
        max_retries: int = 3,
        on_enter: str | None = None,
        on_exit: str | None = None,
        on_error: str | None = None,
        continue_on_error: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowBuilder:
        """Add a node to the workflow."""
        if node_id in self._node_ids:
            raise ValueError(f"Node '{node_id}' already exists")

        self._node_ids.add(node_id)
        self._spec.nodes[node_id] = WorkflowNode(
            id=node_id,
            name=name,
            timeout_s=timeout_s,
            max_retries=max_retries,
            on_enter=on_enter,
            on_exit=on_exit,
            on_error=on_error,
            continue_on_error=continue_on_error,
            metadata=metadata or {},
        )
        return self

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        *,
        condition: str | None = None,
        on_error: str | None = None,
    ) -> WorkflowBuilder:
        """Add an edge between nodes."""
        if from_node not in self._node_ids:
            raise ValueError(f"Source node '{from_node}' does not exist")
        if to_node not in self._node_ids:
            raise ValueError(f"Target node '{to_node}' does not exist")

        self._spec.edges.append(
            WorkflowEdge(
                from_node=from_node,
                to_node=to_node,
                condition=condition,
                on_error=on_error,
            )
        )
        return self

    def set_initial(self, node_id: str) -> WorkflowBuilder:
        """Set the initial node."""
        if node_id not in self._node_ids:
            raise ValueError(f"Node '{node_id}' does not exist")
        self._spec.initial_node = node_id
        return self

    def build(self) -> WorkflowSpec:
        """Build and return the workflow specification."""
        if not self._spec.initial_node:
            raise ValueError("Initial node not set")

        for edge in self._spec.edges:
            from_node = self._spec.get_node(edge.from_node)
            to_node = self._spec.get_node(edge.to_node)
            if from_node is None or to_node is None:
                raise ValueError(f"Invalid edge: {edge.from_node} -> {edge.to_node}")

        return self._spec
