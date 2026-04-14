"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Tests for workflow state machine.
"""

import asyncio

import pytest
from afk.agents.workflow.state_machine import (
    WorkflowBuilder,
    WorkflowEdge,
    WorkflowEvent,
    WorkflowNode,
    WorkflowSpec,
    WorkflowState,
    WorkflowTransition,
)
from afk.agents.workflow.executor import (
    WorkflowExecutionContext,
    WorkflowExecutionResult,
    WorkflowExecutor,
    create_workflow_executor,
)


class TestWorkflowBuilder:
    """Tests for WorkflowBuilder."""

    def test_add_node(self) -> None:
        """Test adding nodes to workflow."""
        builder = WorkflowBuilder("test", "Test Workflow")
        builder.add_node("start", "Start Node", timeout_s=30.0)

        spec = builder.set_initial("start").build()

        assert spec.id == "test"
        assert "start" in spec.nodes
        assert spec.nodes["start"].timeout_s == 30.0

    def test_add_edge(self) -> None:
        """Test adding edges between nodes."""
        builder = WorkflowBuilder("test", "Test Workflow")
        builder.add_node("a", "Node A")
        builder.add_node("b", "Node B")
        builder.add_edge("a", "b")
        builder.set_initial("a")

        spec = builder.build()

        edges = spec.get_outgoing_edges("a")
        assert len(edges) == 1
        assert edges[0].to_node == "b"

    def test_duplicate_node_error(self) -> None:
        """Test that duplicate nodes raise error."""
        builder = WorkflowBuilder("test", "Test Workflow")
        builder.add_node("a", "Node A")

        with pytest.raises(ValueError, match="already exists"):
            builder.add_node("a", "Node A")

    def test_missing_node_error(self) -> None:
        """Test that missing nodes raise error."""
        builder = WorkflowBuilder("test", "Test Workflow")
        builder.add_node("a", "Node A")

        with pytest.raises(ValueError, match="does not exist"):
            builder.add_edge("a", "b")

    def test_no_initial_error(self) -> None:
        """Test that missing initial node raises error."""
        builder = WorkflowBuilder("test", "Test Workflow")
        builder.add_node("a", "Node A")

        with pytest.raises(ValueError, match="Initial node not set"):
            builder.build()


class TestWorkflowNode:
    """Tests for WorkflowNode."""

    def test_can_retry(self) -> None:
        """Test retry capability."""
        node = WorkflowNode("test", "Test", max_retries=3)

        assert node.can_retry() is True

        node.increment_retry()
        node.increment_retry()
        node.increment_retry()

        assert node.can_retry() is False

    def test_increment_retry(self) -> None:
        """Test retry increment."""
        node = WorkflowNode("test", "Test", max_retries=5, metadata={})

        node.increment_retry()
        node.increment_retry()

        assert node.metadata["_retries"] == 2


class TestWorkflowExecutionContext:
    """Tests for WorkflowExecutionContext."""

    def test_get_node_state(self) -> None:
        """Test getting node state."""
        ctx = WorkflowExecutionContext(
            workflow_id="test",
            run_id="run-1",
            thread_id="thread-1",
        )

        state = ctx.get_node_state("node-a")
        assert state == WorkflowState.PENDING

    def test_set_node_state(self) -> None:
        """Test setting node state."""
        ctx = WorkflowExecutionContext(
            workflow_id="test",
            run_id="run-1",
            thread_id="thread-1",
        )

        ctx.set_node_state("node-a", WorkflowState.RUNNING)

        state = ctx.get_node_state("node-a")
        assert state == WorkflowState.RUNNING


class TestWorkflowExecutor:
    """Tests for WorkflowExecutor."""

    @pytest.mark.asyncio
    async def test_execute_simple_workflow(self) -> None:
        """Test executing simple workflow."""
        spec = WorkflowSpec(
            id="test",
            name="Test",
            nodes={
                "start": WorkflowNode(id="start", name="Start"),
            },
            initial_node="start",
        )

        executor = WorkflowExecutor()
        ctx = WorkflowExecutionContext(
            workflow_id="test",
            run_id="run-1",
            thread_id="thread-1",
        )

        result = await executor.execute(spec, ctx)

        assert result.workflow_id == "test"
        assert "start" in result.node_states

    @pytest.mark.asyncio
    async def test_executor_creation(self) -> None:
        """Test creating executor."""
        executor = WorkflowExecutor()
        assert executor._max_parallel == 4

        executor2 = create_workflow_executor(
            lambda ctx, node: (WorkflowState.COMPLETED, None), max_parallel=2
        )
        assert executor2._max_parallel == 2

    @pytest.mark.asyncio
    async def test_parallel_create(self) -> None:
        """Test creating parallel executor."""
        executor = WorkflowExecutor(max_parallel=8)
        assert executor._max_parallel == 8


class TestWorkflowSpec:
    """Tests for WorkflowSpec."""

    def test_get_node(self) -> None:
        """Test getting node."""
        spec = WorkflowSpec(
            id="test",
            name="Test",
            nodes={"a": WorkflowNode(id="a", name="A")},
        )

        node = spec.get_node("a")
        assert node is not None
        assert node.name == "A"

    def test_get_outgoing_edges(self) -> None:
        """Test getting outgoing edges."""
        spec = WorkflowSpec(
            id="test",
            name="Test",
            nodes={
                "a": WorkflowNode(id="a", name="A"),
                "b": WorkflowNode(id="b", name="B"),
            },
            edges=[WorkflowEdge(from_node="a", to_node="b")],
        )

        edges = spec.get_outgoing_edges("a")
        assert len(edges) == 1

    def test_get_incoming_edges(self) -> None:
        """Test getting incoming edges."""
        spec = WorkflowSpec(
            id="test",
            name="Test",
            nodes={
                "a": WorkflowNode(id="a", name="A"),
                "b": WorkflowNode(id="b", name="B"),
            },
            edges=[WorkflowEdge(from_node="a", to_node="b")],
        )

        edges = spec.get_incoming_edges("b")
        assert len(edges) == 1

    def test_is_terminal(self) -> None:
        """Test terminal state check."""
        spec = WorkflowSpec(
            id="test",
            name="Test",
            final_states=[WorkflowState.COMPLETED, WorkflowState.FAILED],
        )

        assert spec.is_terminal(WorkflowState.COMPLETED) is True
        assert spec.is_terminal(WorkflowState.FAILED) is True
        assert spec.is_terminal(WorkflowState.RUNNING) is False
