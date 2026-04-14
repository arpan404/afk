"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Property-based testing for agent components.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

from hypothesis import given, settings, example, Phase
from hypothesis import strategies as st

T = TypeVar("T")


@dataclass
class PropertyTestResult:
    """Result of a property-based test run."""

    property_name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    examples_tried: int = 0
    runtime_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "property_name": self.property_name,
            "passed": self.passed,
            "failures": self.failures,
            "examples_tried": self.examples_tried,
            "runtime_s": self.runtime_s,
        }


class AgentPropertyTester:
    """
    Property-based testing utilities for agent components.

    Provides strategies and test helpers for fuzzing agent
    inputs and verifying invariants.
    """

    # Strategies for common agent types
    thread_id_strategy = st.text(min_size=1, max_size=64, alphabet=st.characters(whitelist_categories=["L", "N"]))
    run_id_strategy = st.text(min_size=1, max_size=64, alphabet=st.characters(whitelist_categories=["L", "N"]))
    json_value_strategy = st.one_of(
        st.none(),
        st.booleans(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=1000),
        st.lists(st.nothing()),  # Will be specialized
        st.dictionaries(st.text(), st.nothing()),  # Will be specialized
    )

    @staticmethod
    def tool_args_strategy(tool_name: str) -> st.SearchStrategy:
        """
        Generate random tool arguments based on tool name patterns.

        Args:
            tool_name: Name of the tool to generate args for.

        Returns:
            Hypothesis strategy for tool arguments.
        """
        # Common tool argument patterns
        patterns = {
            "file": st.fixed_dictionaries({
                "path": st.text(min_size=1, max_size=256),
                "content": st.text(max_size=10000),
            }),
            "search": st.fixed_dictionaries({
                "query": st.text(min_size=1, max_size=500),
                "limit": st.integers(min_value=1, max_value=100),
            }),
            "http": st.fixed_dictionaries({
                "url": st.from_domain(str, exclude=[""]),
                "method": st.sampled_from(["GET", "POST", "PUT", "DELETE"]),
                "headers": st.dictionaries(st.text(), st.text()),
            }),
            "database": st.fixed_dictionaries({
                "query": st.text(max_size=2000),
                "params": st.lists(st.one_of(st.text(), st.integers(), st.floats())),
            }),
        }

        # Try to match tool name to pattern
        for key, strategy in patterns.items():
            if key in tool_name.lower():
                return strategy

        # Default: arbitrary dict with string values
        return st.dictionaries(
            st.text(min_size=1, max_size=32),
            st.text(max_size=500),
            max_size=10,
        )

    @staticmethod
    def agent_context_strategy() -> st.SearchStrategy:
        """Generate random agent context dictionaries."""
        return st.dictionaries(
            keys=st.text(min_size=1, max_size=64),
            values=st.one_of(
                st.none(),
                st.booleans(),
                st.floats(),
                st.text(max_size=500),
                st.lists(st.text(max_size=100), max_size=20),
            ),
            max_size=20,
        )

    @staticmethod
    def user_message_strategy() -> st.SearchStrategy:
        """Generate random user messages."""
        return st.text(
            min_size=0,
            max_size=2000,
            alphabet=st.characters(whitelist_categories=["L", "N", "P", "S"]),
        )

    @classmethod
    def memory_event_strategy(cls) -> st.SearchStrategy:
        """Generate random memory events."""
        return st.fixed_dictionaries({
            "id": st.text(min_size=1, max_size=64),
            "thread_id": cls.thread_id_strategy,
            "user_id": st.one_of(st.none(), st.text(min_size=1, max_size=64)),
            "type": st.sampled_from(["tool_call", "tool_result", "message", "system", "trace"]),
            "timestamp": st.integers(min_value=0, max_value=2000000000000),
            "payload": st.dictionaries(st.text(), cls.json_value_strategy),
            "tags": st.lists(st.text(max_size=32), max_size=10),
        })

    @classmethod
    def agent_result_strategy(cls) -> st.SearchStrategy:
        """Generate random agent results."""
        return st.fixed_dictionaries({
            "run_id": st.text(min_size=1, max_size=64),
            "thread_id": st.text(min_size=1, max_size=64),
            "state": st.sampled_from(["completed", "failed", "degraded"]),
            "final_text": st.text(max_size=5000),
            "total_cost_usd": st.floats(min_value=0.0, max_value=100.0),
        })


# Helper functions for writing property tests

def test_memory_importance_invariant(
    event_type: str,
    has_tool_failure: bool,
    has_user_correction: bool,
) -> bool:
    """
    Test invariant: importance scores should be higher for important events.

    This is a helper for writing property-based tests.
    """
    from ..agents.types.memory import compute_event_importance

    score = compute_event_importance(
        event_type=event_type,
        has_tool_failure=has_tool_failure,
        has_user_correction=has_user_correction,
    )

    # Tool failures should always be high importance
    if has_tool_failure:
        assert score >= 0.8, f"Tool failure should have high importance, got {score}"

    # User corrections should be high importance
    if has_user_correction:
        assert score >= 0.8, f"User correction should have high importance, got {score}"

    # Score should always be in [0, 1]
    assert 0.0 <= score <= 1.0, f"Importance score out of range: {score}"

    return True


def test_degraded_result_invariant(
    confidence: float,
    partial_text: str,
    degraded_steps: int,
    max_steps: int,
) -> bool:
    """
    Test invariant: DegradedResult should be consistent.

    Args:
        confidence: Confidence score (0.0 to 1.0).
        partial_text: Partial text output.
        degraded_steps: Steps completed before degradation.
        max_steps: Maximum steps allowed.

    Returns:
        True if invariants hold.
    """
    from ..agents.types.degraded import DegradedResult

    result = DegradedResult(
        confidence=confidence,
        partial_text=partial_text,
        degraded_steps=degraded_steps,
        max_steps=max_steps,
    )

    # Confidence should always be in [0, 1]
    assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range: {confidence}"

    # Degraded steps should never exceed max steps
    assert result.degraded_steps <= result.max_steps, "Degraded steps exceed max"

    # is_usable should be consistent
    if result.is_usable():
        assert result.confidence >= 0.5, "Usable result should have confidence >= 0.5"
        assert bool(result.partial_text), "Usable result should have text"

    return True


def test_checkpoint_replay_invariants(
    step: int,
    checkpoint_data: dict[str, Any],
) -> bool:
    """
    Test invariants for checkpoint replay.

    Args:
        step: Step number.
        checkpoint_data: Checkpoint data dictionary.

    Returns:
        True if invariants hold.
    """
    if checkpoint_data:
        # Step in checkpoint should match requested step
        if "step" in checkpoint_data:
            assert checkpoint_data["step"] == step, f"Checkpoint step mismatch: {checkpoint_data['step']} != {step}"

        # State should be valid
        valid_states = ["pending", "running", "paused", "cancelling", "completed", "failed", "degraded"]
        if "state" in checkpoint_data:
            assert checkpoint_data["state"] in valid_states, f"Invalid state: {checkpoint_data['state']}"

    return True


# Export Hypothesis strategies for use in test files
__all__ = [
    "AgentPropertyTester",
    "PropertyTestResult",
    "test_memory_importance_invariant",
    "test_degraded_result_invariant",
    "test_checkpoint_replay_invariants",
]