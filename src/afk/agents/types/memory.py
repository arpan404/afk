"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Memory importance scoring and prioritization types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import AgentState


@dataclass(frozen=True, slots=True)
class ScoredMemoryEvent:
    """
    A memory event with an importance score for prioritization.

    Attributes:
        event_id: Original event identifier.
        thread_id: Thread this event belongs to.
        importance: Importance score from 0.0 (lowest) to 1.0 (highest).
        event_type: Type of the original event.
        timestamp_ms: Event timestamp in milliseconds.
        summary: Auto-generated summary of the event content.
        tags: Tags from the original event.
        should_compact: Whether this event should be compacted under memory pressure.
        payload_preview: Brief preview of the event payload.
    """

    event_id: str
    thread_id: str
    importance: float
    event_type: str
    timestamp_ms: int
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    should_compact: bool = False
    payload_preview: str = ""


def compute_event_importance(
    event_type: str,
    *,
    has_tool_failure: bool = False,
    has_user_correction: bool = False,
    is_goal_milestone: bool = False,
    payload_size: int = 0,
    has_error: bool = False,
) -> float:
    """
    Compute importance score for a memory event.

    Higher scores indicate more important events that should be
    retained even under memory pressure.

    Args:
        event_type: Type of event (tool_call, tool_result, message, etc.).
        has_tool_failure: Whether a tool call failed.
        has_user_correction: Whether user corrected the agent.
        is_goal_milestone: Whether this event marks goal completion.
        payload_size: Size of event payload in bytes.
        has_error: Whether any error occurred.

    Returns:
        Importance score from 0.0 to 1.0.
    """
    score = 0.5  # Base score

    # Event type weights
    type_weights = {
        "tool_call": 0.6,
        "tool_result": 0.5,
        "message": 0.4,
        "system": 0.7,
        "trace": 0.3,
    }
    score = type_weights.get(event_type, 0.5)

    # Tool failures are high value for debugging
    if has_tool_failure:
        score = max(score, 0.9)

    # User corrections indicate learning opportunities
    if has_user_correction:
        score = max(score, 0.85)

    # Goal milestones are critical
    if is_goal_milestone:
        score = max(score, 0.95)

    # Errors are important to retain
    if has_error:
        score = max(score, 0.8)

    # Large payloads may contain important context
    if payload_size > 5000:
        score = min(score + 0.1, 1.0)

    return round(score, 3)


def should_compact_event(importance: float, memory_pressure: float = 0.5) -> bool:
    """
    Determine if an event should be compacted based on importance and pressure.

    Args:
        importance: Event importance score (0.0 to 1.0).
        memory_pressure: Memory pressure level (0.0 to 1.0), higher = more pressure.

    Returns:
        True if event should be compacted/summarized.
    """
    # Events with importance below the memory pressure threshold should compact
    return importance < memory_pressure


@dataclass(frozen=True, slots=True)
class Episode:
    """
    A consolidated memory episode containing summarized events.

    Episodes are created during memory consolidation to reduce the
    storage footprint of long-running conversations while preserving
    the essential information.

    Attributes:
        id: Unique episode identifier.
        thread_id: Thread this episode belongs to.
        start_ms: Start timestamp in milliseconds.
        end_ms: End timestamp in milliseconds.
        summary: LLM-generated summary of what happened in this episode.
        event_count: Number of original events consolidated.
        importance: Average importance of constituent events.
        tags: Combined tags from constituent events.
        metadata: Additional episode metadata.
    """

    id: str
    thread_id: str
    start_ms: int
    end_ms: int
    summary: str
    event_count: int = 0
    importance: float = 0.5
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidationResult:
    """
    Result of a memory consolidation operation.

    Attributes:
        episodes_created: Number of new episodes created.
        events_compacted: Number of events that were summarized into episodes.
        events_dropped: Number of low-importance events that were dropped.
        memory_saved_bytes: Estimated bytes saved by consolidation.
        consolidation_time_ms: Time taken to complete consolidation.
    """

    episodes_created: int = 0
    events_compacted: int = 0
    events_dropped: int = 0
    memory_saved_bytes: int = 0
    consolidation_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes_created": self.episodes_created,
            "events_compacted": self.events_compacted,
            "events_dropped": self.events_dropped,
            "memory_saved_bytes": self.memory_saved_bytes,
            "consolidation_time_ms": self.consolidation_time_ms,
        }