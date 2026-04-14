"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Checkpoint Replay API for human-in-loop review and debugging.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from ..errors import AgentCheckpointCorruptionError
from .replay import (
    CheckpointInfo,
    CheckpointReplayHandle,
    CheckpointReplayManager,
)


@dataclass
class ReplayTimelineEvent:
    """A single event in the replay timeline."""

    step: int
    timestamp_ms: int
    timestamp_iso: str
    event_type: str
    content: str
    summary: str
    has_error: bool = False


@dataclass
class ReplayTimeline:
    """Timeline of events for review."""

    run_id: str
    thread_id: str
    total_steps: int
    events: list[ReplayTimelineEvent] = field(default_factory=list)

    def get_event_at(self, step: int) -> ReplayTimelineEvent | None:
        """Get event at specific step."""
        for event in self.events:
            if event.step == step:
                return event
        return None

    def get_events_between(
        self,
        start_step: int,
        end_step: int,
    ) -> list[ReplayTimelineEvent]:
        """Get events in range [start, end)."""
        return [e for e in self.events if start_step <= e.step < end_step]

    def get_errors(self) -> list[ReplayTimelineEvent]:
        """Get all events with errors."""
        return [e for e in self.events if e.has_error]


@dataclass
class ReplayDecisionPoint:
    """A point where human decision is needed."""

    step: int
    timestamp_iso: str
    decision_type: str  # approval, input, rollback, etc.
    description: str
    options: list[str] = field(default_factory=list)
    current_choice: str | None = None
    made_at_ms: int | None = None


@dataclass
class ReplaySession:
    """
    Complete replay session for human review.

    Provides timeline navigation, decision points,
    and continuation controls.
    """

    run_id: str
    thread_id: str
    timeline: ReplayTimeline | None = None
    decision_points: list[ReplayDecisionPoint] = field(default_factory=list)
    current_step: int = 0
    started_at_ms: int = 0
    finished_at_ms: int | None = None
    resumed_from_step: int | None = None

    def can_rollback(self) -> bool:
        """Check if rollback is possible."""
        return self.current_step > 0

    def can_continue(self) -> bool:
        """Check if continuation is possible."""
        return self.current_step < self.timeline.total_steps


class ReplayAPI:
    """
    First-class API for checkpoint replay and human-in-loop review.

    Provides:
    - Timeline view of entire run
    - Decision point identification
    - Rollback to any step
    - Continuation from checkpoint
    """

    def __init__(self, memory_store: Any) -> None:
        self._manager = CheckpointReplayManager(memory_store)

    async def open_session(
        self,
        run_id: str,
        thread_id: str,
        target_step: int | None = None,
    ) -> ReplaySession:
        """
        Open a replay session.

        Args:
            run_id: Run to replay.
            thread_id: Thread ID.
            target_step: Step to stop at (None = latest).

        Returns:
            ReplaySession for review.
        """
        handle = await self._manager.open_replay(run_id, thread_id)

        latest = await self._manager.get_latest_checkpoint(run_id, thread_id)
        total_steps = latest.step if latest else 0

        timeline = await self._build_timeline(handle)
        decision_points = self._identify_decision_points(timeline)

        target = target_step or total_steps

        return ReplaySession(
            run_id=run_id,
            thread_id=thread_id,
            timeline=timeline,
            decision_points=decision_points,
            current_step=target,
            started_at_ms=int(time.time() * 1000),
            resumed_from_step=target if target < total_steps else None,
        )

    async def get_timeline(
        self,
        run_id: str,
        thread_id: str,
    ) -> ReplayTimeline:
        """Get timeline for a run."""
        handle = await self._manager.open_replay(run_id, thread_id)
        return await self._build_timeline(handle)

    async def get_step_snapshot(
        self,
        run_id: str,
        thread_id: str,
        step: int,
    ) -> dict[str, Any]:
        """
        Get runtime snapshot at a step.

        Args:
            run_id: Run ID.
            thread_id: Thread ID.
            step: Step number.

        Returns:
            Snapshot dict with runtime state.
        """
        handle = await self._manager.open_replay(run_id, thread_id)

        checkpoint = await handle.get_checkpoint_at_step(step)
        if not checkpoint:
            raise AgentCheckpointCorruptionError(f"No checkpoint at step {step}")

        snapshot = await handle.get_replay_snapshot(step)
        return snapshot

    async def rollback(
        self,
        session: ReplaySession,
        target_step: int,
    ) -> ReplaySession:
        """
        Rollback session to target step.

        Args:
            session: Current session.
            target_step: Step to rollback to.

        Returns:
            New session from target step.
        """
        if target_step < 0 or target_step > session.current_step:
            raise ValueError(
                f"Invalid rollback step {target_step}, must be 0-{session.current_step}"
            )

        return ReplaySession(
            run_id=session.run_id,
            thread_id=session.thread_id,
            timeline=session.timeline,
            decision_points=session.decision_points,
            current_step=target_step,
            started_at_ms=session.started_at_ms,
            resumed_from_step=target_step,
        )

    async def list_sessions(
        self,
        thread_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        List all runs for a thread.

        Args:
            thread_id: Thread ID.
            limit: Max results.

        Returns:
            List of run summaries.
        """
        # Would need to query memory store for all runs
        # This is simplified
        return []

    async def _build_timeline(
        self,
        handle: CheckpointReplayHandle,
    ) -> ReplayTimeline:
        """Build timeline from replay handle."""
        events = []

        checkpoints = await handle.list_checkpoints()
        for ckpt in checkpoints:
            step = ckpt.step

            # Get event data
            data = await handle.get_checkpoint_at_step(step)

            event_type = data.get("phase", "step") if data else "unknown"
            content = str(data.get("summary", "")) if data else ""
            has_error = bool(data.get("error")) if data else False

            events.append(
                ReplayTimelineEvent(
                    step=step,
                    timestamp_ms=ckpt.timestamp_ms,
                    timestamp_iso=datetime.fromtimestamp(ckpt.timestamp_ms / 1000).isoformat(),
                    event_type=event_type,
                    content=content,
                    summary=data.get("summary", f"Step {step}") if data else f"Step {step}",
                    has_error=has_error,
                )
            )

        return ReplayTimeline(
            run_id=handle.run_id,
            thread_id=handle.thread_id,
            total_steps=len(events),
            events=events,
        )

    def _identify_decision_points(
        self,
        timeline: ReplayTimeline,
    ) -> list[ReplayDecisionPoint]:
        """Identify decision points in timeline."""
        decisions = []

        for event in timeline.events:
            if event.event_type == "approval_pending":
                decisions.append(
                    ReplayDecisionPoint(
                        step=event.step,
                        timestamp_iso=event.timestamp_iso,
                        decision_type="approval",
                        description=event.summary,
                    )
                )
            elif event.event_type == "user_input":
                decisions.append(
                    ReplayDecisionPoint(
                        step=event.step,
                        timestamp_iso=event.timestamp_iso,
                        decision_type="input",
                        description=event.summary,
                    )
                )

        return decisions


class InteractiveReplayHandler:
    """
    Handler for interactive replay with human decisions.

    Integrates with runner to allow human-in-loop control.
    """

    def __init__(self, replay_api: ReplayAPI) -> None:
        self._api = replay_api
        self._pending_decisions: dict[str, ReplaySession] = {}

    async def start_review(
        self,
        run_id: str,
        thread_id: str,
    ) -> ReplaySession:
        """Start a review session."""
        session = await self._api.open_session(run_id, thread_id)
        self._pending_decisions[run_id] = session
        return session

    async def make_decision(
        self,
        run_id: str,
        decision: str,
        choice: str,
    ) -> ReplaySession:
        """
        Make a decision at a decision point.

        Args:
            run_id: Run ID.
            decision: Decision type.
            choice: Selected choice.

        Returns:
            Updated session.
        """
        session = self._pending_decisions.get(run_id)
        if not session:
            raise ValueError(f"No review session for run {run_id}")

        # Update decision
        for dp in session.decision_points:
            if dp.decision_type == decision and dp.current_choice is None:
                dp.current_choice = choice
                dp.made_at_ms = int(time.time() * 1000)
                break

        return session

    async def continue_from(
        self,
        run_id: str,
        from_step: int | None = None,
    ) -> dict[str, Any]:
        """
        Get state to continue execution.

        Args:
            run_id: Run ID.
            from_step: Step to continue from (None = current).

        Returns:
            Runtime state dict.
        """
        session = self._pending_decisions.get(run_id)
        if not session:
            raise ValueError(f"No review session for run {run_id}")

        step = from_step or session.current_step
        return await self._api.get_step_snapshot(
            session.run_id,
            session.thread_id,
            step,
        )


def create_replay_api(memory_store: Any) -> ReplayAPI:
    """Create a ReplayAPI instance."""
    return ReplayAPI(memory_store)
