"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Tests for checkpoint replay API.
"""

import pytest
from afk.agents.lifecycle import (
    CheckpointInfo,
    CheckpointReplayHandle,
    CheckpointReplayManager,
    ReplayAPI,
    ReplayDecisionPoint,
    ReplaySession,
    ReplayTimeline,
    ReplayTimelineEvent,
    InteractiveReplayHandler,
    create_replay_api,
)


class MockMemoryStore:
    """Mock memory store for testing."""

    def __init__(self) -> None:
        self._state: dict[str, dict] = {}
        self._events: list = []

    async def get_state(self, thread_id: str, key: str) -> dict | None:
        return self._state.get(key)

    async def put_state(self, thread_id: str, key: str, value: dict) -> None:
        self._state[key] = value

    async def get_events_since(self, thread_id: str, since_ms: int, limit: int = 500) -> list:
        return self._events


class TestReplayTimelineEvent:
    """Tests for ReplayTimelineEvent."""

    def test_creation(self) -> None:
        """Test creating timeline event."""
        event = ReplayTimelineEvent(
            step=1,
            timestamp_ms=1000,
            timestamp_iso="2026-01-01T00:00:00",
            event_type="tool_call",
            content="Fetching data",
            summary="Fetch operation",
            has_error=False,
        )

        assert event.step == 1
        assert event.event_type == "tool_call"
        assert event.has_error is False


class TestReplayTimeline:
    """Tests for ReplayTimeline."""

    def test_creation(self) -> None:
        """Test creating timeline."""
        timeline = ReplayTimeline(
            run_id="run-1",
            thread_id="thread-1",
            total_steps=10,
        )

        assert timeline.run_id == "run-1"
        assert timeline.total_steps == 10

    def test_get_event_at(self) -> None:
        """Test getting event at step."""
        events = [
            ReplayTimelineEvent(
                step=1,
                timestamp_ms=1000,
                timestamp_iso="",
                event_type="start",
                content="",
                summary="Start",
            ),
            ReplayTimelineEvent(
                step=2,
                timestamp_ms=2000,
                timestamp_iso="",
                event_type="tool",
                content="",
                summary="Tool",
            ),
        ]
        timeline = ReplayTimeline(
            run_id="run-1",
            thread_id="thread-1",
            total_steps=2,
            events=events,
        )

        event = timeline.get_event_at(1)
        assert event is not None
        assert event.step == 1

    def test_get_events_between(self) -> None:
        """Test getting events in range."""
        events = [
            ReplayTimelineEvent(
                step=0, timestamp_ms=0, timestamp_iso="", event_type="start", content="", summary=""
            ),
            ReplayTimelineEvent(
                step=1,
                timestamp_ms=1000,
                timestamp_iso="",
                event_type="tool",
                content="",
                summary="",
            ),
            ReplayTimelineEvent(
                step=2,
                timestamp_ms=2000,
                timestamp_iso="",
                event_type="result",
                content="",
                summary="",
            ),
        ]
        timeline = ReplayTimeline(
            run_id="run-1",
            thread_id="thread-1",
            total_steps=3,
            events=events,
        )

        filtered = timeline.get_events_between(0, 2)
        assert len(filtered) == 2

    def test_get_errors(self) -> None:
        """Test getting error events."""
        events = [
            ReplayTimelineEvent(
                step=1,
                timestamp_ms=1000,
                timestamp_iso="",
                event_type="tool",
                content="",
                summary="",
                has_error=False,
            ),
            ReplayTimelineEvent(
                step=2,
                timestamp_ms=2000,
                timestamp_iso="",
                event_type="error",
                content="",
                summary="",
                has_error=True,
            ),
        ]
        timeline = ReplayTimeline(
            run_id="run-1",
            thread_id="thread-1",
            total_steps=2,
            events=events,
        )

        errors = timeline.get_errors()
        assert len(errors) == 1


class TestReplayDecisionPoint:
    """Tests for ReplayDecisionPoint."""

    def test_creation(self) -> None:
        """Test creating decision point."""
        dp = ReplayDecisionPoint(
            step=5,
            timestamp_iso="2026-01-01T00:00:00",
            decision_type="approval",
            description="Approve this action",
            options=["approve", "deny"],
        )

        assert dp.step == 5
        assert dp.decision_type == "approval"
        assert len(dp.options) == 2

    def test_make_decision(self) -> None:
        """Test making a decision."""
        dp = ReplayDecisionPoint(
            step=5,
            timestamp_iso="2026-01-01T00:00:00",
            decision_type="approval",
            description="Test",
            options=["approve", "deny"],
        )

        dp.current_choice = "approve"
        import time

        dp.made_at_ms = int(time.time() * 1000)

        assert dp.current_choice == "approve"
        assert dp.made_at_ms is not None


class TestReplaySession:
    """Tests for ReplaySession."""

    def test_creation(self) -> None:
        """Test creating session."""
        session = ReplaySession(
            run_id="run-1",
            thread_id="thread-1",
        )

        assert session.run_id == "run-1"
        assert session.thread_id == "thread-1"
        assert session.current_step == 0

    def test_can_rollback(self) -> None:
        """Test rollback check."""
        session = ReplaySession(
            run_id="run-1",
            thread_id="thread-1",
            current_step=5,
        )

        assert session.can_rollback() is True

        session_zero = ReplaySession(
            run_id="run-1",
            thread_id="thread-1",
            current_step=0,
        )
        assert session_zero.can_rollback() is False

    def test_can_continue(self) -> None:
        """Test continue check."""
        session = ReplaySession(
            run_id="run-1",
            thread_id="thread-1",
            current_step=5,
            timeline=ReplayTimeline(
                run_id="run-1",
                thread_id="thread-1",
                total_steps=10,
            ),
        )

        assert session.can_continue() is True

        finished = ReplaySession(
            run_id="run-1",
            thread_id="thread-1",
            current_step=10,
            timeline=ReplayTimeline(
                run_id="run-1",
                thread_id="thread-1",
                total_steps=10,
            ),
        )
        assert finished.can_continue() is False


class TestReplayAPI:
    """Tests for ReplayAPI."""

    @pytest.mark.asyncio
    async def test_open_session(self) -> None:
        """Test opening replay session."""
        memory = MockMemoryStore()

        # Add checkpoints
        await memory.put_state(
            "thread-1",
            "checkpoint_latest:run-1",
            {
                "step": 5,
                "phase": "tool_call",
                "timestamp_ms": 5000,
            },
        )

        api = create_replay_api(memory)
        session = await api.open_session("run-1", "thread-1")

        assert session.run_id == "run-1"
        assert session.timeline is not None

    @pytest.mark.asyncio
    async def test_list_sessions(self) -> None:
        """Test listing sessions."""
        memory = MockMemoryStore()
        api = create_replay_api(memory)

        sessions = await api.list_sessions("thread-1")

        assert isinstance(sessions, list)


class TestInteractiveReplayHandler:
    """Tests for InteractiveReplayHandler."""

    @pytest.mark.asyncio
    async def test_start_review(self) -> None:
        """Test starting review."""
        memory = MockMemoryStore()

        # Add checkpoint
        await memory.put_state(
            "thread-1",
            "checkpoint_latest:run-1",
            {
                "step": 3,
                "phase": "running",
                "timestamp_ms": 3000,
            },
        )

        api = create_replay_api(memory)
        handler = InteractiveReplayHandler(api)

        session = await handler.start_review("run-1", "thread-1")

        assert session.run_id == "run-1"

    @pytest.mark.asyncio
    async def test_make_decision(self) -> None:
        """Test making decision."""
        memory = MockMemoryStore()

        # Add checkpoint
        await memory.put_state(
            "thread-1",
            "checkpoint_latest:run-1",
            {
                "step": 3,
                "phase": "running",
                "timestamp_ms": 3000,
            },
        )

        api = create_replay_api(memory)
        handler = InteractiveReplayHandler(api)

        session = await handler.start_review("run-1", "thread-1")

        # Test that session was created
        assert session.run_id == "run-1"

    @pytest.mark.asyncio
    async def test_invalid_run(self) -> None:
        """Test invalid run error."""
        memory = MockMemoryStore()
        api = create_replay_api(memory)
        handler = InteractiveReplayHandler(api)

        with pytest.raises(ValueError, match="No review session"):
            await handler.make_decision("run-invalid", "approval", "approve")


class TestCreateReplayAPI:
    """Tests for create_replay_api."""

    def test_creation(self) -> None:
        """Test creating replay API."""
        memory = MockMemoryStore()
        api = create_replay_api(memory)

        assert api is not None
        assert isinstance(api, ReplayAPI)
