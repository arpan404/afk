"""
Comprehensive tests for afk.core.streaming and afk.core.telemetry.
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from afk.agents.types import AgentResult
from afk.core.streaming import (
    AgentStreamEvent,
    AgentStreamHandle,
    status_update,
    step_completed,
    step_started,
    stream_completed,
    stream_error,
    text_delta,
    tool_completed,
    tool_deferred,
    tool_started,
)
from afk.core.telemetry import TelemetryEvent, TelemetrySpan

# ======================================================================
# AgentStreamEvent dataclass
# ======================================================================


class TestAgentStreamEventDataclass:
    """Tests for the AgentStreamEvent frozen dataclass."""

    def test_type_field_is_required(self):
        """Constructing without 'type' must raise TypeError."""
        with pytest.raises(TypeError):
            AgentStreamEvent()  # type: ignore[call-arg]

    def test_optional_fields_default_to_none(self):
        event = AgentStreamEvent(type="text_delta")
        assert event.text_delta is None
        assert event.tool_name is None
        assert event.tool_call_id is None
        assert event.tool_success is None
        assert event.tool_output is None
        assert event.tool_error is None
        assert event.tool_ticket_id is None
        assert event.step is None
        assert event.state is None
        assert event.run_event is None
        assert event.result is None
        assert event.error is None

    def test_data_defaults_to_empty_dict(self):
        event = AgentStreamEvent(type="text_delta")
        assert event.data == {}

    def test_frozen(self):
        event = AgentStreamEvent(type="text_delta")
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.type = "error"  # type: ignore[misc]

    def test_slots(self):
        assert AgentStreamEvent.__dataclass_params__.slots is True

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(AgentStreamEvent)


# ======================================================================
# Convenience constructors
# ======================================================================


class TestTextDelta:
    """Tests for the text_delta convenience constructor."""

    def test_text_delta_with_step(self):
        event = text_delta("hi", step=1)
        assert event.type == "text_delta"
        assert event.text_delta == "hi"
        assert event.step == 1

    def test_text_delta_without_step(self):
        event = text_delta("hi")
        assert event.type == "text_delta"
        assert event.text_delta == "hi"
        assert event.step is None


class TestToolStarted:
    """Tests for the tool_started convenience constructor."""

    def test_tool_started_full(self):
        event = tool_started("my_tool", "call_1", step=2)
        assert event.type == "tool_started"
        assert event.tool_name == "my_tool"
        assert event.tool_call_id == "call_1"
        assert event.step == 2

    def test_tool_started_minimal(self):
        event = tool_started("t")
        assert event.type == "tool_started"
        assert event.tool_name == "t"
        assert event.tool_call_id is None
        assert event.step is None


class TestToolCompleted:
    """Tests for the tool_completed convenience constructor."""

    def test_tool_completed_all_fields(self):
        event = tool_completed(
            "t", "c1", success=True, output={"x": 1}, error=None, step=3
        )
        assert event.type == "tool_completed"
        assert event.tool_name == "t"
        assert event.tool_call_id == "c1"
        assert event.tool_success is True
        assert event.tool_output == {"x": 1}
        assert event.tool_error is None
        assert event.step == 3

    def test_tool_completed_failure(self):
        event = tool_completed("t", success=False, error="boom")
        assert event.type == "tool_completed"
        assert event.tool_success is False
        assert event.tool_error == "boom"


class TestToolDeferred:
    """Tests for the tool_deferred convenience constructor."""

    def test_tool_deferred_full(self):
        event = tool_deferred(
            "t", "c1", ticket_id="tkt-1", step=4, data={"key": "val"}
        )
        assert event.type == "tool_deferred"
        assert event.tool_name == "t"
        assert event.tool_call_id == "c1"
        assert event.tool_ticket_id == "tkt-1"
        assert event.step == 4
        assert event.data == {"key": "val"}

    def test_tool_deferred_minimal(self):
        event = tool_deferred("t")
        assert event.type == "tool_deferred"
        assert event.tool_name == "t"
        assert event.tool_ticket_id is None
        assert event.step is None
        assert event.data == {}


class TestStepStarted:
    """Tests for the step_started convenience constructor."""

    def test_step_started(self):
        event = step_started(5, "running")
        assert event.type == "step_started"
        assert event.step == 5
        assert event.state == "running"


class TestStepCompleted:
    """Tests for the step_completed convenience constructor."""

    def test_step_completed(self):
        event = step_completed(5, "running")
        assert event.type == "step_completed"
        assert event.step == 5
        assert event.state == "running"


class TestStatusUpdate:
    """Tests for the status_update convenience constructor."""

    def test_status_update_full(self):
        event = status_update("paused", step=3, data={"reason": "waiting"})
        assert event.type == "status_update"
        assert event.state == "paused"
        assert event.step == 3
        assert event.data == {"reason": "waiting"}

    def test_status_update_minimal(self):
        event = status_update("running")
        assert event.type == "status_update"
        assert event.state == "running"
        assert event.step is None
        assert event.data == {}


class TestStreamError:
    """Tests for the stream_error convenience constructor."""

    def test_stream_error(self):
        event = stream_error("oops")
        assert event.type == "error"
        assert event.error == "oops"


# ======================================================================
# AgentStreamHandle
# ======================================================================


class TestAgentStreamHandle:
    """Tests for AgentStreamHandle async streaming handle."""

    def test_emit_stores_events_in_queue(self):
        async def scenario():
            handle = AgentStreamHandle()
            ev = text_delta("chunk")
            await handle.emit(ev)
            item = await handle._queue.get()
            assert item is ev

        asyncio.run(scenario())

    def test_close_signals_end_of_stream(self):
        async def scenario():
            handle = AgentStreamHandle()
            assert handle.done is False
            await handle.close()
            assert handle.done is True

        asyncio.run(scenario())

    def test_aiter_yields_events_until_close(self):
        async def scenario():
            handle = AgentStreamHandle()
            ev1 = text_delta("a")
            ev2 = text_delta("b")
            await handle.emit(ev1)
            await handle.emit(ev2)
            await handle.close()

            collected = []
            async for event in handle:
                collected.append(event)

            assert len(collected) == 2
            assert collected[0] is ev1
            assert collected[1] is ev2

        asyncio.run(scenario())

    def test_result_returns_none_before_stream_completed(self):
        async def scenario():
            handle = AgentStreamHandle()
            assert handle.result is None
            await handle.emit(text_delta("hello"))
            assert handle.result is None

        asyncio.run(scenario())

    def test_result_returns_agent_result_after_stream_completed(self):
        async def scenario():
            handle = AgentStreamHandle()
            agent_result = AgentResult(
                run_id="r1",
                thread_id="t1",
                state="completed",
                final_text="done",
            )
            completed_event = stream_completed(agent_result)
            await handle.emit(completed_event)
            assert handle.result is agent_result

        asyncio.run(scenario())

    def test_done_property_reflects_close_state(self):
        async def scenario():
            handle = AgentStreamHandle()
            assert handle.done is False
            await handle.emit(text_delta("x"))
            assert handle.done is False
            await handle.close()
            assert handle.done is True

        asyncio.run(scenario())

    def test_collect_text_joins_all_text_delta_events(self):
        async def scenario():
            handle = AgentStreamHandle()
            await handle.emit(text_delta("hello "))
            await handle.emit(text_delta("world"))
            await handle.close()
            text = await handle.collect_text()
            assert text == "hello world"

        asyncio.run(scenario())

    def test_collect_text_ignores_non_text_events(self):
        async def scenario():
            handle = AgentStreamHandle()
            await handle.emit(text_delta("start"))
            await handle.emit(tool_started("some_tool"))
            await handle.emit(text_delta(" end"))
            await handle.close()
            text = await handle.collect_text()
            assert text == "start end"

        asyncio.run(scenario())

    def test_collect_text_empty_stream(self):
        async def scenario():
            handle = AgentStreamHandle()
            await handle.close()
            text = await handle.collect_text()
            assert text == ""

        asyncio.run(scenario())

    def test_emit_error_event_stores_error(self):
        async def scenario():
            handle = AgentStreamHandle()
            await handle.emit(stream_error("something broke"))
            assert handle._error == "something broke"

        asyncio.run(scenario())


# ======================================================================
# TelemetryEvent and TelemetrySpan
# ======================================================================


class TestTelemetryEvent:
    """Tests for TelemetryEvent frozen dataclass."""

    def test_fields(self):
        event = TelemetryEvent(name="test_event", timestamp_ms=1234567890)
        assert event.name == "test_event"
        assert event.timestamp_ms == 1234567890
        assert event.attributes == {}

    def test_attributes_default_empty_dict(self):
        event = TelemetryEvent(name="e", timestamp_ms=0)
        assert event.attributes == {}

    def test_attributes_custom(self):
        event = TelemetryEvent(
            name="e", timestamp_ms=100, attributes={"foo": "bar"}
        )
        assert event.attributes == {"foo": "bar"}

    def test_frozen(self):
        event = TelemetryEvent(name="e", timestamp_ms=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            event.name = "other"  # type: ignore[misc]

    def test_slots(self):
        assert TelemetryEvent.__dataclass_params__.slots is True

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(TelemetryEvent)


class TestTelemetrySpan:
    """Tests for TelemetrySpan frozen dataclass."""

    def test_fields(self):
        span = TelemetrySpan(name="span1", started_at_ms=999)
        assert span.name == "span1"
        assert span.started_at_ms == 999
        assert span.attributes == {}
        assert span.native_span is None

    def test_attributes_default_empty_dict(self):
        span = TelemetrySpan(name="s", started_at_ms=0)
        assert span.attributes == {}

    def test_native_span_default_none(self):
        span = TelemetrySpan(name="s", started_at_ms=0)
        assert span.native_span is None

    def test_native_span_custom(self):
        sentinel = object()
        span = TelemetrySpan(name="s", started_at_ms=0, native_span=sentinel)
        assert span.native_span is sentinel

    def test_frozen(self):
        span = TelemetrySpan(name="s", started_at_ms=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            span.name = "other"  # type: ignore[misc]

    def test_slots(self):
        assert TelemetrySpan.__dataclass_params__.slots is True

    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(TelemetrySpan)
