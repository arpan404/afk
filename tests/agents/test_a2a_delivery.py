"""
Tests for afk.agents.a2a.delivery and afk.agents.a2a.internal_protocol.
"""

import asyncio

import pytest

from afk.agents.a2a.delivery import InMemoryA2ADeliveryStore
from afk.agents.a2a.internal_protocol import InternalA2AEnvelope, InternalA2AProtocol
from afk.agents.contracts import (
    AgentDeadLetter,
    AgentInvocationRequest,
    AgentInvocationResponse,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_request(**kwargs):
    defaults = dict(
        run_id="r1",
        thread_id="t1",
        conversation_id="c1",
        correlation_id="corr1",
        idempotency_key="idem1",
        source_agent="agent_a",
        target_agent="agent_b",
    )
    defaults.update(kwargs)
    return AgentInvocationRequest(**defaults)


def _make_response(request, *, success=True, output=None, error=None):
    return AgentInvocationResponse(
        run_id=request.run_id,
        thread_id=request.thread_id,
        conversation_id=request.conversation_id,
        correlation_id=request.correlation_id,
        idempotency_key=request.idempotency_key,
        source_agent=request.target_agent,
        target_agent=request.source_agent,
        success=success,
        output=output,
        error=error,
    )


# ── InMemoryA2ADeliveryStore ────────────────────────────────────────────────


class TestInMemoryA2ADeliveryStore:
    def test_get_success_returns_none_for_unknown_key(self):
        store = InMemoryA2ADeliveryStore()
        result = asyncio.run(store.get_success("nonexistent"))
        assert result is None

    def test_record_success_then_get_success(self):
        store = InMemoryA2ADeliveryStore()
        req = _make_request()
        resp = _make_response(req, output="done")

        async def _run():
            await store.record_success("key1", resp)
            return await store.get_success("key1")

        result = asyncio.run(_run())
        assert result is not None
        assert result.success is True
        assert result.output == "done"
        assert result.run_id == req.run_id

    def test_record_dead_letter_stores_entry(self):
        store = InMemoryA2ADeliveryStore()
        req = _make_request()
        dead = AgentDeadLetter(request=req, error="timeout", attempts=3)

        async def _run():
            await store.record_dead_letter(dead)
            return await store.list_dead_letters()

        result = asyncio.run(_run())
        assert len(result) == 1
        assert result[0].error == "timeout"
        assert result[0].attempts == 3

    def test_list_dead_letters_empty_by_default(self):
        store = InMemoryA2ADeliveryStore()
        result = asyncio.run(store.list_dead_letters())
        assert result == []

    def test_list_dead_letters_returns_all(self):
        store = InMemoryA2ADeliveryStore()
        req1 = _make_request(correlation_id="c1")
        req2 = _make_request(correlation_id="c2")
        dead1 = AgentDeadLetter(request=req1, error="err1", attempts=1)
        dead2 = AgentDeadLetter(request=req2, error="err2", attempts=2)

        async def _run():
            await store.record_dead_letter(dead1)
            await store.record_dead_letter(dead2)
            return await store.list_dead_letters()

        result = asyncio.run(_run())
        assert len(result) == 2

    def test_multiple_success_keys(self):
        store = InMemoryA2ADeliveryStore()
        req1 = _make_request(idempotency_key="k1")
        req2 = _make_request(idempotency_key="k2")
        resp1 = _make_response(req1, output="out1")
        resp2 = _make_response(req2, output="out2")

        async def _run():
            await store.record_success("k1", resp1)
            await store.record_success("k2", resp2)
            r1 = await store.get_success("k1")
            r2 = await store.get_success("k2")
            return r1, r2

        r1, r2 = asyncio.run(_run())
        assert r1.output == "out1"
        assert r2.output == "out2"


# ── InternalA2AEnvelope ─────────────────────────────────────────────────────


class TestInternalA2AEnvelope:
    def test_default_values(self):
        env = InternalA2AEnvelope(
            message_type="request",
            run_id="r1",
            thread_id="t1",
            conversation_id="c1",
            correlation_id="corr1",
            idempotency_key="idem1",
            source_agent="a",
            target_agent="b",
        )
        assert env.payload == {}
        assert env.metadata == {}
        assert env.causation_id is None
        assert isinstance(env.timestamp_ms, int)
        assert env.timestamp_ms > 0

    def test_custom_values(self):
        env = InternalA2AEnvelope(
            message_type="response",
            run_id="r2",
            thread_id="t2",
            conversation_id="c2",
            correlation_id="corr2",
            idempotency_key="idem2",
            source_agent="x",
            target_agent="y",
            payload={"key": "value"},
            metadata={"trace": "abc"},
            causation_id="caus1",
        )
        assert env.payload == {"key": "value"}
        assert env.metadata == {"trace": "abc"}
        assert env.causation_id == "caus1"

    def test_frozen_dataclass(self):
        env = InternalA2AEnvelope(
            message_type="event",
            run_id="r1",
            thread_id="t1",
            conversation_id="c1",
            correlation_id="corr1",
            idempotency_key="idem1",
            source_agent="a",
            target_agent="b",
        )
        with pytest.raises(AttributeError):
            env.run_id = "changed"

    def test_timestamp_ms_auto_generated(self):
        env1 = InternalA2AEnvelope(
            message_type="request",
            run_id="r1",
            thread_id="t1",
            conversation_id="c1",
            correlation_id="corr1",
            idempotency_key="idem1",
            source_agent="a",
            target_agent="b",
        )
        env2 = InternalA2AEnvelope(
            message_type="request",
            run_id="r1",
            thread_id="t1",
            conversation_id="c1",
            correlation_id="corr1",
            idempotency_key="idem1",
            source_agent="a",
            target_agent="b",
        )
        # Both should have timestamps and they should be very close
        assert abs(env1.timestamp_ms - env2.timestamp_ms) < 1000


# ── InternalA2AProtocol ─────────────────────────────────────────────────────


class TestInternalA2AProtocol:
    def test_protocol_id(self):
        async def _noop(req):
            return _make_response(req)

        protocol = InternalA2AProtocol(dispatch=_noop)
        assert protocol.protocol_id == "internal.a2a.v1"

    def test_invoke_dispatches_and_returns_response(self):
        async def _echo(req):
            return _make_response(req, success=True, output="echoed")

        protocol = InternalA2AProtocol(dispatch=_echo)
        req = _make_request()

        result = asyncio.run(protocol.invoke(req))
        assert result.success is True
        assert result.output == "echoed"
        assert result.source_agent == "agent_b"
        assert result.target_agent == "agent_a"

    def test_invoke_records_protocol_events(self):
        async def _echo(req):
            return _make_response(req, success=True, output="ok")

        protocol = InternalA2AProtocol(dispatch=_echo)
        req = _make_request()

        asyncio.run(protocol.invoke(req))
        events = protocol.events()
        event_types = [e.type for e in events]
        assert "queued" in event_types
        assert "dispatched" in event_types
        assert "acked" in event_types
        assert "completed" in event_types

    def test_successful_response_recorded_for_dedupe(self):
        async def _echo(req):
            return _make_response(req, success=True, output="first")

        store = InMemoryA2ADeliveryStore()
        protocol = InternalA2AProtocol(dispatch=_echo, delivery_store=store)
        req = _make_request()

        asyncio.run(protocol.invoke(req))
        cached = asyncio.run(store.get_success(req.idempotency_key))
        assert cached is not None
        assert cached.output == "first"

    def test_second_invoke_with_same_idempotency_key_returns_cached(self):
        call_count = 0

        async def _counting_echo(req):
            nonlocal call_count
            call_count += 1
            return _make_response(req, success=True, output=f"call_{call_count}")

        protocol = InternalA2AProtocol(dispatch=_counting_echo)
        req = _make_request()

        async def _run():
            r1 = await protocol.invoke(req)
            r2 = await protocol.invoke(req)
            return r1, r2

        r1, r2 = asyncio.run(_run())
        # First call dispatches, second is deduped
        assert r1.output == "call_1"
        assert r2.output == "call_1"  # cached, not "call_2"
        assert call_count == 1

    def test_dedupe_emits_ignored_late_response_event(self):
        async def _echo(req):
            return _make_response(req, success=True, output="ok")

        protocol = InternalA2AProtocol(dispatch=_echo)
        req = _make_request()

        async def _run():
            await protocol.invoke(req)
            await protocol.invoke(req)

        asyncio.run(_run())
        events = protocol.events()
        event_types = [e.type for e in events]
        assert "ignored_late_response" in event_types

    def test_failed_response_records_nacked_and_failed(self):
        async def _fail(req):
            return _make_response(req, success=False, error="something broke")

        protocol = InternalA2AProtocol(dispatch=_fail)
        req = _make_request()

        result = asyncio.run(protocol.invoke(req))
        assert result.success is False
        assert result.error == "something broke"

        events = protocol.events()
        event_types = [e.type for e in events]
        assert "nacked" in event_types
        assert "failed" in event_types
        # Should NOT record acked or completed
        assert "acked" not in event_types
        assert "completed" not in event_types

    def test_failed_response_not_cached_for_dedupe(self):
        call_count = 0

        async def _fail_then_succeed(req):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_response(req, success=False, error="first fail")
            return _make_response(req, success=True, output="recovered")

        protocol = InternalA2AProtocol(dispatch=_fail_then_succeed)
        req = _make_request()

        async def _run():
            r1 = await protocol.invoke(req)
            r2 = await protocol.invoke(req)
            return r1, r2

        r1, r2 = asyncio.run(_run())
        assert r1.success is False
        assert r2.success is True
        assert r2.output == "recovered"
        assert call_count == 2

    def test_record_dead_letter(self):
        async def _echo(req):
            return _make_response(req)

        protocol = InternalA2AProtocol(dispatch=_echo)
        req = _make_request()

        asyncio.run(
            protocol.record_dead_letter(req, error="exhausted retries", attempts=5)
        )

        dead_letters = protocol.dead_letters()
        assert len(dead_letters) == 1
        assert dead_letters[0].error == "exhausted retries"
        assert dead_letters[0].attempts == 5

        events = protocol.events()
        event_types = [e.type for e in events]
        assert "dead_letter" in event_types

    def test_record_dead_letter_also_in_delivery_store(self):
        async def _echo(req):
            return _make_response(req)

        store = InMemoryA2ADeliveryStore()
        protocol = InternalA2AProtocol(dispatch=_echo, delivery_store=store)
        req = _make_request()

        async def _run():
            await protocol.record_dead_letter(req, error="timeout", attempts=3)
            return await store.list_dead_letters()

        result = asyncio.run(_run())
        assert len(result) == 1
        assert result[0].error == "timeout"

    def test_get_task_returns_task_metadata(self):
        async def _echo(req):
            return _make_response(req, success=True, output="done")

        protocol = InternalA2AProtocol(dispatch=_echo)
        req = _make_request()

        async def _run():
            await protocol.invoke(req)
            return await protocol.get_task(req.correlation_id)

        task = asyncio.run(_run())
        assert task["status"] == "completed"
        assert task["run_id"] == "r1"
        assert task["target_agent"] == "agent_b"

    def test_get_task_raises_key_error_for_unknown(self):
        async def _echo(req):
            return _make_response(req)

        protocol = InternalA2AProtocol(dispatch=_echo)

        with pytest.raises(KeyError, match="Unknown task_id"):
            asyncio.run(protocol.get_task("nonexistent_id"))

    def test_cancel_task_marks_as_cancel_requested(self):
        cancelled = False

        async def _slow(req):
            nonlocal cancelled
            # Simulate a running task; we will cancel before it completes
            # For this test, we need the task to be in "running" state,
            # so we invoke, and then cancel from the _tasks dict directly
            # But since _invoke_internal is sequential, we test cancel on
            # a task that was not yet completed via direct _tasks manipulation.
            return _make_response(req, success=True, output="done")

        protocol = InternalA2AProtocol(dispatch=_slow)
        req = _make_request()

        async def _run():
            # First, invoke to create the task
            await protocol.invoke(req)
            # Manually reset the status to "running" to simulate an in-flight task
            async with protocol._lock:
                protocol._tasks[req.correlation_id] = {
                    **protocol._tasks[req.correlation_id],
                    "status": "running",
                }
            # Now cancel
            result = await protocol.cancel_task(req.correlation_id)
            return result

        result = asyncio.run(_run())
        assert result["status"] == "cancel_requested"

    def test_cancel_task_on_completed_task_unchanged(self):
        async def _echo(req):
            return _make_response(req, success=True, output="done")

        protocol = InternalA2AProtocol(dispatch=_echo)
        req = _make_request()

        async def _run():
            await protocol.invoke(req)
            return await protocol.cancel_task(req.correlation_id)

        result = asyncio.run(_run())
        # Completed task should remain completed
        assert result["status"] == "completed"

    def test_cancel_task_on_failed_task_unchanged(self):
        async def _fail(req):
            return _make_response(req, success=False, error="broke")

        protocol = InternalA2AProtocol(dispatch=_fail)
        req = _make_request()

        async def _run():
            await protocol.invoke(req)
            return await protocol.cancel_task(req.correlation_id)

        result = asyncio.run(_run())
        assert result["status"] == "failed"

    def test_cancel_task_unknown_raises_key_error(self):
        async def _echo(req):
            return _make_response(req)

        protocol = InternalA2AProtocol(dispatch=_echo)

        with pytest.raises(KeyError, match="Unknown task_id"):
            asyncio.run(protocol.cancel_task("missing"))

    def test_default_delivery_store_created_if_none(self):
        async def _echo(req):
            return _make_response(req)

        protocol = InternalA2AProtocol(dispatch=_echo)
        assert isinstance(protocol._delivery_store, InMemoryA2ADeliveryStore)

    def test_events_initially_empty(self):
        async def _echo(req):
            return _make_response(req)

        protocol = InternalA2AProtocol(dispatch=_echo)
        assert protocol.events() == []

    def test_dead_letters_initially_empty(self):
        async def _echo(req):
            return _make_response(req)

        protocol = InternalA2AProtocol(dispatch=_echo)
        assert protocol.dead_letters() == []
