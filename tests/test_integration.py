"""
Cross-module integration tests for the AFK framework.

These tests verify that different subsystems compose correctly when used
together, exercising realistic multi-module workflows end-to-end.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# LLMs
# ---------------------------------------------------------------------------
from afk.llms.cache.inmemory import InMemoryLLMCache
from afk.llms.errors import LLMRetryableError
from afk.llms.routing.defaults import OrderedFallbackRouter
from afk.llms.runtime.circuit_breaker import CircuitBreaker
from afk.llms.runtime.contracts import CircuitBreakerPolicy, RoutePolicy
from afk.llms.runtime.contracts import RetryPolicy as LLMRetryPolicy
from afk.llms.runtime.retry import call_with_retry
from afk.llms.types import LLMRequest, LLMResponse
from afk.llms.utils import clamp_str, extract_json_object, safe_json_loads

# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
from afk.memory.adapters.in_memory import InMemoryMemoryStore
from afk.memory.lifecycle import RetentionPolicy, compact_thread_memory
from afk.memory.types import MemoryEvent
from afk.memory.utils import new_id, now_ms
from afk.queues.contracts import RUNNER_CHAT_CONTRACT

# ---------------------------------------------------------------------------
# Queues
# ---------------------------------------------------------------------------
from afk.queues.memory import InMemoryTaskQueue
from afk.tools.core.base import ToolContext
from afk.tools.core.decorator import middleware, posthook, prehook, tool
from afk.tools.core.errors import ToolPolicyError

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
from afk.tools.registry import ToolRegistry
from afk.tools.security import SandboxProfile, build_registry_sandbox_policy


def run_async(coro):
    return asyncio.run(coro)


# ===================================================================
# 1. Tool Registry + Security Sandbox Integration
# ===================================================================


class FetchArgs(BaseModel):
    url: str


class _FetchArgsLocal(BaseModel):
    url: str


def test_registry_sandbox_policy_blocks_network_url():
    """
    A sandbox profile with allow_network=False must cause the registry
    policy to reject tool calls that contain network URLs, while still
    allowing non-network arguments through.

    Note: the tool name is intentionally *not* one of the hardcoded
    network tool names (webfetch, web_fetch, etc.) so the test isolates
    URL-argument-level blocking.
    """
    profile = SandboxProfile(
        profile_id="no-network",
        allow_network=False,
    )
    policy = build_registry_sandbox_policy(profile=profile, cwd=Path("/tmp"))

    @tool(args_model=FetchArgs, name="fetch_resource", description="Fetch a resource")
    def fetch_resource(args: FetchArgs) -> str:
        return f"fetched: {args.url}"

    registry = ToolRegistry(max_concurrency=10, policy=policy)
    registry.register(fetch_resource)

    # Calling with a network URL should be blocked by the sandbox policy
    with pytest.raises(ToolPolicyError, match="denied"):
        run_async(
            registry.call(
                "fetch_resource",
                {"url": "https://example.com"},
                ctx=ToolContext(),
            )
        )

    # Calling with a local / non-network value should succeed
    result = run_async(
        registry.call(
            "fetch_resource",
            {"url": "/tmp/local_file.txt"},
            ctx=ToolContext(),
        )
    )
    assert result.success is True
    assert result.output == "fetched: /tmp/local_file.txt"


def test_registry_sandbox_policy_denies_network_tool_name():
    """
    Tools whose names match the hardcoded network tool names (e.g.
    'webfetch') should be blocked even without a URL argument.
    """
    profile = SandboxProfile(
        profile_id="strict",
        allow_network=False,
    )
    policy = build_registry_sandbox_policy(profile=profile, cwd=Path("/tmp"))

    class EmptyArgs(BaseModel):
        query: str

    @tool(args_model=EmptyArgs, name="webfetch", description="Fetch from web")
    def webfetch_tool(args: EmptyArgs) -> str:
        return args.query

    registry = ToolRegistry(max_concurrency=10, policy=policy)
    registry.register(webfetch_tool)

    with pytest.raises(ToolPolicyError, match="denied"):
        run_async(
            registry.call("webfetch", {"query": "hello"}, ctx=ToolContext())
        )


# ===================================================================
# 2. Memory Store + Lifecycle Compaction Integration
# ===================================================================


def test_memory_store_compaction_prunes_events():
    """
    After adding many events to a thread, compact_thread_memory with a
    low max_events_per_thread should prune events and report correct
    before/after counts.
    """
    thread_id = "thread-compaction-test"
    total_events = 30
    max_keep = 10

    async def _run():
        async with InMemoryMemoryStore() as store:
            # Add events with incrementing timestamps
            for i in range(total_events):
                event = MemoryEvent(
                    id=new_id("evt"),
                    thread_id=thread_id,
                    user_id="user-1",
                    type="message",
                    timestamp=now_ms() + i,
                    payload={"text": f"message-{i}"},
                )
                await store.append_event(event)

            # Verify all events were stored
            before = await store.get_recent_events(thread_id, limit=1000)
            assert len(before) == total_events

            # Run compaction
            policy = RetentionPolicy(
                max_events_per_thread=max_keep,
                keep_event_types=["trace"],
                scan_limit=1000,
            )
            result = await compact_thread_memory(
                store,
                thread_id=thread_id,
                event_policy=policy,
            )

            # Verify counts
            assert result.events_before == total_events
            assert result.events_after == max_keep
            assert result.events_removed == total_events - max_keep

            # Verify the store actually has fewer events now
            after = await store.get_recent_events(thread_id, limit=1000)
            assert len(after) == max_keep

    run_async(_run())


def test_memory_compaction_preserves_kept_event_types():
    """
    Events whose type matches keep_event_types should be preserved
    even when aggressively compacting.
    """
    thread_id = "thread-preserve-test"

    async def _run():
        async with InMemoryMemoryStore() as store:
            ts_base = now_ms()
            # Add mostly message events plus some trace events
            for i in range(20):
                event_type = "trace" if i % 5 == 0 else "message"
                event = MemoryEvent(
                    id=new_id("evt"),
                    thread_id=thread_id,
                    user_id="user-1",
                    type=event_type,
                    timestamp=ts_base + i,
                    payload={"idx": i},
                )
                await store.append_event(event)

            policy = RetentionPolicy(
                max_events_per_thread=8,
                keep_event_types=["trace"],
                scan_limit=1000,
            )
            result = await compact_thread_memory(
                store,
                thread_id=thread_id,
                event_policy=policy,
            )

            assert result.events_before == 20
            assert result.events_after == 8

            # All trace events should be in the retained set
            remaining = await store.get_recent_events(thread_id, limit=1000)
            trace_remaining = [e for e in remaining if e.type == "trace"]
            # We had 4 trace events (indices 0, 5, 10, 15); all should survive
            assert len(trace_remaining) == 4

    run_async(_run())


# ===================================================================
# 3. LLM Cache + Retry + Circuit Breaker Integration
# ===================================================================


def test_llm_cache_store_and_retrieve():
    """InMemoryLLMCache round-trip: set then get returns the cached response."""

    async def _run():
        cache = InMemoryLLMCache()
        response = LLMResponse(text="cached hello", model="test-model")

        await cache.set("key-1", response, ttl_s=60.0)
        hit = await cache.get("key-1")
        assert hit is not None
        assert hit.text == "cached hello"
        assert hit.model == "test-model"

        # Non-existent key returns None
        miss = await cache.get("does-not-exist")
        assert miss is None

    run_async(_run())


def test_circuit_breaker_opens_on_failures_and_resets_on_success():
    """
    After enough consecutive failures, the circuit breaker enters open
    state and raises LLMRetryableError. A recorded success resets it.
    """

    async def _run():
        cb = CircuitBreaker()
        policy = CircuitBreakerPolicy(
            failure_threshold=3,
            cooldown_s=9999.0,  # very long so we don't hit half-open
            half_open_max_calls=0,
        )

        key = "provider-x"

        # Record 3 failures to trip the breaker
        for _ in range(3):
            await cb.record_failure(key, policy)

        # Circuit should now be open
        with pytest.raises(LLMRetryableError, match="Circuit open"):
            await cb.ensure_available(key, policy)

        # Record a success to reset the state
        await cb.record_success(key)

        # Circuit should be closed again -- no exception
        await cb.ensure_available(key, policy)

    run_async(_run())


def test_call_with_retry_retries_on_retryable_errors():
    """
    call_with_retry should retry when the callable raises retryable errors
    and eventually return the result when the call succeeds.
    """

    async def _run():
        call_count = 0
        fail_times = 2

        async def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count <= fail_times:
                raise LLMRetryableError(f"transient failure #{call_count}")
            return "success"

        policy = LLMRetryPolicy(
            max_retries=5,
            backoff_base_s=0.0,  # no delay for tests
            backoff_jitter_s=0.0,
        )
        result = await call_with_retry(flaky_fn, policy=policy, can_retry=True)
        assert result == "success"
        # 2 failures + 1 success = 3 total calls
        assert call_count == 3

    run_async(_run())


def test_call_with_retry_raises_on_exhaust():
    """
    When retries are exhausted, call_with_retry should propagate the error.
    """

    async def _run():
        async def always_fail():
            raise LLMRetryableError("permanent transient")

        policy = LLMRetryPolicy(
            max_retries=2,
            backoff_base_s=0.0,
            backoff_jitter_s=0.0,
        )
        with pytest.raises(LLMRetryableError, match="permanent transient"):
            await call_with_retry(always_fail, policy=policy, can_retry=True)

    run_async(_run())


# ===================================================================
# 4. Queue Enqueue + Dequeue + Fail + Retry Lifecycle
# ===================================================================


def test_task_queue_full_lifecycle():
    """
    Walk a task through the complete lifecycle:
    enqueue -> dequeue (running) -> fail (retryable) -> dequeue (retry) -> complete.
    """

    async def _run():
        queue = InMemoryTaskQueue()

        # Enqueue with execution contract
        task = await queue.enqueue_contract(
            RUNNER_CHAT_CONTRACT,
            {"user_message": "hello"},
            agent_name="test-agent",
            max_retries=3,
        )
        task_id = task.id
        assert task.status == "pending"
        assert task.execution_contract == RUNNER_CHAT_CONTRACT

        # Dequeue -- task should move to running
        running = await queue.dequeue(timeout=1.0)
        assert running is not None
        assert running.id == task_id
        assert running.status == "running"
        assert running.started_at is not None

        # Fail with retryable=True -- task should be requeued
        await queue.fail(task_id, error="transient error", retryable=True)
        retrying = await queue.get(task_id)
        assert retrying is not None
        assert retrying.retry_count == 1
        assert retrying.status == "retrying"

        # Dequeue again -- should get the same task with incremented retry_count
        retry_task = await queue.dequeue(timeout=1.0)
        assert retry_task is not None
        assert retry_task.id == task_id
        assert retry_task.status == "running"
        assert retry_task.retry_count == 1

        # Complete the task
        await queue.complete(task_id, result={"response": "done"})
        final = await queue.get(task_id)
        assert final is not None
        assert final.status == "completed"
        assert final.result == {"response": "done"}
        assert final.completed_at is not None

    run_async(_run())


def test_task_queue_non_retryable_failure_becomes_terminal():
    """
    Failing a task with retryable=False should move it directly to
    terminal 'failed' status regardless of remaining retry budget.
    """

    async def _run():
        queue = InMemoryTaskQueue()

        task = await queue.enqueue_contract(
            RUNNER_CHAT_CONTRACT,
            {"user_message": "fail me"},
            agent_name="test-agent",
            max_retries=10,
        )
        task_id = task.id

        # Dequeue
        await queue.dequeue(timeout=1.0)

        # Non-retryable failure
        await queue.fail(task_id, error="fatal error", retryable=False)
        final = await queue.get(task_id)
        assert final is not None
        assert final.status == "failed"
        assert final.is_terminal is True

    run_async(_run())


# ===================================================================
# 5. Tool Decorator + Prehook + Posthook + Middleware Full Pipeline
# ===================================================================


class GreetArgs(BaseModel):
    name: str


class GreetPostArgs(BaseModel):
    output: Any = None
    tool_name: str | None = None


def test_tool_full_pipeline_prehook_posthook_middleware():
    """
    Create a tool with a prehook (uppercases name), a posthook (wraps
    output), and a middleware (adds timing), then verify the full chain
    executes in correct order.
    """

    # Prehook: uppercase the name argument
    @prehook(args_model=GreetArgs, name="uppercase_name")
    def uppercase_hook(args: GreetArgs) -> dict:
        return {"name": args.name.upper()}

    # Posthook: wrap the output in a dict with a "wrapped" key
    @posthook(args_model=GreetPostArgs, name="wrap_output")
    def wrap_hook(args: GreetPostArgs) -> dict:
        return {"wrapped": True, "value": args.output}

    # Middleware: add timing metadata by decorating the call
    @middleware(name="timing_middleware")
    async def timing_mw(call_next, args: GreetArgs, ctx: ToolContext):
        start = time.monotonic()
        result = await call_next(args, ctx)
        elapsed = time.monotonic() - start
        # Return a dict with timing metadata attached
        if isinstance(result, dict):
            result["elapsed_ms"] = round(elapsed * 1000, 2)
            return result
        return {"result": result, "elapsed_ms": round(elapsed * 1000, 2)}

    # Create the tool with all hooks and middleware applied
    @tool(
        args_model=GreetArgs,
        name="greet",
        description="Greet someone with full pipeline",
        prehooks=[uppercase_hook],
        posthooks=[wrap_hook],
        middlewares=[timing_mw],
    )
    def greet_fn(args: GreetArgs) -> str:
        return f"Hello, {args.name}!"

    result = run_async(greet_fn.call({"name": "world"}, ctx=ToolContext()))

    assert result.success is True

    # Prehook should have uppercased the name
    # The main function returns "Hello, WORLD!" (uppercased by prehook)
    # The posthook wraps output into {"wrapped": True, "value": ...}
    output = result.output
    assert isinstance(output, dict)
    assert output["wrapped"] is True
    assert "WORLD" in str(output["value"])


def test_tool_prehook_transforms_args():
    """
    Verify the prehook receives args and transforms them before the
    main tool body runs.
    """

    class PadArgs(BaseModel):
        text: str

    @prehook(args_model=PadArgs, name="pad_prehook")
    def pad_hook(args: PadArgs) -> dict:
        return {"text": f"[{args.text}]"}

    @tool(
        args_model=PadArgs,
        name="echo_padded",
        description="Echo with padding",
        prehooks=[pad_hook],
    )
    def echo_padded(args: PadArgs) -> str:
        return args.text

    result = run_async(echo_padded.call({"text": "inner"}))
    assert result.success is True
    assert result.output == "[inner]"


# ===================================================================
# 6. LLM Utils Integration
# ===================================================================


def test_extract_json_from_fenced_block_and_safe_loads():
    """
    Feed a markdown-fenced JSON block through extract_json_object,
    parse it with safe_json_loads, then verify clamp_str on the
    serialized result.
    """
    markdown_text = """Here is some data:
```json
{"name": "Alice", "scores": [95, 87, 100], "active": true}
```
"""
    extracted = extract_json_object(markdown_text)
    assert extracted is not None

    parsed = safe_json_loads(extracted)
    assert parsed is not None
    assert parsed["name"] == "Alice"
    assert parsed["scores"] == [95, 87, 100]
    assert parsed["active"] is True

    # clamp_str should truncate long strings
    long_string = "A" * 100
    clamped = clamp_str(long_string, 50)
    assert len(clamped) == 51  # 50 chars + ellipsis character
    assert clamped.endswith("\u2026")

    # Short strings are not altered
    short_string = "hello"
    assert clamp_str(short_string, 50) == "hello"


def test_extract_json_object_with_nested_structures():
    """
    extract_json_object should handle nested braces/brackets inside
    JSON objects correctly.
    """
    text = 'Some preamble {"outer": {"inner": [1, 2, {"deep": true}]}} trailing text'
    extracted = extract_json_object(text)
    assert extracted is not None

    parsed = safe_json_loads(extracted)
    assert parsed is not None
    assert parsed["outer"]["inner"][2]["deep"] is True


def test_safe_json_loads_returns_none_on_invalid():
    """safe_json_loads should return None for non-dict or invalid JSON."""
    assert safe_json_loads("not json at all") is None
    assert safe_json_loads("[1, 2, 3]") is None  # array, not dict
    assert safe_json_loads("") is None


def test_extract_json_object_returns_none_for_no_json():
    """extract_json_object should return None when no JSON is present."""
    assert extract_json_object("no json here") is None
    assert extract_json_object("") is None


# ===================================================================
# 7. OrderedFallbackRouter + RoutePolicy Integration
# ===================================================================


def test_ordered_fallback_router_uses_route_policy_order():
    """
    When a request specifies a RoutePolicy.provider_order, the router
    should honor that order and filter to available providers.
    """
    router = OrderedFallbackRouter()
    available = ["openai", "anthropic", "litellm"]

    # Request with explicit order preferring anthropic first
    req = LLMRequest(
        model="gpt-4",
        route_policy=RoutePolicy(provider_order=("anthropic", "openai")),
    )

    order = router.route(
        req,
        available_providers=available,
        default_provider="openai",
    )

    # anthropic should come first (from route_policy), then openai (from default)
    assert order[0] == "anthropic"
    assert "openai" in order


def test_ordered_fallback_router_default_provider_when_no_policy():
    """
    Without a route_policy, the router should fall back to the
    default_provider only.
    """
    router = OrderedFallbackRouter()
    available = ["openai", "anthropic"]

    req = LLMRequest(model="gpt-4")

    order = router.route(
        req,
        available_providers=available,
        default_provider="openai",
    )

    assert order == ["openai"]


def test_ordered_fallback_router_filters_unavailable_providers():
    """
    Providers listed in route_policy.provider_order that are *not* in
    available_providers should be silently dropped.
    """
    router = OrderedFallbackRouter()
    available = ["openai"]

    req = LLMRequest(
        model="gpt-4",
        route_policy=RoutePolicy(provider_order=("anthropic", "openai", "cohere")),
    )

    order = router.route(
        req,
        available_providers=available,
        default_provider="openai",
    )

    # Only openai is available; anthropic and cohere should be filtered out
    assert order == ["openai"]


def test_ordered_fallback_router_deduplicates():
    """
    If the default provider already appears in route_policy.provider_order,
    it should not be duplicated in the output.
    """
    router = OrderedFallbackRouter()
    available = ["openai", "anthropic"]

    req = LLMRequest(
        model="gpt-4",
        route_policy=RoutePolicy(provider_order=("openai", "anthropic")),
    )

    order = router.route(
        req,
        available_providers=available,
        default_provider="openai",
    )

    # No duplicates
    assert len(order) == len(set(order))
    assert order[0] == "openai"
    assert order[1] == "anthropic"


def test_ordered_fallback_router_multiple_requests_different_policies():
    """
    Route multiple requests with varying route_policies through the same
    router and verify each gets the appropriate ordering.
    """
    router = OrderedFallbackRouter()
    available = ["openai", "anthropic", "litellm"]

    requests_and_expected = [
        (
            LLMRequest(
                model="gpt-4",
                route_policy=RoutePolicy(provider_order=("litellm",)),
            ),
            "litellm",
        ),
        (
            LLMRequest(
                model="claude-3",
                route_policy=RoutePolicy(provider_order=("anthropic", "litellm")),
            ),
            "anthropic",
        ),
        (
            LLMRequest(model="gpt-4"),
            "openai",
        ),
    ]

    for req, expected_first in requests_and_expected:
        order = router.route(
            req,
            available_providers=available,
            default_provider="openai",
        )
        assert len(order) >= 1
        assert order[0] == expected_first
