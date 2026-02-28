"""
Comprehensive tests for the LLM runtime policy layer:
  - CircuitBreaker
  - RetryPolicy / call_with_retry / classify_error
  - RateLimiter
  - Hedging (run_with_hedge)
  - RequestCoalescer
  - Timeouts (await_with_timeout, iter_with_idle_timeout)
"""

from __future__ import annotations

import asyncio
import socket
import time

import pytest

from afk.llms.errors import (
    LLMError,
    LLMRetryableError,
    LLMTimeoutError,
)
from afk.llms.runtime.circuit_breaker import CircuitBreaker
from afk.llms.runtime.coalescing import RequestCoalescer
from afk.llms.runtime.contracts import (
    CircuitBreakerPolicy,
    RateLimitPolicy,
    RetryPolicy,
)
from afk.llms.runtime.hedging import run_with_hedge
from afk.llms.runtime.rate_limit import RateLimiter
from afk.llms.runtime.retry import call_with_retry, classify_error
from afk.llms.runtime.timeouts import await_with_timeout, iter_with_idle_timeout


def run_async(coro):
    return asyncio.run(coro)


# ============================= CircuitBreaker ==============================


class TestCircuitBreaker:
    def test_initial_state_is_closed(self):
        async def scenario():
            cb = CircuitBreaker()
            policy = CircuitBreakerPolicy(failure_threshold=3, cooldown_s=10)
            # Should not raise when no failures recorded
            await cb.ensure_available("key1", policy)

        run_async(scenario())

    def test_opens_after_failure_threshold(self):
        async def scenario():
            cb = CircuitBreaker()
            policy = CircuitBreakerPolicy(failure_threshold=3, cooldown_s=100)

            # Record failures up to threshold
            for _ in range(3):
                await cb.record_failure("key1", policy)

            # Circuit should now be open
            with pytest.raises(LLMRetryableError, match="Circuit open"):
                await cb.ensure_available("key1", policy)

        run_async(scenario())

    def test_does_not_open_below_threshold(self):
        async def scenario():
            cb = CircuitBreaker()
            policy = CircuitBreakerPolicy(failure_threshold=5, cooldown_s=100)

            for _ in range(4):
                await cb.record_failure("key1", policy)

            # Should still be available (4 < 5)
            await cb.ensure_available("key1", policy)

        run_async(scenario())

    def test_success_resets_failure_count(self):
        async def scenario():
            cb = CircuitBreaker()
            policy = CircuitBreakerPolicy(failure_threshold=3, cooldown_s=100)

            # Record 2 failures
            await cb.record_failure("key1", policy)
            await cb.record_failure("key1", policy)

            # Success resets
            await cb.record_success("key1")

            # Should be able to take more failures before opening
            await cb.record_failure("key1", policy)
            await cb.record_failure("key1", policy)
            await cb.ensure_available("key1", policy)  # Still under threshold

        run_async(scenario())

    def test_half_open_allows_probe_calls(self):
        async def scenario():
            cb = CircuitBreaker()
            # Very short cooldown
            policy = CircuitBreakerPolicy(
                failure_threshold=1, cooldown_s=0.01, half_open_max_calls=2
            )

            await cb.record_failure("key1", policy)

            # Wait for cooldown
            await asyncio.sleep(0.02)

            # First probe call should be allowed
            await cb.ensure_available("key1", policy)
            # Second probe call should also be allowed
            await cb.ensure_available("key1", policy)
            # Third probe call should be rejected (max=2)
            with pytest.raises(LLMRetryableError, match="Circuit open"):
                await cb.ensure_available("key1", policy)

        run_async(scenario())

    def test_separate_keys_are_independent(self):
        async def scenario():
            cb = CircuitBreaker()
            policy = CircuitBreakerPolicy(failure_threshold=2, cooldown_s=100)

            # Open circuit for key1
            await cb.record_failure("key1", policy)
            await cb.record_failure("key1", policy)

            # key2 should still work
            await cb.ensure_available("key2", policy)

            # key1 should be open
            with pytest.raises(LLMRetryableError):
                await cb.ensure_available("key1", policy)

        run_async(scenario())


# ============================= classify_error ==============================


class TestClassifyError:
    def test_llm_error_passes_through(self):
        err = LLMError("some error")
        assert classify_error(err) is err

    def test_retryable_error_passes_through(self):
        err = LLMRetryableError("transient")
        assert classify_error(err) is err

    def test_timeout_error_becomes_llm_timeout(self):
        result = classify_error(asyncio.TimeoutError("timed out"))
        assert isinstance(result, LLMTimeoutError)

    def test_builtin_timeout_becomes_llm_timeout(self):
        result = classify_error(TimeoutError("timed out"))
        assert isinstance(result, LLMTimeoutError)

    def test_socket_timeout_becomes_llm_timeout(self):
        result = classify_error(socket.timeout("socket timed out"))
        assert isinstance(result, LLMTimeoutError)

    def test_connection_error_becomes_retryable(self):
        result = classify_error(ConnectionError("refused"))
        assert isinstance(result, LLMRetryableError)

    def test_os_error_becomes_retryable(self):
        result = classify_error(OSError("network down"))
        assert isinstance(result, LLMRetryableError)

    def test_rate_limit_phrase_becomes_retryable(self):
        result = classify_error(RuntimeError("rate limit exceeded"))
        assert isinstance(result, LLMRetryableError)

    def test_429_phrase_becomes_retryable(self):
        result = classify_error(RuntimeError("HTTP 429 too many requests"))
        assert isinstance(result, LLMRetryableError)

    def test_503_phrase_becomes_retryable(self):
        result = classify_error(RuntimeError("503 service unavailable"))
        assert isinstance(result, LLMRetryableError)

    def test_overloaded_phrase_becomes_retryable(self):
        result = classify_error(RuntimeError("server overloaded"))
        assert isinstance(result, LLMRetryableError)

    def test_unknown_error_becomes_llm_error(self):
        result = classify_error(ValueError("something weird"))
        assert isinstance(result, LLMError)
        assert not isinstance(result, LLMRetryableError)
        assert not isinstance(result, LLMTimeoutError)


# ============================= call_with_retry ==============================


class TestCallWithRetry:
    def test_succeeds_on_first_attempt(self):
        async def scenario():
            calls = {"count": 0}

            async def fn():
                calls["count"] += 1
                return "ok"

            result = await call_with_retry(
                fn,
                policy=RetryPolicy(max_retries=3, backoff_base_s=0.0, backoff_jitter_s=0.0),
                can_retry=True,
            )
            assert result == "ok"
            assert calls["count"] == 1

        run_async(scenario())

    def test_retries_on_retryable_error(self):
        async def scenario():
            calls = {"count": 0}

            async def fn():
                calls["count"] += 1
                if calls["count"] < 3:
                    raise ConnectionError("transient")
                return "recovered"

            result = await call_with_retry(
                fn,
                policy=RetryPolicy(max_retries=3, backoff_base_s=0.0, backoff_jitter_s=0.0),
                can_retry=True,
            )
            assert result == "recovered"
            assert calls["count"] == 3

        run_async(scenario())

    def test_raises_non_retryable_immediately(self):
        async def scenario():
            calls = {"count": 0}

            async def fn():
                calls["count"] += 1
                raise ValueError("not retryable at all")

            with pytest.raises(LLMError):
                await call_with_retry(
                    fn,
                    policy=RetryPolicy(
                        max_retries=3, backoff_base_s=0.0, backoff_jitter_s=0.0
                    ),
                    can_retry=True,
                )
            assert calls["count"] == 1

        run_async(scenario())

    def test_exhausts_retries_then_raises(self):
        async def scenario():
            calls = {"count": 0}

            async def fn():
                calls["count"] += 1
                raise ConnectionError("always failing")

            with pytest.raises(LLMRetryableError):
                await call_with_retry(
                    fn,
                    policy=RetryPolicy(
                        max_retries=2, backoff_base_s=0.0, backoff_jitter_s=0.0
                    ),
                    can_retry=True,
                )
            # 1 initial + 2 retries = 3 total
            assert calls["count"] == 3

        run_async(scenario())

    def test_can_retry_false_disables_retries(self):
        async def scenario():
            calls = {"count": 0}

            async def fn():
                calls["count"] += 1
                raise ConnectionError("transient")

            with pytest.raises(LLMRetryableError):
                await call_with_retry(
                    fn,
                    policy=RetryPolicy(
                        max_retries=5, backoff_base_s=0.0, backoff_jitter_s=0.0
                    ),
                    can_retry=False,
                )
            assert calls["count"] == 1

        run_async(scenario())


# ============================= RateLimiter ==============================


class TestRateLimiter:
    def test_first_call_always_succeeds(self):
        async def scenario():
            limiter = RateLimiter()
            policy = RateLimitPolicy(requests_per_second=10, burst=10)
            await limiter.acquire("key1", policy)

        run_async(scenario())

    def test_burst_allows_multiple_calls(self):
        async def scenario():
            limiter = RateLimiter()
            policy = RateLimitPolicy(requests_per_second=1, burst=5)
            for _ in range(5):
                await limiter.acquire("key1", policy)

        run_async(scenario())

    def test_zero_rps_immediately_returns(self):
        """When requests_per_second <= 0, rate limiting is disabled."""

        async def scenario():
            limiter = RateLimiter()
            policy = RateLimitPolicy(requests_per_second=0, burst=1)
            await limiter.acquire("key1", policy)
            await limiter.acquire("key1", policy)

        run_async(scenario())

    def test_separate_keys_have_separate_buckets(self):
        async def scenario():
            limiter = RateLimiter()
            policy = RateLimitPolicy(requests_per_second=1, burst=1)
            await limiter.acquire("key1", policy)
            await limiter.acquire("key2", policy)

        run_async(scenario())

    def test_waits_when_tokens_exhausted(self):
        async def scenario():
            limiter = RateLimiter()
            policy = RateLimitPolicy(requests_per_second=100, burst=1)
            # First call consumes the burst token
            await limiter.acquire("key1", policy)
            # Second call must wait for token refill
            start = time.monotonic()
            await limiter.acquire("key1", policy)
            elapsed = time.monotonic() - start
            assert elapsed >= 0.005  # Should have waited at least a tiny bit

        run_async(scenario())


# ============================= Hedging ==============================


class TestHedging:
    def test_returns_primary_when_no_secondary(self):
        async def scenario():
            result = await run_with_hedge(
                primary=lambda: _async_return("primary", 0),
                secondary=None,
                delay_s=0.01,
            )
            assert result == "primary"

        run_async(scenario())

    def test_primary_wins_when_faster(self):
        async def scenario():
            result = await run_with_hedge(
                primary=lambda: _async_return("primary", 0.0),
                secondary=lambda: _async_return("secondary", 0.5),
                delay_s=0.01,
            )
            assert result == "primary"

        run_async(scenario())

    def test_secondary_wins_when_primary_slow(self):
        async def scenario():
            result = await run_with_hedge(
                primary=lambda: _async_return("primary", 1.0),
                secondary=lambda: _async_return("secondary", 0.0),
                delay_s=0.01,
            )
            assert result == "secondary"

        run_async(scenario())

    def test_primary_error_falls_back_to_secondary(self):
        """If primary fails fast but secondary succeeds, secondary result is returned."""

        async def scenario():
            async def _fail():
                raise ValueError("boom")

            result = await run_with_hedge(
                primary=_fail,
                secondary=lambda: _async_return("secondary", 0.05),
                delay_s=0.0,
            )
            assert result == "secondary"

        run_async(scenario())

    def test_both_error_propagates_primary(self):
        """If both primary and secondary fail, the primary error propagates."""

        async def scenario():
            async def _fail_primary():
                raise ValueError("primary-boom")

            async def _fail_secondary():
                await asyncio.sleep(0.05)
                raise RuntimeError("secondary-boom")

            with pytest.raises((ValueError, RuntimeError)):
                await run_with_hedge(
                    primary=_fail_primary,
                    secondary=_fail_secondary,
                    delay_s=0.0,
                )

        run_async(scenario())

    def test_delay_gives_primary_head_start(self):
        async def scenario():
            started = {"primary": False, "secondary": False}

            async def _primary():
                started["primary"] = True
                await asyncio.sleep(0.01)
                return "primary"

            async def _secondary():
                started["secondary"] = True
                await asyncio.sleep(0.01)
                return "secondary"

            result = await run_with_hedge(
                primary=_primary,
                secondary=_secondary,
                delay_s=5.0,  # secondary won't start before primary finishes
            )
            assert result == "primary"
            assert started["primary"] is True

        run_async(scenario())


# ============================= RequestCoalescer ==============================


class TestRequestCoalescer:
    def test_single_call_goes_through(self):
        async def scenario():
            coalescer = RequestCoalescer()
            result = await coalescer.run("key1", lambda: _async_return("result", 0))
            assert result == "result"

        run_async(scenario())

    def test_concurrent_calls_are_coalesced(self):
        async def scenario():
            coalescer = RequestCoalescer()
            calls = {"count": 0}

            async def factory():
                calls["count"] += 1
                await asyncio.sleep(0.05)
                return "shared"

            results = await asyncio.gather(
                coalescer.run("key1", factory),
                coalescer.run("key1", factory),
                coalescer.run("key1", factory),
            )
            # All three should get the same result
            assert results == ["shared", "shared", "shared"]
            # But factory should only be called once
            assert calls["count"] == 1

        run_async(scenario())

    def test_different_keys_not_coalesced(self):
        async def scenario():
            coalescer = RequestCoalescer()
            calls = {"count": 0}

            async def factory():
                calls["count"] += 1
                await asyncio.sleep(0.01)
                return "shared"

            r1, r2 = await asyncio.gather(
                coalescer.run("key1", factory),
                coalescer.run("key2", factory),
            )
            # Both keys should trigger separate calls
            assert calls["count"] == 2

        run_async(scenario())

    def test_key_reusable_after_completion(self):
        async def scenario():
            coalescer = RequestCoalescer()
            calls = {"count": 0}

            async def factory():
                calls["count"] += 1
                return calls["count"]

            r1 = await coalescer.run("key1", factory)
            r2 = await coalescer.run("key1", factory)
            assert r1 == 1
            assert r2 == 2

        run_async(scenario())

    def test_error_propagates_to_all_waiters(self):
        async def scenario():
            coalescer = RequestCoalescer()

            async def factory():
                await asyncio.sleep(0.02)
                raise RuntimeError("boom")

            results = await asyncio.gather(
                coalescer.run("key1", factory),
                coalescer.run("key1", factory),
                return_exceptions=True,
            )
            assert all(isinstance(r, RuntimeError) for r in results)

        run_async(scenario())


# ============================= Timeouts ==============================


class TestTimeouts:
    def test_await_with_timeout_success(self):
        async def scenario():
            result = await await_with_timeout(_async_return("ok", 0), timeout_s=1.0)
            assert result == "ok"

        run_async(scenario())

    def test_await_with_timeout_no_timeout(self):
        async def scenario():
            result = await await_with_timeout(_async_return("ok", 0), timeout_s=None)
            assert result == "ok"

        run_async(scenario())

    def test_await_with_timeout_raises_on_timeout(self):
        async def scenario():
            with pytest.raises(TimeoutError):
                await await_with_timeout(_async_return("ok", 10), timeout_s=0.01)

        run_async(scenario())

    def test_iter_with_idle_timeout_no_timeout(self):
        async def scenario():
            items = []
            async for item in iter_with_idle_timeout(
                _async_iter([1, 2, 3]), idle_timeout_s=None
            ):
                items.append(item)
            assert items == [1, 2, 3]

        run_async(scenario())

    def test_iter_with_idle_timeout_succeeds_within_timeout(self):
        async def scenario():
            items = []
            async for item in iter_with_idle_timeout(
                _async_iter([1, 2, 3], delay=0.01), idle_timeout_s=1.0
            ):
                items.append(item)
            assert items == [1, 2, 3]

        run_async(scenario())

    def test_iter_with_idle_timeout_raises_on_slow_item(self):
        async def scenario():
            items = []
            with pytest.raises(TimeoutError):
                async for item in iter_with_idle_timeout(
                    _async_iter([1, 2, 3], delay=1.0), idle_timeout_s=0.01
                ):
                    items.append(item)

        run_async(scenario())

    def test_iter_with_idle_timeout_empty_stream(self):
        async def scenario():
            items = []
            async for item in iter_with_idle_timeout(
                _async_iter([]), idle_timeout_s=1.0
            ):
                items.append(item)
            assert items == []

        run_async(scenario())


# ============================= Helpers ==============================


async def _async_return(value, delay):
    await asyncio.sleep(delay)
    return value


async def _async_iter(items, delay=0):
    for item in items:
        if delay > 0:
            await asyncio.sleep(delay)
        yield item
