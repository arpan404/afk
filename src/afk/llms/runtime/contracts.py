"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Typed runtime policies for enterprise LLM execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry semantics for one request path."""

    max_retries: int = 3
    backoff_base_s: float = 0.5
    backoff_jitter_s: float = 0.15
    require_idempotency_key: bool = True


@dataclass(frozen=True, slots=True)
class TimeoutPolicy:
    """Timeout semantics for unary and stream operations."""

    request_timeout_s: float | None = 30.0
    stream_idle_timeout_s: float | None = 45.0


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """Token bucket policy used per provider and operation."""

    requests_per_second: float = 20.0
    burst: int = 40


@dataclass(frozen=True, slots=True)
class CircuitBreakerPolicy:
    """Consecutive failure policy with half-open probes."""

    failure_threshold: int = 5
    cooldown_s: float = 30.0
    half_open_max_calls: int = 1


@dataclass(frozen=True, slots=True)
class HedgingPolicy:
    """Optional speculative secondary call to reduce tail latency."""

    enabled: bool = False
    delay_s: float = 0.2


@dataclass(frozen=True, slots=True)
class CachePolicy:
    """Response cache controls."""

    enabled: bool = False
    ttl_s: float = 30.0
    namespace: str | None = None


@dataclass(frozen=True, slots=True)
class CoalescingPolicy:
    """In-flight request deduplication controls."""

    enabled: bool = True


@dataclass(frozen=True, slots=True)
class RoutePolicy:
    """Provider routing and fallback order."""

    provider_order: tuple[str, ...] = field(default_factory=tuple)
