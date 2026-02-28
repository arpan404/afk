"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Module: runtime/client.py.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ..cache.registry import create_llm_cache
from ..errors import LLMCapabilityError, LLMError
from ..middleware import MiddlewareStack
from ..observability import LLMObserver
from ..providers.contracts import LLMTransport
from ..providers.registry import get_llm_provider, list_llm_providers
from ..routing.registry import create_llm_router
from ..settings import LLMSettings
from ..types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMRequest,
    LLMResponse,
    LLMSessionHandle,
    LLMStreamHandle,
)
from ..utils import run_sync
from .circuit_breaker import CircuitBreaker
from .coalescing import RequestCoalescer
from .contracts import (
    CachePolicy,
    CircuitBreakerPolicy,
    CoalescingPolicy,
    HedgingPolicy,
    RateLimitPolicy,
    RetryPolicy,
    RoutePolicy,
    TimeoutPolicy,
)
from .hedging import run_with_hedge
from .rate_limit import RateLimiter
from .retry import call_with_retry
from .streaming import RuntimeStreamHandle
from .timeouts import await_with_timeout, iter_with_idle_timeout


class LLMClient:
    """Provider-driven runtime with pluggable enterprise execution policies."""

    def __init__(
        self,
        *,
        provider: str,
        settings: LLMSettings,
        provider_settings: Mapping[str, Any] | None = None,
        middlewares: MiddlewareStack | None = None,
        observers: list[LLMObserver] | None = None,
        cache_backend=None,
        router=None,
        retry_policy: RetryPolicy | None = None,
        timeout_policy: TimeoutPolicy | None = None,
        rate_limit_policy: RateLimitPolicy | None = None,
        circuit_breaker_policy: CircuitBreakerPolicy | None = None,
        hedging_policy: HedgingPolicy | None = None,
        cache_policy: CachePolicy | None = None,
        coalescing_policy: CoalescingPolicy | None = None,
    ) -> None:
        self.settings = settings
        self._provider_settings = dict(provider_settings or {})
        self._middlewares = middlewares
        self._observers = observers

        self._transports: dict[str, LLMTransport] = {}
        self._default_provider = provider.strip().lower()
        self._cache = create_llm_cache(cache_backend)
        self._router = create_llm_router(router)

        self._retry_policy = retry_policy or RetryPolicy(
            max_retries=settings.max_retries,
            backoff_base_s=settings.backoff_base_s,
            backoff_jitter_s=settings.backoff_jitter_s,
        )
        self._timeout_policy = timeout_policy or TimeoutPolicy(
            request_timeout_s=settings.timeout_s,
            stream_idle_timeout_s=settings.stream_idle_timeout_s,
        )
        self._rate_limit_policy = rate_limit_policy or RateLimitPolicy()
        self._breaker_policy = circuit_breaker_policy or CircuitBreakerPolicy()
        self._hedging_policy = hedging_policy or HedgingPolicy()
        self._cache_policy = cache_policy or CachePolicy()
        self._coalescing_policy = coalescing_policy or CoalescingPolicy()

        self._rate_limiter = RateLimiter()
        self._breaker = CircuitBreaker()
        self._coalescer = RequestCoalescer()

    @property
    def provider_id(self) -> str:
        """Return default provider id configured for this client."""
        return self._default_provider

    @property
    def capabilities(self):
        """Expose capability flags from the default provider transport."""
        return self._get_transport(self._default_provider).capabilities

    def _get_transport(self, provider_id: str) -> LLMTransport:
        key = provider_id.strip().lower()
        existing = self._transports.get(key)
        if existing is not None:
            return existing

        provider = get_llm_provider(key)
        provider_settings = (
            self._provider_settings.get(key)
            if isinstance(self._provider_settings.get(key), Mapping)
            else None
        )
        transport = provider.create_transport(
            settings=self.settings,
            middlewares=self._middlewares,
            observers=self._observers,
            provider_settings=provider_settings,
        )
        self._transports[key] = transport
        return transport

    def _providers_for_request(self, req: LLMRequest) -> list[str]:
        """Resolve ordered provider candidates for one request."""
        available = list_llm_providers()
        if self._default_provider not in available:
            available = [self._default_provider, *available]
        return self._router.route(
            req,
            available_providers=available,
            default_provider=self._default_provider,
        )

    def _cache_key(self, provider: str, req: LLMRequest) -> str:
        """Build deterministic cache key for request payload + provider id."""
        payload = {
            "provider": provider,
            "model": req.model,
            "messages": [
                {"role": m.role, "name": m.name, "content": m.content}
                for m in req.messages
            ],
            "tools": req.tools,
            "tool_choice": req.tool_choice,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
            "thinking": req.thinking,
            "thinking_effort": req.thinking_effort,
        }
        normalized = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    async def _call_one(
        self,
        provider_id: str,
        req: LLMRequest,
        *,
        response_model=None,
    ) -> LLMResponse:
        """Execute one non-streaming call against a specific provider id."""
        transport = self._get_transport(provider_id)

        await self._rate_limiter.acquire(
            f"{provider_id}:chat",
            self._rate_limit_policy,
        )
        await self._breaker.ensure_available(provider_id, self._breaker_policy)

        retry_policy = req.retry_policy or self._retry_policy
        timeout_policy = req.timeout_policy or self._timeout_policy

        can_retry = True
        if retry_policy.require_idempotency_key:
            can_retry = bool(req.idempotency_key) and bool(
                getattr(transport.capabilities, "idempotency", False)
            )

        async def _do_call() -> LLMResponse:
            result = await await_with_timeout(
                transport.chat(req, response_model=response_model),
                timeout_policy.request_timeout_s,
            )
            await self._breaker.record_success(provider_id)
            return result

        try:
            return await call_with_retry(
                _do_call, policy=retry_policy, can_retry=can_retry
            )
        except Exception:
            await self._breaker.record_failure(provider_id, self._breaker_policy)
            raise

    async def chat(
        self,
        req: LLMRequest,
        *,
        response_model=None,
    ) -> LLMResponse:
        """
        Execute one non-streaming request with cache/coalescing/fallback support.

        This method is the primary high-level entrypoint for agent turn calls.
        """
        providers = self._providers_for_request(req)
        if not providers:
            raise LLMError("No providers available for request")

        cache_policy = req.cache_policy or self._cache_policy
        primary_cache_key = self._cache_key(providers[0], req)
        if cache_policy.enabled:
            cached = await self._cache.get(primary_cache_key)
            if cached is not None:
                return cached

        async def _call_primary() -> LLMResponse:
            if self._coalescing_policy.enabled:
                return await self._coalescer.run(
                    primary_cache_key,
                    lambda: self._call_one(
                        providers[0], req, response_model=response_model
                    ),
                )
            return await self._call_one(
                providers[0], req, response_model=response_model
            )

        async def _call_secondary() -> LLMResponse:
            if len(providers) < 2:
                raise LLMError("No secondary provider configured")
            secondary_req = replace(
                req, route_policy=RoutePolicy(provider_order=(providers[1],))
            )
            return await self._call_one(
                providers[1], secondary_req, response_model=response_model
            )

        result_provider = providers[0]
        try:
            if self._hedging_policy.enabled and len(providers) > 1:
                result = await run_with_hedge(
                    _call_primary,
                    _call_secondary,
                    delay_s=self._hedging_policy.delay_s,
                )
            else:
                result = await _call_primary()
        except Exception:
            for provider_id in providers[1:]:
                try:
                    result = await self._call_one(
                        provider_id, req, response_model=response_model
                    )
                    result_provider = provider_id
                    break
                except Exception:
                    continue
            else:
                raise

        if cache_policy.enabled:
            cache_key = self._cache_key(result_provider, req)
            await self._cache.set(cache_key, result, ttl_s=cache_policy.ttl_s)
        return result

    def chat_sync(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        """Synchronous wrapper around `chat`."""
        return run_sync(self.chat(req, response_model=response_model))

    async def chat_stream(
        self,
        req: LLMRequest,
        *,
        response_model=None,
    ):
        """
        Execute one streaming request with startup retry and provider fallback.

        Returned iterator enforces idle timeout when configured.
        """
        providers = self._providers_for_request(req)
        if not providers:
            raise LLMError("No providers available for request")

        timeout_policy = req.timeout_policy or self._timeout_policy
        retry_policy = req.retry_policy or self._retry_policy

        for provider_id in providers:
            transport = self._get_transport(provider_id)
            try:
                await self._rate_limiter.acquire(
                    f"{provider_id}:stream",
                    self._rate_limit_policy,
                )
                await self._breaker.ensure_available(provider_id, self._breaker_policy)

                can_retry = True
                if retry_policy.require_idempotency_key:
                    can_retry = bool(req.idempotency_key) and bool(
                        getattr(transport.capabilities, "idempotency", False)
                    )

                async def _start_stream() -> Any:
                    stream = await await_with_timeout(
                        transport.chat_stream(req, response_model=response_model),
                        timeout_policy.request_timeout_s,
                    )
                    await self._breaker.record_success(provider_id)
                    return stream

                stream = await call_with_retry(
                    _start_stream,
                    policy=retry_policy,
                    can_retry=can_retry,
                )
                return iter_with_idle_timeout(
                    stream,
                    idle_timeout_s=req.stream_idle_timeout_s
                    if req.stream_idle_timeout_s is not None
                    else timeout_policy.stream_idle_timeout_s,
                )
            except Exception:
                await self._breaker.record_failure(provider_id, self._breaker_policy)
                continue

        raise LLMError("All providers failed for streaming request")

    async def chat_stream_handle(
        self,
        req: LLMRequest,
        *,
        response_model=None,
    ) -> LLMStreamHandle:
        """Execute streaming request and return control handle for cancel/interrupt."""
        providers = self._providers_for_request(req)
        timeout_policy = req.timeout_policy or self._timeout_policy

        for provider_id in providers:
            transport = self._get_transport(provider_id)
            try:
                await self._rate_limiter.acquire(
                    f"{provider_id}:stream",
                    self._rate_limit_policy,
                )
                await self._breaker.ensure_available(provider_id, self._breaker_policy)
                base_handle = await await_with_timeout(
                    transport.chat_stream_handle(req, response_model=response_model),
                    timeout_policy.request_timeout_s,
                )
                await self._breaker.record_success(provider_id)

                stream = iter_with_idle_timeout(
                    base_handle.events,
                    idle_timeout_s=req.stream_idle_timeout_s
                    if req.stream_idle_timeout_s is not None
                    else timeout_policy.stream_idle_timeout_s,
                )
                return RuntimeStreamHandle(
                    source=stream,
                    interrupt_callback=base_handle.interrupt,
                    cancel_callback=base_handle.cancel,
                )
            except Exception:
                await self._breaker.record_failure(provider_id, self._breaker_policy)
                continue

        raise LLMError("All providers failed for streaming-handle request")

    async def embed(self, req: EmbeddingRequest) -> EmbeddingResponse:
        """Execute embedding request on default provider with runtime policies."""
        provider_id = self._default_provider
        transport = self._get_transport(provider_id)
        if not getattr(transport.capabilities, "embeddings", False):
            raise LLMCapabilityError(
                f"Provider '{provider_id}' does not support capability 'embeddings'"
            )

        await self._rate_limiter.acquire(
            f"{provider_id}:embed", self._rate_limit_policy
        )
        await self._breaker.ensure_available(provider_id, self._breaker_policy)

        timeout_policy = self._timeout_policy
        try:
            result = await await_with_timeout(
                transport.embed(req),
                timeout_policy.request_timeout_s,
            )
            await self._breaker.record_success(provider_id)
            return result
        except Exception:
            await self._breaker.record_failure(provider_id, self._breaker_policy)
            raise

    def embed_sync(self, req: EmbeddingRequest) -> EmbeddingResponse:
        """Synchronous wrapper around `embed`."""
        return run_sync(self.embed(req))

    def resolve_thinking(self, req: LLMRequest):
        """Resolve thinking config using default provider adapter behavior."""
        transport = self._get_transport(self._default_provider)
        if hasattr(transport, "resolve_thinking"):
            return transport.resolve_thinking(req)
        raise LLMError(
            f"Provider '{self._default_provider}' does not expose thinking resolution"
        )

    def start_session(
        self,
        *,
        session_token: str | None = None,
        checkpoint_token: str | None = None,
    ) -> LLMSessionHandle:
        """Start one provider-native session handle when supported."""
        transport = self._get_transport(self._default_provider)
        return transport.start_session(
            session_token=session_token,
            checkpoint_token=checkpoint_token,
        )
