"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Module: __init__.py.
"""

from __future__ import annotations

from .builder import LLMBuilder
from .cache.registry import (
    create_llm_cache,
    list_llm_cache_backends,
    register_llm_cache_backend,
)
from .config import LLMConfig
from .errors import (
    LLMCancelledError,
    LLMCapabilityError,
    LLMConfigurationError,
    LLMError,
    LLMInterruptedError,
    LLMInvalidResponseError,
    LLMRetryableError,
    LLMSessionError,
    LLMSessionPausedError,
    LLMTimeoutError,
)
from .llm import LLM
from .middleware import MiddlewareStack
from .observability import LLMLifecycleEvent, LLMObserver
from .profiles import PROFILES, LLMProfile
from .providers import (
    AnthropicAgentProvider,
    LiteLLMProvider,
    LLMProvider,
    LLMProviderError,
    LLMTransport,
    OpenAIProvider,
    ProviderSettingsSchema,
    get_llm_provider,
    list_llm_providers,
    register_llm_provider,
)
from .routing.registry import create_llm_router, list_llm_routers, register_llm_router
from .runtime import (
    CachePolicy,
    CircuitBreakerPolicy,
    CoalescingPolicy,
    HedgingPolicy,
    LLMClient,
    RateLimitPolicy,
    RetryPolicy,
    RoutePolicy,
    TimeoutPolicy,
)
from .settings import LLMSettings
from .tool_export import (
    export_tools_for_provider,
    normalize_json_schema,
    to_openai_tools,
    to_openai_tools_from_specs,
    tool_to_openai_tool,
    toolspec_to_openai_tool,
)
from .types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
    LLMSessionHandle,
    LLMSessionSnapshot,
    LLMStreamEvent,
    LLMStreamHandle,
    Message,
    StreamCompletedEvent,
    StreamErrorEvent,
    StreamMessageStartEvent,
    StreamMessageStopEvent,
    StreamTextDeltaEvent,
    StreamToolCallDeltaEvent,
    ToolCall,
    Usage,
)

# Built-in provider bootstrap (hard-break API surface).
register_llm_provider(OpenAIProvider(), overwrite=True)
register_llm_provider(LiteLLMProvider(), overwrite=True)
register_llm_provider(AnthropicAgentProvider(), overwrite=True)


def create_llm_client(
    *,
    provider: str,
    settings: LLMSettings | None = None,
    provider_settings: dict[str, dict] | None = None,
    middlewares: MiddlewareStack | None = None,
    observers: list[LLMObserver] | None = None,
    router=None,
    retry_policy: RetryPolicy | None = None,
    timeout_policy: TimeoutPolicy | None = None,
    rate_limit_policy: RateLimitPolicy | None = None,
    circuit_breaker_policy: CircuitBreakerPolicy | None = None,
    hedging_policy: HedgingPolicy | None = None,
    cache_policy: CachePolicy | None = None,
    coalescing_policy: CoalescingPolicy | None = None,
) -> LLMClient:
    """Create enterprise runtime client with explicit provider selection.

    Args:
        provider: Provider id (e.g., "openai", "anthropic_agent", "litellm")
        settings: LLMSettings instance (defaults to env)
        provider_settings: Per-provider API keys/URLs
        middlewares: MiddlewareStack for transport paths
        observers: LLM lifecycle observers
        router: LLMRouter for multi-provider fallback
        retry_policy: Retry policy for transient failures
        timeout_policy: Request/stream timeout policy
        rate_limit_policy: Rate limiting policy
        circuit_breaker_policy: Circuit breaker for fault isolation
        hedging_policy: Tail latency hedging (fires duplicate requests)
        cache_policy: Response caching policy
        coalescing_policy: Request coalescing for identical payloads

    Example:
        # Use hedging for latency-sensitive applications
        client = create_llm_client(
            provider="openai",
            hedging_policy=HedgingPolicy(enabled=True, delay_s=0.1),
        )
    """
    return LLMClient(
        provider=provider,
        settings=settings or LLMSettings.from_env(),
        provider_settings=provider_settings,
        middlewares=middlewares,
        observers=observers,
        router=router,
        retry_policy=retry_policy,
        timeout_policy=timeout_policy,
        rate_limit_policy=rate_limit_policy,
        circuit_breaker_policy=circuit_breaker_policy,
        hedging_policy=hedging_policy,
        cache_policy=cache_policy,
        coalescing_policy=coalescing_policy,
    )


__all__ = [
    "LLM",
    "LLMClient",
    "LLMBuilder",
    "LLMProfile",
    "LLMSettings",
    "LLMProvider",
    "LLMTransport",
    "ProviderSettingsSchema",
    "register_llm_provider",
    "get_llm_provider",
    "list_llm_providers",
    "LLMProviderError",
    "create_llm_client",
    "register_llm_cache_backend",
    "create_llm_cache",
    "list_llm_cache_backends",
    "register_llm_router",
    "create_llm_router",
    "list_llm_routers",
    "RetryPolicy",
    "TimeoutPolicy",
    "RateLimitPolicy",
    "CircuitBreakerPolicy",
    "HedgingPolicy",
    "CachePolicy",
    "CoalescingPolicy",
    "RoutePolicy",
    "PROFILES",
    "LLMConfig",
    "MiddlewareStack",
    "LLMError",
    "LLMTimeoutError",
    "LLMRetryableError",
    "LLMInvalidResponseError",
    "LLMConfigurationError",
    "LLMCapabilityError",
    "LLMCancelledError",
    "LLMInterruptedError",
    "LLMSessionError",
    "LLMSessionPausedError",
    "LLMRequest",
    "LLMResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "LLMCapabilities",
    "LLMStreamHandle",
    "LLMSessionHandle",
    "LLMSessionSnapshot",
    "LLMStreamEvent",
    "Message",
    "ToolCall",
    "Usage",
    "StreamMessageStartEvent",
    "StreamTextDeltaEvent",
    "StreamToolCallDeltaEvent",
    "StreamMessageStopEvent",
    "StreamErrorEvent",
    "StreamCompletedEvent",
    "LLMObserver",
    "LLMLifecycleEvent",
    "normalize_json_schema",
    "toolspec_to_openai_tool",
    "tool_to_openai_tool",
    "to_openai_tools",
    "to_openai_tools_from_specs",
    "export_tools_for_provider",
]
