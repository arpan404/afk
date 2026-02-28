"""
Comprehensive tests for the LLM routing layer:
  - OrderedFallbackRouter (deterministic provider ordering)
  - Routing registry (register, create, list, error handling)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from afk.llms.routing.defaults import OrderedFallbackRouter
from afk.llms.routing.registry import (
    LLMRouterError,
    create_llm_router,
    list_llm_routers,
    register_llm_router,
)
from afk.llms.routing import registry
from afk.llms.runtime.contracts import RoutePolicy
from afk.llms.types import LLMRequest


def run_async(coro):
    return asyncio.run(coro)


# ======================== OrderedFallbackRouter =============================


class TestOrderedFallbackRouter:
    """Tests for the default ordered-fallback routing strategy."""

    def setup_method(self):
        self.router = OrderedFallbackRouter()

    # --- router_id ---

    def test_router_id_is_ordered_fallback(self):
        assert self.router.router_id == "ordered_fallback"

    # --- no route_policy on request ---

    def test_returns_default_provider_when_no_route_policy(self):
        req = LLMRequest(model="gpt-4")
        result = self.router.route(
            req,
            available_providers=["openai", "anthropic"],
            default_provider="openai",
        )
        assert result == ["openai"]

    def test_returns_default_provider_only_if_available(self):
        req = LLMRequest(model="gpt-4")
        result = self.router.route(
            req,
            available_providers=["anthropic"],
            default_provider="openai",
        )
        assert result == []

    # --- route_policy.provider_order honored ---

    def test_honors_provider_order_from_route_policy(self):
        policy = RoutePolicy(provider_order=("anthropic", "openai"))
        req = LLMRequest(model="gpt-4", route_policy=policy)
        result = self.router.route(
            req,
            available_providers=["openai", "anthropic"],
            default_provider="openai",
        )
        assert result == ["anthropic", "openai"]

    def test_provider_order_places_requested_before_default(self):
        policy = RoutePolicy(provider_order=("anthropic",))
        req = LLMRequest(model="gpt-4", route_policy=policy)
        result = self.router.route(
            req,
            available_providers=["openai", "anthropic"],
            default_provider="openai",
        )
        assert result == ["anthropic", "openai"]

    # --- filters out providers not in available_providers ---

    def test_filters_unavailable_providers(self):
        policy = RoutePolicy(provider_order=("azure", "anthropic"))
        req = LLMRequest(model="gpt-4", route_policy=policy)
        result = self.router.route(
            req,
            available_providers=["openai", "anthropic"],
            default_provider="openai",
        )
        # "azure" is not available so it is dropped; "anthropic" stays,
        # then default "openai" is appended
        assert result == ["anthropic", "openai"]

    def test_all_requested_unavailable_falls_back_to_default(self):
        policy = RoutePolicy(provider_order=("azure", "gcp"))
        req = LLMRequest(model="gpt-4", route_policy=policy)
        result = self.router.route(
            req,
            available_providers=["openai"],
            default_provider="openai",
        )
        assert result == ["openai"]

    # --- deduplication ---

    def test_deduplicates_providers_in_order(self):
        policy = RoutePolicy(provider_order=("openai", "anthropic", "openai"))
        req = LLMRequest(model="gpt-4", route_policy=policy)
        result = self.router.route(
            req,
            available_providers=["openai", "anthropic"],
            default_provider="openai",
        )
        # "openai" appears twice in provider_order; only first occurrence kept
        assert result == ["openai", "anthropic"]

    def test_deduplicates_default_when_also_in_provider_order(self):
        policy = RoutePolicy(provider_order=("openai",))
        req = LLMRequest(model="gpt-4", route_policy=policy)
        result = self.router.route(
            req,
            available_providers=["openai", "anthropic"],
            default_provider="openai",
        )
        # default "openai" already included via provider_order; not duplicated
        assert result == ["openai"]

    # --- strip/lowercase ---

    def test_strips_and_lowercases_provider_names(self):
        policy = RoutePolicy(provider_order=("  OpenAI  ", " Anthropic"))
        req = LLMRequest(model="gpt-4", route_policy=policy)
        result = self.router.route(
            req,
            available_providers=["openai", "anthropic"],
            default_provider="openai",
        )
        assert result == ["openai", "anthropic"]

    def test_strips_whitespace_only_entries(self):
        policy = RoutePolicy(provider_order=("  ", "anthropic"))
        req = LLMRequest(model="gpt-4", route_policy=policy)
        result = self.router.route(
            req,
            available_providers=["openai", "anthropic"],
            default_provider="openai",
        )
        # whitespace-only entry is skipped; "anthropic" + default "openai"
        assert result == ["anthropic", "openai"]

    # --- empty provider_order falls back to default ---

    def test_empty_provider_order_falls_back_to_default(self):
        policy = RoutePolicy(provider_order=())
        req = LLMRequest(model="gpt-4", route_policy=policy)
        result = self.router.route(
            req,
            available_providers=["openai", "anthropic"],
            default_provider="openai",
        )
        assert result == ["openai"]

    def test_route_policy_with_empty_tuple_same_as_none(self):
        policy = RoutePolicy(provider_order=())
        req_with = LLMRequest(model="gpt-4", route_policy=policy)
        req_without = LLMRequest(model="gpt-4")
        avail = ["openai", "anthropic"]
        default = "openai"

        result_with = self.router.route(
            req_with, available_providers=avail, default_provider=default
        )
        result_without = self.router.route(
            req_without, available_providers=avail, default_provider=default
        )
        assert result_with == result_without

    # --- default provider appended after requested ---

    def test_default_provider_appended_after_requested(self):
        policy = RoutePolicy(provider_order=("anthropic",))
        req = LLMRequest(model="gpt-4", route_policy=policy)
        result = self.router.route(
            req,
            available_providers=["openai", "anthropic"],
            default_provider="openai",
        )
        assert result == ["anthropic", "openai"]
        assert result[0] == "anthropic"
        assert result[-1] == "openai"

    def test_default_provider_not_appended_when_unavailable(self):
        policy = RoutePolicy(provider_order=("anthropic",))
        req = LLMRequest(model="gpt-4", route_policy=policy)
        result = self.router.route(
            req,
            available_providers=["anthropic"],
            default_provider="openai",
        )
        # default "openai" is not available
        assert result == ["anthropic"]


# ========================== Routing Registry ================================


class TestRoutingRegistry:
    """Tests for register_llm_router, create_llm_router, list_llm_routers."""

    def setup_method(self):
        """Clear the module-level registry before each test."""
        registry._REGISTRY.clear()

    def teardown_method(self):
        """Clear after each test to avoid cross-contamination."""
        registry._REGISTRY.clear()

    # --- register_llm_router ---

    def test_register_llm_router_registers_a_router(self):
        router = OrderedFallbackRouter()
        register_llm_router(router)
        assert "ordered_fallback" in registry._REGISTRY
        assert registry._REGISTRY["ordered_fallback"] is router

    def test_register_llm_router_raises_on_duplicate(self):
        router = OrderedFallbackRouter()
        register_llm_router(router)
        with pytest.raises(LLMRouterError, match="already registered"):
            register_llm_router(router)

    def test_register_llm_router_allows_overwrite(self):
        router1 = OrderedFallbackRouter()
        router2 = OrderedFallbackRouter()
        register_llm_router(router1)
        register_llm_router(router2, overwrite=True)
        assert registry._REGISTRY["ordered_fallback"] is router2

    def test_register_llm_router_raises_on_empty_router_id(self):
        @dataclass(slots=True)
        class EmptyIdRouter:
            router_id: str = ""
            def route(self, req, *, available_providers, default_provider):
                return []

        router = EmptyIdRouter()
        with pytest.raises(LLMRouterError, match="non-empty"):
            register_llm_router(router)

    def test_register_llm_router_raises_on_whitespace_only_router_id(self):
        @dataclass(slots=True)
        class WhitespaceIdRouter:
            router_id: str = "   "
            def route(self, req, *, available_providers, default_provider):
                return []

        router = WhitespaceIdRouter()
        with pytest.raises(LLMRouterError, match="non-empty"):
            register_llm_router(router)

    # --- create_llm_router ---

    def test_create_llm_router_none_returns_default(self):
        router = create_llm_router(None)
        assert isinstance(router, OrderedFallbackRouter)
        assert router.router_id == "ordered_fallback"

    def test_create_llm_router_none_registers_default(self):
        """Calling with None also registers the default instance."""
        router = create_llm_router(None)
        assert "ordered_fallback" in registry._REGISTRY
        assert registry._REGISTRY["ordered_fallback"] is router

    def test_create_llm_router_none_returns_cached_instance(self):
        """Subsequent calls with None return the same cached instance."""
        router1 = create_llm_router(None)
        router2 = create_llm_router(None)
        assert router1 is router2

    def test_create_llm_router_by_id_returns_registered(self):
        router = OrderedFallbackRouter()
        register_llm_router(router)
        resolved = create_llm_router("ordered_fallback")
        assert resolved is router

    def test_create_llm_router_unknown_raises(self):
        with pytest.raises(LLMRouterError, match="Unknown LLM router"):
            create_llm_router("unknown")

    def test_create_llm_router_instance_passes_through(self):
        router = OrderedFallbackRouter()
        result = create_llm_router(router)
        assert result is router

    def test_create_llm_router_instance_not_registered(self):
        """Passing an instance directly does not auto-register it."""
        router = OrderedFallbackRouter()
        create_llm_router(router)
        assert "ordered_fallback" not in registry._REGISTRY

    # --- list_llm_routers ---

    def test_list_llm_routers_returns_sorted(self):
        @dataclass(slots=True)
        class RouterA:
            router_id: str = "zebra"
            def route(self, req, *, available_providers, default_provider):
                return []

        @dataclass(slots=True)
        class RouterB:
            router_id: str = "alpha"
            def route(self, req, *, available_providers, default_provider):
                return []

        register_llm_router(RouterA())
        register_llm_router(RouterB())

        result = list_llm_routers()
        assert result == ["alpha", "zebra"]

    def test_list_llm_routers_empty_when_clean(self):
        assert list_llm_routers() == []

    def test_list_llm_routers_reflects_registrations(self):
        assert list_llm_routers() == []
        router = OrderedFallbackRouter()
        register_llm_router(router)
        assert list_llm_routers() == ["ordered_fallback"]
