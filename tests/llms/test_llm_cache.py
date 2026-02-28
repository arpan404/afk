"""Tests for LLM cache: InMemoryLLMCache, CacheEntry, and the cache registry."""

from __future__ import annotations

import asyncio
import time

import pytest

from afk.llms.cache.base import CacheEntry
from afk.llms.cache.inmemory import InMemoryLLMCache
from afk.llms.cache import registry
from afk.llms.cache.registry import (
    LLMCacheError,
    create_llm_cache,
    list_llm_cache_backends,
    register_llm_cache_backend,
)
from afk.llms.types import LLMResponse


def run_async(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resp(text: str = "hello") -> LLMResponse:
    return LLMResponse(text=text)


# ---------------------------------------------------------------------------
# InMemoryLLMCache
# ---------------------------------------------------------------------------


class TestInMemoryLLMCache:
    def test_get_returns_none_for_missing_key(self):
        cache = InMemoryLLMCache()
        result = run_async(cache.get("nonexistent"))
        assert result is None

    def test_set_then_get_returns_stored_value(self):
        cache = InMemoryLLMCache()
        resp = _resp("stored")

        async def _run():
            await cache.set("k1", resp, ttl_s=60)
            return await cache.get("k1")

        result = run_async(_run())
        assert result is not None
        assert result.text == "stored"

    def test_get_returns_none_for_expired_entry(self):
        cache = InMemoryLLMCache()
        resp = _resp("ephemeral")

        async def _run():
            await cache.set("k1", resp, ttl_s=0.01)
            await asyncio.sleep(0.05)
            return await cache.get("k1")

        result = run_async(_run())
        assert result is None

    def test_delete_removes_existing_key(self):
        cache = InMemoryLLMCache()
        resp = _resp("deleteme")

        async def _run():
            await cache.set("k1", resp, ttl_s=60)
            await cache.delete("k1")
            return await cache.get("k1")

        result = run_async(_run())
        assert result is None

    def test_delete_is_noop_for_missing_key(self):
        cache = InMemoryLLMCache()
        # Should not raise
        run_async(cache.delete("does_not_exist"))

    def test_multiple_keys_stored_independently(self):
        cache = InMemoryLLMCache()
        r1 = _resp("first")
        r2 = _resp("second")

        async def _run():
            await cache.set("a", r1, ttl_s=60)
            await cache.set("b", r2, ttl_s=60)
            got_a = await cache.get("a")
            got_b = await cache.get("b")
            return got_a, got_b

        got_a, got_b = run_async(_run())
        assert got_a is not None and got_a.text == "first"
        assert got_b is not None and got_b.text == "second"

    def test_overwrite_existing_key_with_new_value(self):
        cache = InMemoryLLMCache()
        r1 = _resp("old")
        r2 = _resp("new")

        async def _run():
            await cache.set("k", r1, ttl_s=60)
            await cache.set("k", r2, ttl_s=60)
            return await cache.get("k")

        result = run_async(_run())
        assert result is not None
        assert result.text == "new"


# ---------------------------------------------------------------------------
# Cache Registry
# ---------------------------------------------------------------------------


class TestCacheRegistry:
    """Each test clears and restores the module-level _REGISTRY to avoid pollution."""

    @pytest.fixture(autouse=True)
    def _isolate_registry(self):
        saved = dict(registry._REGISTRY)
        registry._REGISTRY.clear()
        yield
        registry._REGISTRY.clear()
        registry._REGISTRY.update(saved)

    # -- register_llm_cache_backend --

    def test_register_backend(self):
        backend = InMemoryLLMCache()
        register_llm_cache_backend(backend)
        assert "inmemory" in list_llm_cache_backends()

    def test_register_duplicate_raises(self):
        backend = InMemoryLLMCache()
        register_llm_cache_backend(backend)
        with pytest.raises(LLMCacheError, match="already registered"):
            register_llm_cache_backend(backend)

    def test_register_duplicate_with_overwrite(self):
        b1 = InMemoryLLMCache()
        b2 = InMemoryLLMCache()
        register_llm_cache_backend(b1)
        register_llm_cache_backend(b2, overwrite=True)
        # Should succeed without error; latest registration wins.
        assert "inmemory" in list_llm_cache_backends()

    def test_register_empty_backend_id_raises(self):
        from dataclasses import dataclass

        @dataclass(slots=True)
        class BlankCache:
            backend_id: str = "   "

            async def get(self, key):
                return None

            async def set(self, key, value, *, ttl_s):
                pass

            async def delete(self, key):
                pass

        with pytest.raises(LLMCacheError, match="non-empty"):
            register_llm_cache_backend(BlankCache())

    # -- create_llm_cache --

    def test_create_llm_cache_none_returns_inmemory(self):
        result = create_llm_cache(None)
        assert result.backend_id == "inmemory"

    def test_create_llm_cache_by_string_returns_backend(self):
        backend = InMemoryLLMCache()
        register_llm_cache_backend(backend)
        result = create_llm_cache("inmemory")
        assert result is backend

    def test_create_llm_cache_unknown_raises(self):
        with pytest.raises(LLMCacheError, match="Unknown"):
            create_llm_cache("unknown")

    def test_create_llm_cache_passthrough_instance(self):
        backend = InMemoryLLMCache()
        result = create_llm_cache(backend)
        assert result is backend

    # -- list_llm_cache_backends --

    def test_list_backends_returns_sorted(self):
        from dataclasses import dataclass

        @dataclass(slots=True)
        class CacheZ:
            backend_id: str = "zzz"

            async def get(self, key):
                return None

            async def set(self, key, value, *, ttl_s):
                pass

            async def delete(self, key):
                pass

        @dataclass(slots=True)
        class CacheA:
            backend_id: str = "aaa"

            async def get(self, key):
                return None

            async def set(self, key, value, *, ttl_s):
                pass

            async def delete(self, key):
                pass

        register_llm_cache_backend(CacheZ())
        register_llm_cache_backend(CacheA())
        result = list_llm_cache_backends()
        assert result == ["aaa", "zzz"]


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def test_create_with_required_fields(self):
        resp = _resp("cached")
        entry = CacheEntry(value=resp, expires_at_s=time.time() + 60)
        assert entry.value is resp
        assert entry.expires_at_s > 0

    def test_default_metadata_is_empty_dict(self):
        resp = _resp("cached")
        entry = CacheEntry(value=resp, expires_at_s=time.time() + 60)
        assert entry.metadata == {}
        assert isinstance(entry.metadata, dict)
