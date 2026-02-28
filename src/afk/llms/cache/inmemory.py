"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Module: cache/inmemory.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..types import LLMResponse
from .base import CacheEntry, LLMCacheBackend


@dataclass(slots=True)
class InMemoryLLMCache(LLMCacheBackend):
    """Process-local cache backend suitable for development/test workloads."""

    backend_id: str = "inmemory"
    max_size: int = 1024

    def __post_init__(self) -> None:
        self._rows: dict[str, CacheEntry] = {}

    async def get(self, key: str) -> LLMResponse | None:
        row = self._rows.get(key)
        if row is None:
            return None
        if row.expires_at_s < time.time():
            self._rows.pop(key, None)
            return None
        return row.value

    async def set(self, key: str, value: LLMResponse, *, ttl_s: float) -> None:
        now = time.time()
        # Evict expired entries first.
        expired = [k for k, v in self._rows.items() if v.expires_at_s < now]
        for k in expired:
            del self._rows[k]
        # If still over capacity, evict oldest entries by expiry time.
        if len(self._rows) >= self.max_size:
            by_expiry = sorted(self._rows.items(), key=lambda kv: kv[1].expires_at_s)
            to_remove = len(self._rows) - self.max_size + 1
            for k, _ in by_expiry[:to_remove]:
                del self._rows[k]
        self._rows[key] = CacheEntry(value=value, expires_at_s=now + ttl_s)

    async def delete(self, key: str) -> None:
        self._rows.pop(key, None)
