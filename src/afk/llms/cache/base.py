"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Module: cache/base.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol

from ..types import JSONValue, LLMResponse


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """One cached LLM response row with expiration metadata."""

    value: LLMResponse
    expires_at_s: float
    metadata: dict[str, JSONValue] = field(default_factory=dict)


class LLMCacheBackend(Protocol):
    """Protocol implemented by cache backends used in runtime client."""

    backend_id: str

    async def get(self, key: str) -> LLMResponse | None: ...

    async def set(self, key: str, value: LLMResponse, *, ttl_s: float) -> None: ...

    async def delete(self, key: str) -> None: ...


def cache_safe_response(value: LLMResponse) -> LLMResponse:
    """Strip request-scoped provider metadata before storing a reusable cache row."""
    return replace(
        value,
        request_id=None,
        provider_request_id=None,
        session_token=None,
        checkpoint_token=None,
        raw={},
    )
