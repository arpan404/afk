"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

This module provides the public API for the AFK memory subsystem, including models, stores, and utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .adapters import (
    InMemoryMemoryStore,
    SQLiteMemoryStore,
)
from .factory import create_memory_store_from_env
from .lifecycle import (
    MemoryCompactionResult,
    RetentionPolicy,
    StateRetentionPolicy,
    apply_event_retention,
    apply_state_retention,
    compact_thread_memory,
)
from .store import MemoryCapabilities, MemoryStore
from .types import JsonObject, JsonValue, LongTermMemory, MemoryEvent
from .utils import new_id, now_ms
from .vector import cosine_similarity
from .compaction import (
    BackgroundCompactor,
    CompactionConfig,
    CompactionStats,
    MemoryCompactor,
)

RedisMemoryStore = None  # type: ignore[assignment]
PostgresMemoryStore = None  # type: ignore[assignment]


__all__ = [
    "JsonValue",
    "JsonObject",
    "MemoryEvent",
    "LongTermMemory",
    "now_ms",
    "new_id",
    "MemoryStore",
    "MemoryCapabilities",
    "cosine_similarity",
    "InMemoryMemoryStore",
    "SQLiteMemoryStore",
    "create_memory_store_from_env",
    "RetentionPolicy",
    "StateRetentionPolicy",
    "MemoryCompactionResult",
    "apply_event_retention",
    "apply_state_retention",
    "compact_thread_memory",
    "RedisMemoryStore",
    "PostgresMemoryStore",
    "MemoryCompactor",
    "BackgroundCompactor",
    "CompactionConfig",
    "CompactionStats",
]

try:
    from .adapters.redis import RedisMemoryStore as _RedisMemoryStore
except ModuleNotFoundError:
    pass
else:
    RedisMemoryStore = _RedisMemoryStore

try:
    from .adapters.postgres import PostgresMemoryStore as _PostgresMemoryStore
except ModuleNotFoundError:
    pass
else:
    PostgresMemoryStore = _PostgresMemoryStore

if TYPE_CHECKING:
    # For type checking, import all store classes directly
    from .adapters.postgres import PostgresMemoryStore
    from .adapters.redis import RedisMemoryStore
