"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

This module provides factory functions for creating memory store backends based on environment variables.
"""

from __future__ import annotations

import os

from afk.memory.adapters.in_memory import InMemoryMemoryStore
from afk.memory.adapters.sqlite import SQLiteMemoryStore
from afk.memory.store import MemoryStore


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable with common truthy values."""
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def create_memory_store_from_env() -> MemoryStore:
    """Create a memory store based on `AFK_MEMORY_BACKEND` and related environment settings."""
    backend = os.getenv("AFK_MEMORY_BACKEND", "memory").strip().lower()

    if backend in ("mem", "memory", "inmemory", "in_memory"):
        return InMemoryMemoryStore()

    if backend in ("sqlite", "sqlite3"):
        path = os.getenv("AFK_SQLITE_PATH", "afk_memory.sqlite3")
        return SQLiteMemoryStore(path=path)

    if backend in ("redis",):
        from afk.memory.adapters.redis import RedisMemoryStore

        url = os.getenv("AFK_REDIS_URL")
        if not url:
            host = os.getenv("AFK_REDIS_HOST", "localhost")
            port = os.getenv("AFK_REDIS_PORT", "6379")
            db = os.getenv("AFK_REDIS_DB", "0")
            password = os.getenv("AFK_REDIS_PASSWORD", "")
            url = (
                f"redis://:{password}@{host}:{port}/{db}"
                if password
                else f"redis://{host}:{port}/{db}"
            )
        max_events = int(os.getenv("AFK_REDIS_EVENTS_MAX", "2000"))
        return RedisMemoryStore(url=url, events_max_per_thread=max_events)

    if backend in ("pg", "postgres", "postgresql"):
        from afk.memory.adapters.postgres import PostgresMemoryStore

        dsn = os.getenv("AFK_PG_DSN")
        if not dsn:
            host = os.getenv("AFK_PG_HOST", "localhost")
            port = os.getenv("AFK_PG_PORT", "5432")
            user = os.getenv("AFK_PG_USER", "postgres")
            password = os.getenv("AFK_PG_PASSWORD", "")
            db = os.getenv("AFK_PG_DB", "afk")
            auth = f"{user}:{password}@" if password else f"{user}@"
            dsn = f"postgresql://{auth}{host}:{port}/{db}"

        ssl = _env_bool("AFK_PG_SSL", False)
        pool_min = int(os.getenv("AFK_PG_POOL_MIN", "1"))
        pool_max = int(os.getenv("AFK_PG_POOL_MAX", "10"))

        dim = os.getenv("AFK_VECTOR_DIM")
        if not dim:
            raise ValueError(
                "AFK_VECTOR_DIM is required for Postgres vector search (e.g. 1536)."
            )

        return PostgresMemoryStore(
            dsn=dsn, vector_dim=int(dim), pool_min=pool_min, pool_max=pool_max, ssl=ssl
        )

    raise ValueError(f"Unknown AFK_MEMORY_BACKEND: {backend}")
