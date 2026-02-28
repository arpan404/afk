"""
---
name: Chat History Manager — Config
description: Memory store configuration using create_memory_store_from_env() factory.
tags: [memory, factory, environment, config]
---
---
This module demonstrates create_memory_store_from_env() — an environment-variable-based
factory function that creates the appropriate memory store backend based on the
AFK_MEMORY_BACKEND environment variable. This is the recommended pattern for production:
configure memory via environment variables and let the factory handle instantiation.

Supported backends (via AFK_MEMORY_BACKEND):
  - "inmemory" / "mem"        → InMemoryMemoryStore()
  - "sqlite" / "sqlite3"     → SQLiteMemoryStore(path=AFK_SQLITE_PATH)
  - "redis"                   → RedisMemoryStore(url=AFK_REDIS_URL)
  - "postgres" / "postgresql" → PostgresMemoryStore(dsn=AFK_PG_DSN)

If AFK_MEMORY_BACKEND is not set, defaults to SQLite with "afk_memory.sqlite3".
---
"""

import os  # <- For setting/reading environment variables.
from afk.memory import (  # <- Memory imports.
    create_memory_store_from_env,  # <- Factory function: reads AFK_MEMORY_BACKEND env var and returns the appropriate MemoryStore instance.
    InMemoryMemoryStore,  # <- Fallback: used if the factory isn't desired.
    MemoryEvent,  # <- Event record type for append_event.
    now_ms,  # <- Timestamp helper: returns current time in milliseconds.
    new_id,  # <- ID generator: new_id("prefix") returns "prefix_<random>".
)


# ===========================================================================
# Memory store via environment factory
# ===========================================================================
# In production, set AFK_MEMORY_BACKEND in your environment:
#   export AFK_MEMORY_BACKEND=sqlite
#   export AFK_SQLITE_PATH=chat_history.sqlite3
#
# For this example, we default to "inmemory" if not set so it runs
# without external dependencies.

if not os.environ.get("AFK_MEMORY_BACKEND"):
    os.environ["AFK_MEMORY_BACKEND"] = "inmemory"  # <- Default to in-memory for this example. Remove this line to let the factory use its default (sqlite).

memory = create_memory_store_from_env()  # <- The factory reads AFK_MEMORY_BACKEND and related env vars (AFK_SQLITE_PATH, AFK_REDIS_URL, AFK_PG_DSN, etc.) and returns the appropriate MemoryStore instance. Same API regardless of backend.

# Print which backend was selected
backend = os.environ.get("AFK_MEMORY_BACKEND", "default")
print(f"  Memory backend: {backend} (set AFK_MEMORY_BACKEND to change)")
