"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

This module provides a PostgreSQL + pgvector memory backend for production workloads.
"""

from __future__ import annotations


from typing import Any, Optional, Sequence, cast

try:
    import asyncpg
except Exception:  # pragma: no cover - optional native dependency
    asyncpg = None  # type: ignore[assignment]

from afk.memory.types import (
    JsonObject,
    JsonValue,
    LongTermMemory,
    MemoryEvent,
)
from afk.memory.utils import json_dumps, now_ms
from afk.memory.vector import format_pgvector
from afk.memory.store import MemoryCapabilities, MemoryStore


class PostgresMemoryStore(MemoryStore):
    """Production-grade memory store with JSONB and pgvector similarity search."""

    capabilities = MemoryCapabilities(
        text_search=True, vector_search=True, atomic_upsert=True, ttl=False
    )

    def __init__(
        self,
        *,
        dsn: str,
        vector_dim: int,
        pool_min: int = 1,
        pool_max: int = 10,
        ssl: bool = False,
    ) -> None:
        super().__init__()
        self.dsn = dsn
        self.vector_dim = int(vector_dim)
        self.pool_min = pool_min
        self.pool_max = pool_max
        self.ssl = ssl
        self._pool: Any | None = None

    async def setup(self) -> None:
        if asyncpg is None:
            raise RuntimeError(
                "asyncpg is required for PostgresMemoryStore but is not available"
            )
        if self._is_setup and self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=self.dsn,
            min_size=self.pool_min,
            max_size=self.pool_max,
            ssl=self.ssl if self.ssl else None,
        )
        await self._create_schema()
        await super().setup()

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        await super().close()

    def _pool_required(self) -> Any:
        if self._pool is None:
            raise RuntimeError(
                "PostgresMemoryStore is not initialized. Call setup() first."
            )
        return self._pool

    async def _create_schema(self) -> None:
        pool = self._pool_required()
        async with pool.acquire() as connection:
            await connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                  id TEXT PRIMARY KEY,
                  thread_id TEXT NOT NULL,
                  user_id TEXT,
                  type TEXT NOT NULL,
                  timestamp BIGINT NOT NULL,
                  payload_json JSONB NOT NULL,
                  tags_json JSONB NOT NULL
                );
                """,
            )
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_thread_time ON events(thread_id, timestamp DESC);"
            )
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS state_kv (
                  thread_id TEXT NOT NULL,
                  key TEXT NOT NULL,
                  value_json JSONB NOT NULL,
                  updated_at BIGINT NOT NULL,
                  PRIMARY KEY(thread_id, key)
                );
                """,
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                  id TEXT PRIMARY KEY,
                  user_id TEXT,
                  scope TEXT NOT NULL,
                  data_json JSONB NOT NULL,
                  text TEXT,
                  tags_json JSONB NOT NULL,
                  metadata_json JSONB NOT NULL,
                  created_at BIGINT NOT NULL,
                  updated_at BIGINT NOT NULL,
                  embedding vector({self.vector_dim})
                );
                """,
            )
            await connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_ltm_user_scope_time ON long_term_memory(user_id, scope, updated_at DESC);"
            )
            try:
                await connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_ltm_embedding_cosine
                    ON long_term_memory
                    USING hnsw (embedding vector_cosine_ops);
                    """,
                )
            except Exception:
                # Older pgvector builds may not support hnsw. Core functionality remains intact.
                pass

    async def append_event(self, event: MemoryEvent) -> None:
        self._ensure_setup()
        pool = self._pool_required()
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO events (id, thread_id, user_id, type, timestamp, payload_json, tags_json)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb)
                """,
                event.id,
                event.thread_id,
                event.user_id,
                event.type,
                event.timestamp,
                json_dumps(event.payload),
                json_dumps(event.tags),
            )

    async def get_recent_events(
        self, thread_id: str, limit: int = 50
    ) -> list[MemoryEvent]:
        self._ensure_setup()
        pool = self._pool_required()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT * FROM events WHERE thread_id=$1 ORDER BY timestamp DESC LIMIT $2",
                thread_id,
                limit,
            )
        return [self._record_to_event(record) for record in reversed(rows)]

    async def get_events_since(
        self, thread_id: str, since_ms: int, limit: int = 500
    ) -> list[MemoryEvent]:
        self._ensure_setup()
        pool = self._pool_required()
        async with pool.acquire() as connection:
            rows = await connection.fetch(
                "SELECT * FROM events WHERE thread_id=$1 AND timestamp>=$2 ORDER BY timestamp ASC LIMIT $3",
                thread_id,
                since_ms,
                limit,
            )
        return [self._record_to_event(record) for record in rows]

    async def put_state(self, thread_id: str, key: str, value: JsonValue) -> None:
        self._ensure_setup()
        pool = self._pool_required()
        async with pool.acquire() as connection:
            await connection.execute(
                """
                INSERT INTO state_kv (thread_id, key, value_json, updated_at)
                VALUES ($1, $2, $3::jsonb, $4)
                ON CONFLICT(thread_id, key) DO UPDATE SET
                  value_json=EXCLUDED.value_json,
                  updated_at=EXCLUDED.updated_at
                """,
                thread_id,
                key,
                json_dumps(value),
                now_ms(),
            )

    async def delete_state(self, thread_id: str, key: str) -> None:
        self._ensure_setup()
        pool = self._pool_required()
        async with pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM state_kv WHERE thread_id=$1 AND key=$2",
                thread_id,
                key,
            )

    async def get_state(self, thread_id: str, key: str) -> JsonValue | None:
        self._ensure_setup()
        pool = self._pool_required()
        async with pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT value_json FROM state_kv WHERE thread_id=$1 AND key=$2",
                thread_id,
                key,
            )
        if row is None:
            return None
        return cast(JsonValue, row["value_json"])

    async def list_state(
        self, thread_id: str, prefix: str | None = None
    ) -> dict[str, JsonValue]:
        self._ensure_setup()
        pool = self._pool_required()
        async with pool.acquire() as connection:
            if prefix:
                rows = await connection.fetch(
                    "SELECT key, value_json FROM state_kv WHERE thread_id=$1 AND key LIKE $2 ORDER BY key ASC",
                    thread_id,
                    f"{prefix}%",
                )
            else:
                rows = await connection.fetch(
                    "SELECT key, value_json FROM state_kv WHERE thread_id=$1 ORDER BY key ASC",
                    thread_id,
                )
        return {
            cast(str, row["key"]): cast(JsonValue, row["value_json"]) for row in rows
        }

    async def replace_thread_events(
        self,
        thread_id: str,
        events: list[MemoryEvent],
    ) -> None:
        self._ensure_setup()
        pool = self._pool_required()
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "DELETE FROM events WHERE thread_id=$1",
                    thread_id,
                )
                for event in events:
                    await connection.execute(
                        """
                        INSERT INTO events (id, thread_id, user_id, type, timestamp, payload_json, tags_json)
                        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb)
                        """,
                        event.id,
                        event.thread_id,
                        event.user_id,
                        event.type,
                        event.timestamp,
                        json_dumps(event.payload),
                        json_dumps(event.tags),
                    )

    async def upsert_long_term_memory(
        self,
        memory: LongTermMemory,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> None:
        self._ensure_setup()
        pool = self._pool_required()
        async with pool.acquire() as connection:
            if embedding is None:
                await connection.execute(
                    """
                    INSERT INTO long_term_memory (id, user_id, scope, data_json, text, tags_json, metadata_json, created_at, updated_at, embedding)
                    VALUES ($1, $2, $3, $4::jsonb, $5, $6::jsonb, $7::jsonb, $8, $9, NULL)
                    ON CONFLICT(id) DO UPDATE SET
                      user_id=EXCLUDED.user_id,
                      scope=EXCLUDED.scope,
                      data_json=EXCLUDED.data_json,
                      text=EXCLUDED.text,
                      tags_json=EXCLUDED.tags_json,
                      metadata_json=EXCLUDED.metadata_json,
                      created_at=long_term_memory.created_at,
                      updated_at=EXCLUDED.updated_at,
                      embedding=COALESCE(long_term_memory.embedding, EXCLUDED.embedding)
                    """,
                    memory.id,
                    memory.user_id,
                    memory.scope,
                    json_dumps(memory.data),
                    memory.text,
                    json_dumps(memory.tags),
                    json_dumps(memory.metadata),
                    memory.created_at,
                    memory.updated_at,
                )
            else:
                await connection.execute(
                    f"""
                    INSERT INTO long_term_memory (id, user_id, scope, data_json, text, tags_json, metadata_json, created_at, updated_at, embedding)
                    VALUES ($1, $2, $3, $4::jsonb, $5, $6::jsonb, $7::jsonb, $8, $9, $10::vector({self.vector_dim}))
                    ON CONFLICT(id) DO UPDATE SET
                      user_id=EXCLUDED.user_id,
                      scope=EXCLUDED.scope,
                      data_json=EXCLUDED.data_json,
                      text=EXCLUDED.text,
                      tags_json=EXCLUDED.tags_json,
                      metadata_json=EXCLUDED.metadata_json,
                      created_at=long_term_memory.created_at,
                      updated_at=EXCLUDED.updated_at,
                      embedding=COALESCE(EXCLUDED.embedding, long_term_memory.embedding)
                    """,
                    memory.id,
                    memory.user_id,
                    memory.scope,
                    json_dumps(memory.data),
                    memory.text,
                    json_dumps(memory.tags),
                    json_dumps(memory.metadata),
                    memory.created_at,
                    memory.updated_at,
                    format_pgvector(embedding),
                )

    async def delete_long_term_memory(
        self, user_id: str | None, memory_id: str
    ) -> None:
        self._ensure_setup()
        pool = self._pool_required()
        async with pool.acquire() as connection:
            await connection.execute(
                "DELETE FROM long_term_memory WHERE id=$1 AND user_id IS NOT DISTINCT FROM $2",
                memory_id,
                user_id,
            )

    async def list_long_term_memories(
        self,
        user_id: str | None,
        *,
        scope: str | None = None,
        limit: int = 100,
    ) -> list[LongTermMemory]:
        self._ensure_setup()
        pool = self._pool_required()
        async with pool.acquire() as connection:
            if scope is None:
                rows = await connection.fetch(
                    """
                    SELECT * FROM long_term_memory
                    WHERE user_id IS NOT DISTINCT FROM $1
                    ORDER BY updated_at DESC
                    LIMIT $2
                    """,
                    user_id,
                    limit,
                )
            else:
                rows = await connection.fetch(
                    """
                    SELECT * FROM long_term_memory
                    WHERE user_id IS NOT DISTINCT FROM $1 AND scope=$2
                    ORDER BY updated_at DESC
                    LIMIT $3
                    """,
                    user_id,
                    scope,
                    limit,
                )
        return [self._record_to_memory(record) for record in rows]

    async def search_long_term_memory_text(
        self,
        user_id: str | None,
        query: str,
        *,
        scope: str | None = None,
        limit: int = 20,
    ) -> list[LongTermMemory]:
        self._ensure_setup()
        normalized_query = query.strip()
        if not normalized_query:
            return []
        pattern = f"%{normalized_query}%"
        pool = self._pool_required()
        async with pool.acquire() as connection:
            if scope is None:
                rows = await connection.fetch(
                    """
                    SELECT * FROM long_term_memory
                    WHERE user_id IS NOT DISTINCT FROM $1
                    AND (
                        text ILIKE $2 OR
                        tags_json::text ILIKE $2 OR
                        data_json::text ILIKE $2
                    )
                    ORDER BY updated_at DESC
                    LIMIT $3
                    """,
                    user_id,
                    pattern,
                    limit,
                )
            else:
                rows = await connection.fetch(
                    """
                    SELECT * FROM long_term_memory
                    WHERE user_id IS NOT DISTINCT FROM $1
                    AND scope = $2
                    AND (
                        text ILIKE $3 OR
                        tags_json::text ILIKE $3 OR
                        data_json::text ILIKE $3
                    )
                    ORDER BY updated_at DESC
                    LIMIT $4
                    """,
                    user_id,
                    scope,
                    pattern,
                    limit,
                )
        return [self._record_to_memory(record) for record in rows]

    async def search_long_term_memory_vector(
        self,
        user_id: str | None,
        query_embedding: Sequence[float],
        *,
        scope: str | None = None,
        limit: int = 20,
        min_score: float | None = None,
    ) -> list[tuple[LongTermMemory, float]]:
        self._ensure_setup()
        pool = self._pool_required()
        query_vector = format_pgvector(query_embedding)

        async with pool.acquire() as connection:
            if scope is None:
                rows = await connection.fetch(
                    f"""
                    SELECT *, (1 - (embedding <=> $2::vector({self.vector_dim}))) AS similarity
                    FROM long_term_memory
                    WHERE user_id IS NOT DISTINCT FROM $1
                    AND embedding IS NOT NULL
                    ORDER BY similarity DESC
                    LIMIT $3
                    """,
                    user_id,
                    query_vector,
                    limit,
                )
            else:
                rows = await connection.fetch(
                    f"""
                    SELECT *, (1 - (embedding <=> $3::vector({self.vector_dim}))) AS similarity
                    FROM long_term_memory
                    WHERE user_id IS NOT DISTINCT FROM $1
                    AND scope=$2
                    AND embedding IS NOT NULL
                    ORDER BY similarity DESC
                    LIMIT $4
                    """,
                    user_id,
                    scope,
                    query_vector,
                    limit,
                )

        ranked: list[tuple[LongTermMemory, float]] = []
        for record in rows:
            similarity = float(cast(float, record["similarity"]))
            if min_score is not None and similarity < min_score:
                continue
            ranked.append((self._record_to_memory(record), similarity))
        return ranked

    @staticmethod
    def _record_to_event(record: asyncpg.Record) -> MemoryEvent:
        return MemoryEvent(
            id=cast(str, record["id"]),
            thread_id=cast(str, record["thread_id"]),
            user_id=cast(Optional[str], record["user_id"]),
            type=cast(str, record["type"]),
            timestamp=int(cast(int, record["timestamp"])),
            payload=cast(JsonObject, record["payload_json"]),
            tags=cast(list[str], record["tags_json"]),
        )

    @staticmethod
    def _record_to_memory(record: asyncpg.Record) -> LongTermMemory:
        return LongTermMemory(
            id=cast(str, record["id"]),
            user_id=cast(Optional[str], record["user_id"]),
            scope=cast(str, record["scope"]),
            data=cast(JsonObject, record["data_json"]),
            text=cast(Optional[str], record["text"]),
            tags=cast(list[str], record["tags_json"]),
            metadata=cast(JsonObject, record["metadata_json"]),
            created_at=int(cast(int, record["created_at"])),
            updated_at=int(cast(int, record["updated_at"])),
        )
