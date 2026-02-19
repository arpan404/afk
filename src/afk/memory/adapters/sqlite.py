"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

This module provides a SQLite memory backend with JSON persistence and local vector search.
"""

from __future__ import annotations


from typing import Optional, Sequence, cast

import aiosqlite
import numpy as np

from afk.memory.types import (
    JsonObject,
    JsonValue,
    LongTermMemory,
    MemoryEvent,
)
from afk.memory.utils import json_dumps, json_loads, now_ms
from afk.memory.vector import cosine_similarity
from afk.memory.store import MemoryCapabilities, MemoryStore


class SQLiteMemoryStore(MemoryStore):
    """Persistent local memory backend backed by SQLite."""

    capabilities = MemoryCapabilities(
        text_search=True, vector_search=True, atomic_upsert=True, ttl=False
    )

    def __init__(self, path: str = "afk_memory.sqlite3") -> None:
        super().__init__()
        self.path = path
        self._connection: aiosqlite.Connection | None = None

    async def setup(self) -> None:
        if self._is_setup and self._connection is not None:
            return
        self._connection = await aiosqlite.connect(self.path)
        self._connection.row_factory = aiosqlite.Row
        await self._connection.execute("PRAGMA journal_mode=WAL;")
        await self._connection.execute("PRAGMA synchronous=NORMAL;")
        await self._connection.execute("PRAGMA foreign_keys=ON;")
        await self._create_tables()
        await self._connection.commit()
        await super().setup()

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
        await super().close()

    def _db(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError(
                "SQLiteMemoryStore is not initialized. Call setup() first."
            )
        return self._connection

    async def _create_tables(self) -> None:
        db = self._db()
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
              id TEXT PRIMARY KEY,
              thread_id TEXT NOT NULL,
              user_id TEXT,
              type TEXT NOT NULL,
              timestamp INTEGER NOT NULL,
              payload_json TEXT NOT NULL,
              tags_json TEXT NOT NULL
            );
            """,
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_thread_time ON events(thread_id, timestamp DESC);"
        )

        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS state_kv (
              thread_id TEXT NOT NULL,
              key TEXT NOT NULL,
              value_json TEXT NOT NULL,
              updated_at INTEGER NOT NULL,
              PRIMARY KEY(thread_id, key)
            );
            """,
        )

        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS long_term_memory (
              id TEXT PRIMARY KEY,
              user_id TEXT,
              scope TEXT NOT NULL,
              data_json TEXT NOT NULL,
              text TEXT,
              tags_json TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL,
              embedding_json TEXT
            );
            """,
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_ltm_user_scope_time ON long_term_memory(user_id, scope, updated_at DESC);"
        )

    @staticmethod
    def _user_filter_sql(column_name: str = "user_id") -> str:
        return f"(({column_name} IS NULL AND ? IS NULL) OR {column_name} = ?)"

    async def append_event(self, event: MemoryEvent) -> None:
        self._ensure_setup()
        db = self._db()
        await db.execute(
            """
            INSERT INTO events (id, thread_id, user_id, type, timestamp, payload_json, tags_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.thread_id,
                event.user_id,
                event.type,
                event.timestamp,
                json_dumps(event.payload),
                json_dumps(event.tags),
            ),
        )
        await db.commit()

    async def get_recent_events(
        self, thread_id: str, limit: int = 50
    ) -> list[MemoryEvent]:
        self._ensure_setup()
        db = self._db()
        cursor = await db.execute(
            "SELECT * FROM events WHERE thread_id=? ORDER BY timestamp DESC LIMIT ?",
            (thread_id, limit),
        )
        rows = await cursor.fetchall()
        events = [self._row_to_event(row) for row in rows]
        return events[::-1]

    async def get_events_since(
        self, thread_id: str, since_ms: int, limit: int = 500
    ) -> list[MemoryEvent]:
        self._ensure_setup()
        db = self._db()
        cursor = await db.execute(
            "SELECT * FROM events WHERE thread_id=? AND timestamp>=? ORDER BY timestamp ASC LIMIT ?",
            (thread_id, since_ms, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_event(row) for row in rows]

    async def put_state(self, thread_id: str, key: str, value: JsonValue) -> None:
        self._ensure_setup()
        db = self._db()
        await db.execute(
            """
            INSERT INTO state_kv (thread_id, key, value_json, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id, key) DO UPDATE SET
              value_json=excluded.value_json,
              updated_at=excluded.updated_at
            """,
            (thread_id, key, json_dumps(value), now_ms()),
        )
        await db.commit()

    async def delete_state(self, thread_id: str, key: str) -> None:
        self._ensure_setup()
        db = self._db()
        await db.execute(
            "DELETE FROM state_kv WHERE thread_id=? AND key=?",
            (thread_id, key),
        )
        await db.commit()

    async def get_state(self, thread_id: str, key: str) -> JsonValue | None:
        self._ensure_setup()
        db = self._db()
        cursor = await db.execute(
            "SELECT value_json FROM state_kv WHERE thread_id=? AND key=?",
            (thread_id, key),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return json_loads(cast(str, row["value_json"]))

    async def list_state(
        self, thread_id: str, prefix: str | None = None
    ) -> dict[str, JsonValue]:
        self._ensure_setup()
        db = self._db()
        if prefix:
            cursor = await db.execute(
                "SELECT key, value_json FROM state_kv WHERE thread_id=? AND key LIKE ? ORDER BY key ASC",
                (thread_id, f"{prefix}%"),
            )
        else:
            cursor = await db.execute(
                "SELECT key, value_json FROM state_kv WHERE thread_id=? ORDER BY key ASC",
                (thread_id,),
            )
        rows = await cursor.fetchall()
        return {
            cast(str, row["key"]): json_loads(cast(str, row["value_json"]))
            for row in rows
        }

    async def replace_thread_events(
        self,
        thread_id: str,
        events: list[MemoryEvent],
    ) -> None:
        self._ensure_setup()
        db = self._db()
        await db.execute("DELETE FROM events WHERE thread_id=?", (thread_id,))
        for event in events:
            await db.execute(
                """
                INSERT INTO events (id, thread_id, user_id, type, timestamp, payload_json, tags_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.thread_id,
                    event.user_id,
                    event.type,
                    event.timestamp,
                    json_dumps(event.payload),
                    json_dumps(event.tags),
                ),
            )
        await db.commit()

    async def upsert_long_term_memory(
        self,
        memory: LongTermMemory,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> None:
        self._ensure_setup()
        db = self._db()
        serialized_embedding = (
            None
            if embedding is None
            else json_dumps([float(value) for value in embedding])
        )
        await db.execute(
            """
            INSERT INTO long_term_memory (id, user_id, scope, data_json, text, tags_json, metadata_json, created_at, updated_at, embedding_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              user_id=excluded.user_id,
              scope=excluded.scope,
              data_json=excluded.data_json,
              text=excluded.text,
              tags_json=excluded.tags_json,
              metadata_json=excluded.metadata_json,
              created_at=long_term_memory.created_at,
              updated_at=excluded.updated_at,
              embedding_json=COALESCE(excluded.embedding_json, long_term_memory.embedding_json)
            """,
            (
                memory.id,
                memory.user_id,
                memory.scope,
                json_dumps(memory.data),
                memory.text,
                json_dumps(memory.tags),
                json_dumps(memory.metadata),
                memory.created_at,
                memory.updated_at,
                serialized_embedding,
            ),
        )
        await db.commit()

    async def delete_long_term_memory(
        self, user_id: str | None, memory_id: str
    ) -> None:
        self._ensure_setup()
        db = self._db()
        if user_id is None:
            await db.execute(
                "DELETE FROM long_term_memory WHERE id=? AND user_id IS NULL",
                (memory_id,),
            )
        else:
            await db.execute(
                "DELETE FROM long_term_memory WHERE id=? AND user_id=?",
                (memory_id, user_id),
            )
        await db.commit()

    async def list_long_term_memories(
        self,
        user_id: str | None,
        *,
        scope: str | None = None,
        limit: int = 100,
    ) -> list[LongTermMemory]:
        self._ensure_setup()
        db = self._db()
        params: list[str | int | None] = [user_id, user_id]
        where_clause = self._user_filter_sql()
        if scope is not None:
            where_clause = f"{where_clause} AND scope=?"
            params.append(scope)
        params.append(limit)
        sql = f"SELECT * FROM long_term_memory WHERE {where_clause} ORDER BY updated_at DESC LIMIT ?"
        cursor = await db.execute(sql, tuple(params))
        rows = await cursor.fetchall()
        return [self._row_to_long_term_memory(row) for row in rows]

    async def search_long_term_memory_text(
        self,
        user_id: str | None,
        query: str,
        *,
        scope: str | None = None,
        limit: int = 20,
    ) -> list[LongTermMemory]:
        normalized_query = query.strip().lower()
        if not normalized_query:
            return []
        candidates = await self.list_long_term_memories(
            user_id=user_id, scope=scope, limit=10_000
        )
        ranked: list[tuple[int, int, LongTermMemory]] = []
        for memory in candidates:
            searchable_text = " ".join(
                [(memory.text or ""), " ".join(memory.tags), json_dumps(memory.data)]
            ).lower()
            if normalized_query not in searchable_text:
                continue
            ranked.append(
                (searchable_text.find(normalized_query), -memory.updated_at, memory)
            )
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [memory for _, _, memory in ranked[:limit]]

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
        query_values = np.asarray(query_embedding, dtype=np.float64)
        db = self._db()
        params: list[str | None] = [user_id, user_id]
        where_clause = f"{self._user_filter_sql()} AND embedding_json IS NOT NULL"
        if scope is not None:
            where_clause = f"{where_clause} AND scope=?"
            params.append(scope)
        cursor = await db.execute(
            f"SELECT * FROM long_term_memory WHERE {where_clause}", tuple(params)
        )
        rows = await cursor.fetchall()

        ranked: list[tuple[LongTermMemory, float]] = []
        for row in rows:
            serialized_embedding = row["embedding_json"]
            if serialized_embedding is None:
                continue
            embedding = json_loads(cast(str, serialized_embedding))
            if not isinstance(embedding, list):
                continue
            try:
                similarity = cosine_similarity(
                    query_values, np.asarray(embedding, dtype=np.float64)
                )
            except ValueError:
                continue
            if min_score is not None and similarity < min_score:
                continue
            ranked.append((self._row_to_long_term_memory(row), similarity))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:limit]

    @staticmethod
    def _row_to_event(row: aiosqlite.Row) -> MemoryEvent:
        return MemoryEvent(
            id=cast(str, row["id"]),
            thread_id=cast(str, row["thread_id"]),
            user_id=cast(Optional[str], row["user_id"]),
            type=cast(str, row["type"]),
            timestamp=int(cast(int, row["timestamp"])),
            payload=cast(JsonObject, json_loads(cast(str, row["payload_json"]))),
            tags=cast(list[str], json_loads(cast(str, row["tags_json"]))),
        )

    @staticmethod
    def _row_to_long_term_memory(row: aiosqlite.Row) -> LongTermMemory:
        return LongTermMemory(
            id=cast(str, row["id"]),
            user_id=cast(Optional[str], row["user_id"]),
            scope=cast(str, row["scope"]),
            data=cast(JsonObject, json_loads(cast(str, row["data_json"]))),
            text=cast(Optional[str], row["text"]),
            tags=cast(list[str], json_loads(cast(str, row["tags_json"]))),
            metadata=cast(JsonObject, json_loads(cast(str, row["metadata_json"]))),
            created_at=int(cast(int, row["created_at"])),
            updated_at=int(cast(int, row["updated_at"])),
        )
