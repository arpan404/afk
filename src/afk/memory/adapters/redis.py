"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

This module provides a Redis memory backend optimized for low-latency state and event operations.
"""

from __future__ import annotations


from typing import Optional, Sequence, cast

import numpy as np
from redis.asyncio import Redis
from redis.exceptions import WatchError

from afk.memory.types import (
    JsonObject,
    JsonValue,
    LongTermMemory,
    MemoryEvent,
)
from afk.memory.utils import json_dumps, json_loads
from afk.memory.vector import cosine_similarity
from afk.memory.store import MemoryCapabilities, MemoryStore


class RedisMemoryStore(MemoryStore):
    """Redis-backed memory store using hashes for state/memories and lists for events."""

    capabilities = MemoryCapabilities(
        text_search=True, vector_search=True, atomic_upsert=True, ttl=True
    )

    def __init__(self, *, url: str, events_max_per_thread: int = 2_000) -> None:
        super().__init__()
        if events_max_per_thread < 1:
            raise ValueError("events_max_per_thread must be >= 1")
        self.url = url
        self.events_max_per_thread = events_max_per_thread
        self._redis_client: Redis | None = None

    async def setup(self) -> None:
        if self._is_setup and self._redis_client is not None:
            return
        self._redis_client = Redis.from_url(self.url, decode_responses=True)
        await self._redis_client.ping()
        await super().setup()

    async def close(self) -> None:
        if self._redis_client is not None:
            await self._redis_client.aclose()
            self._redis_client = None
        await super().close()

    def _redis(self) -> Redis:
        if self._redis_client is None:
            raise RuntimeError(
                "RedisMemoryStore is not initialized. Call setup() first."
            )
        return self._redis_client

    @staticmethod
    def _events_key(thread_id: str) -> str:
        return f"afk:events:{thread_id}"

    @staticmethod
    def _state_key(thread_id: str) -> str:
        return f"afk:state:{thread_id}"

    @staticmethod
    def _memory_hash_key(user_id: str | None) -> str:
        # Keep global/system memory in a separate namespace from user memories.
        return f"afk:ltm:{user_id or '_none_'}"

    async def append_event(self, event: MemoryEvent) -> None:
        self._ensure_setup()
        redis_client = self._redis()
        serialized_event = json_dumps(
            {
                "id": event.id,
                "thread_id": event.thread_id,
                "user_id": event.user_id,
                "type": event.type,
                "timestamp": event.timestamp,
                "payload": event.payload,
                "tags": event.tags,
            },
        )
        pipeline = redis_client.pipeline()
        events_key = self._events_key(event.thread_id)
        pipeline.rpush(events_key, serialized_event)
        pipeline.ltrim(events_key, -self.events_max_per_thread, -1)
        await pipeline.execute()

    async def get_recent_events(
        self, thread_id: str, limit: int = 50
    ) -> list[MemoryEvent]:
        self._ensure_setup()
        if limit <= 0:
            return []
        redis_client = self._redis()
        serialized_events = await redis_client.lrange(
            self._events_key(thread_id), -limit, -1
        )
        return [self._deserialize_event(payload) for payload in serialized_events]

    async def get_events_since(
        self, thread_id: str, since_ms: int, limit: int = 500
    ) -> list[MemoryEvent]:
        self._ensure_setup()
        events = await self.get_recent_events(
            thread_id, limit=self.events_max_per_thread
        )
        matches = [event for event in events if event.timestamp >= since_ms]
        return matches[:limit]

    async def put_state(self, thread_id: str, key: str, value: JsonValue) -> None:
        self._ensure_setup()
        redis_client = self._redis()
        await redis_client.hset(self._state_key(thread_id), key, json_dumps(value))

    async def delete_state(self, thread_id: str, key: str) -> None:
        self._ensure_setup()
        redis_client = self._redis()
        await redis_client.hdel(self._state_key(thread_id), key)

    async def get_state(self, thread_id: str, key: str) -> JsonValue | None:
        self._ensure_setup()
        redis_client = self._redis()
        value = await redis_client.hget(self._state_key(thread_id), key)
        if value is None:
            return None
        return json_loads(value)

    async def list_state(
        self, thread_id: str, prefix: str | None = None
    ) -> dict[str, JsonValue]:
        self._ensure_setup()
        redis_client = self._redis()
        raw_values = await redis_client.hgetall(self._state_key(thread_id))
        filtered_state: dict[str, JsonValue] = {}
        for state_key, serialized_state_value in raw_values.items():
            if prefix is not None and not state_key.startswith(prefix):
                continue
            filtered_state[state_key] = json_loads(serialized_state_value)
        return dict(sorted(filtered_state.items(), key=lambda item: item[0]))

    async def replace_thread_events(
        self,
        thread_id: str,
        events: list[MemoryEvent],
    ) -> None:
        self._ensure_setup()
        redis_client = self._redis()
        events_key = self._events_key(thread_id)
        pipeline = redis_client.pipeline()
        pipeline.delete(events_key)
        for event in events:
            serialized_event = json_dumps(
                {
                    "id": event.id,
                    "thread_id": event.thread_id,
                    "user_id": event.user_id,
                    "type": event.type,
                    "timestamp": event.timestamp,
                    "payload": event.payload,
                    "tags": event.tags,
                },
            )
            pipeline.rpush(events_key, serialized_event)
        pipeline.ltrim(events_key, -self.events_max_per_thread, -1)
        await pipeline.execute()

    async def upsert_long_term_memory(
        self,
        memory: LongTermMemory,
        *,
        embedding: Optional[Sequence[float]] = None,
    ) -> None:
        """Atomically insert or update a long-term memory in Redis.

        Guarantees:
        - If `embedding` is None we preserve any existing stored embedding
          (no race between concurrent upserts).
        - If `embedding` is provided, it replaces the stored embedding.

        Implementation uses WATCH / MULTI with retries to provide atomic
        read-modify-write semantics.
        """
        self._ensure_setup()
        redis_client = self._redis()
        hash_key = self._memory_hash_key(memory.user_id)

        max_retries = 5
        for attempt in range(max_retries):
            async with redis_client.pipeline() as pipeline:
                try:
                    # WATCH + read + MULTI must use the same pipeline/connection.
                    await pipeline.watch(hash_key)

                    existing = await pipeline.hget(hash_key, memory.id)
                    existing_embedding: list[float] | None = None
                    if existing is not None:
                        existing_payload = json_loads(existing)
                        if isinstance(existing_payload, dict):
                            candidate_embedding = existing_payload.get("embedding")
                            if isinstance(candidate_embedding, list):
                                existing_embedding = [
                                    float(v) for v in candidate_embedding
                                ]

                    chosen_embedding = (
                        existing_embedding
                        if embedding is None
                        else [float(v) for v in embedding]
                    )

                    payload: JsonObject = {
                        "id": memory.id,
                        "user_id": memory.user_id,
                        "scope": memory.scope,
                        "data": memory.data,
                        "text": memory.text,
                        "tags": memory.tags,
                        "metadata": memory.metadata,
                        "created_at": memory.created_at,
                        "updated_at": memory.updated_at,
                        "embedding": chosen_embedding,
                    }

                    pipeline.multi()
                    pipeline.hset(hash_key, memory.id, json_dumps(payload))
                    await pipeline.execute()
                    return
                except WatchError:
                    if attempt == max_retries - 1:
                        raise RuntimeError(
                            "Failed to upsert long-term memory after concurrent update retries."
                        ) from None
                    continue

    async def delete_long_term_memory(
        self, user_id: str | None, memory_id: str
    ) -> None:
        self._ensure_setup()
        redis_client = self._redis()
        await redis_client.hdel(self._memory_hash_key(user_id), memory_id)

    async def list_long_term_memories(
        self,
        user_id: str | None,
        *,
        scope: str | None = None,
        limit: int = 100,
    ) -> list[LongTermMemory]:
        self._ensure_setup()
        redis_client = self._redis()
        values = await redis_client.hgetall(self._memory_hash_key(user_id))
        memories: list[LongTermMemory] = []
        for serialized_memory in values.values():
            memory_payload = json_loads(serialized_memory)
            if not isinstance(memory_payload, dict):
                continue
            if scope is not None and memory_payload.get("scope") != scope:
                continue
            memories.append(self._payload_to_memory(memory_payload))
        memories.sort(key=lambda memory: memory.updated_at, reverse=True)
        return memories[:limit]

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
        memories = await self.list_long_term_memories(
            user_id=user_id, scope=scope, limit=10_000
        )
        ranked: list[tuple[int, int, LongTermMemory]] = []
        for memory in memories:
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
        redis_client = self._redis()
        query_values = np.asarray(query_embedding, dtype=np.float64)

        values = await redis_client.hgetall(self._memory_hash_key(user_id))
        ranked: list[tuple[LongTermMemory, float]] = []
        for serialized_memory in values.values():
            memory_payload = json_loads(serialized_memory)
            if not isinstance(memory_payload, dict):
                continue
            if scope is not None and memory_payload.get("scope") != scope:
                continue
            embedding = memory_payload.get("embedding")
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
            ranked.append((self._payload_to_memory(memory_payload), similarity))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:limit]

    @staticmethod
    def _deserialize_event(serialized_event: str) -> MemoryEvent:
        payload = json_loads(serialized_event)
        if not isinstance(payload, dict):
            raise ValueError("Invalid serialized event payload.")
        return MemoryEvent(
            id=cast(str, payload["id"]),
            thread_id=cast(str, payload["thread_id"]),
            user_id=cast(Optional[str], payload.get("user_id")),
            type=cast(str, payload["type"]),
            timestamp=int(cast(int, payload["timestamp"])),
            payload=cast(JsonObject, payload["payload"]),
            tags=cast(list[str], payload.get("tags", [])),
        )

    @staticmethod
    def _payload_to_memory(payload: dict[str, object]) -> LongTermMemory:
        return LongTermMemory(
            id=cast(str, payload["id"]),
            user_id=cast(Optional[str], payload.get("user_id")),
            scope=cast(str, payload["scope"]),
            data=cast(JsonObject, payload["data"]),
            text=cast(Optional[str], payload.get("text")),
            tags=cast(list[str], payload.get("tags", [])),
            metadata=cast(JsonObject, payload.get("metadata", {})),
            created_at=int(cast(int, payload.get("created_at", 0))),
            updated_at=int(cast(int, payload.get("updated_at", 0))),
        )
