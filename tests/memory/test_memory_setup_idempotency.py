from __future__ import annotations

import asyncio

from afk.memory.adapters.postgres import PostgresMemoryStore
from afk.memory.adapters.redis import RedisMemoryStore
from afk.memory.adapters.sqlite import SQLiteMemoryStore


def test_sqlite_setup_is_idempotent(tmp_path):
    async def _scenario():
        store = SQLiteMemoryStore(path=str(tmp_path / "idempotent.sqlite3"))
        await store.setup()
        first = store._connection
        await store.setup()
        assert store._connection is first

    asyncio.run(_scenario())


def test_redis_setup_is_idempotent(monkeypatch):
    class _FakeRedis:
        async def ping(self):
            return True

        async def aclose(self):
            return None

    calls = {"count": 0}

    def _fake_from_url(*args, **kwargs):
        _ = args
        _ = kwargs
        calls["count"] += 1
        return _FakeRedis()

    monkeypatch.setattr("afk.memory.adapters.redis.Redis.from_url", _fake_from_url)

    async def _scenario():
        store = RedisMemoryStore(url="redis://example")
        await store.setup()
        first = store._redis_client
        await store.setup()
        assert calls["count"] == 1
        assert store._redis_client is first

    asyncio.run(_scenario())


def test_postgres_setup_is_idempotent(monkeypatch):
    class _FakeConn:
        async def execute(self, *args, **kwargs):
            _ = args
            _ = kwargs
            return None

    class _Acquire:
        async def __aenter__(self):
            return _FakeConn()

        async def __aexit__(self, exc_type, exc, tb):
            _ = exc_type
            _ = exc
            _ = tb
            return False

    class _FakePool:
        def acquire(self):
            return _Acquire()

        async def close(self):
            return None

    calls = {"count": 0}

    async def _fake_create_pool(*args, **kwargs):
        _ = args
        _ = kwargs
        calls["count"] += 1
        return _FakePool()

    class _FakeAsyncPG:
        create_pool = staticmethod(_fake_create_pool)

    monkeypatch.setattr("afk.memory.adapters.postgres.asyncpg", _FakeAsyncPG)

    async def _scenario():
        store = PostgresMemoryStore(
            dsn="postgresql://test",
            vector_dim=3,
        )
        await store.setup()
        first = store._pool
        await store.setup()
        assert calls["count"] == 1
        assert store._pool is first

    asyncio.run(_scenario())
