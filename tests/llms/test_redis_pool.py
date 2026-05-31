from __future__ import annotations

import asyncio

from afk.llms.cache import redis_pool
from afk.llms.cache.redis_pool import (
    PoolConfig,
    RedisConnectionPool,
    close_all_pools,
    get_redis_pool,
)


def run_async(coro):
    return asyncio.run(coro)


class _FakeConnectionPool:
    created: list[dict[str, object]] = []
    closed = 0

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @classmethod
    def from_url(cls, url: str, **kwargs):
        row = {"url": url, **kwargs}
        cls.created.append(row)
        return cls(**row)

    async def aclose(self) -> None:
        type(self).closed += 1


class _FakeRedis:
    instances: list[_FakeRedis] = []

    def __init__(self, *, connection_pool) -> None:
        self.connection_pool = connection_pool
        self.closed = False
        self.ping_ok = True
        type(self).instances.append(self)

    async def aclose(self) -> None:
        self.closed = True

    async def ping(self) -> bool:
        if not self.ping_ok:
            raise RuntimeError("redis down")
        return True


def setup_function():
    _FakeConnectionPool.created.clear()
    _FakeConnectionPool.closed = 0
    _FakeRedis.instances.clear()
    run_async(close_all_pools())


def test_pool_connect_disconnect_and_stats(monkeypatch):
    monkeypatch.setattr(redis_pool.redis.ConnectionPool, "from_url", _FakeConnectionPool.from_url)
    monkeypatch.setattr(redis_pool.redis, "Redis", _FakeRedis)

    config = PoolConfig(max_connections=7, max_idle_connections=3, health_check_interval=9.5)
    pool = RedisConnectionPool("redis://localhost:6379/0", config=config)

    assert pool.is_connected is False
    assert pool.pool_stats is None

    run_async(pool.connect())

    assert pool.is_connected is True
    assert _FakeConnectionPool.created[0]["url"] == "redis://localhost:6379/0"
    assert _FakeConnectionPool.created[0]["max_connections"] == 7
    assert _FakeConnectionPool.created[0]["max_idle_connections"] == 3
    assert _FakeConnectionPool.created[0]["health_check_interval"] == 9
    assert pool.pool_stats == {
        "max_connections": 7,
        "max_idle_connections": 3,
        "connected": True,
    }

    run_async(pool.disconnect())

    assert pool.is_connected is False
    assert _FakeRedis.instances[0].closed is True
    assert _FakeConnectionPool.closed == 1


def test_client_context_connects_lazily_and_yields_shared_client(monkeypatch):
    monkeypatch.setattr(redis_pool.redis.ConnectionPool, "from_url", _FakeConnectionPool.from_url)
    monkeypatch.setattr(redis_pool.redis, "Redis", _FakeRedis)
    pool = RedisConnectionPool("redis://localhost:6379/1")

    async def scenario():
        async with pool.client() as client:
            from_context = client
        direct = await pool.get_client()
        return from_context, direct

    from_context, direct = run_async(scenario())

    assert from_context is direct
    assert pool.is_connected is True
    assert len(_FakeRedis.instances) == 1
    run_async(pool.disconnect())


def test_health_check_returns_false_when_ping_fails(monkeypatch):
    monkeypatch.setattr(redis_pool.redis.ConnectionPool, "from_url", _FakeConnectionPool.from_url)
    monkeypatch.setattr(redis_pool.redis, "Redis", _FakeRedis)
    pool = RedisConnectionPool("redis://localhost:6379/2")

    async def scenario():
        client = await pool.get_client()
        client.ping_ok = False
        return await pool.health_check()

    assert run_async(scenario()) is False
    run_async(pool.disconnect())


def test_get_redis_pool_reuses_by_url_and_close_all(monkeypatch):
    monkeypatch.setattr(redis_pool.redis.ConnectionPool, "from_url", _FakeConnectionPool.from_url)
    monkeypatch.setattr(redis_pool.redis, "Redis", _FakeRedis)

    async def scenario():
        first = await get_redis_pool("redis://localhost:6379/3")
        second = await get_redis_pool("redis://localhost:6379/3")
        other = await get_redis_pool("redis://localhost:6379/4")
        await close_all_pools()
        return first, second, other

    first, second, other = run_async(scenario())

    assert first is second
    assert first is not other
    assert first.is_connected is False
    assert other.is_connected is False
    assert redis_pool._POOLS == {}
