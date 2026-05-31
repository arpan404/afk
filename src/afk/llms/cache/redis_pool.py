"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Module: llms/cache/redis_pool.py
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import redis.asyncio as redis

logger = logging.getLogger("afk.redis.pool")


@dataclass
class PoolConfig:
    """Configuration for Redis connection pool."""

    max_connections: int = 50
    max_idle_connections: int = 10
    socket_keepalive: bool = True
    socket_keepalive_options: dict[int, int] = field(default_factory=dict)
    socket_connect_timeout: float = 5.0
    socket_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: float = 30.0


class RedisConnectionPool:
    """
    Managed Redis connection pool with lifecycle control.

    Provides centralized connection pooling for high-throughput scenarios,
    reducing connection overhead and preventing connection exhaustion.

    Usage:
        pool = RedisConnectionPool("redis://localhost:6379/0")
        async with pool.client() as redis_client:
            await redis_client.get("key")
        await pool.disconnect()
    """

    def __init__(
        self,
        url: str,
        *,
        config: PoolConfig | None = None,
    ) -> None:
        self._url = url
        self._config = config or PoolConfig()
        self._pool: redis.ConnectionPool | None = None
        self._client: redis.Redis | None = None
        self._lock = asyncio.Lock()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Initialize the connection pool."""
        async with self._lock:
            if self._connected:
                return

            self._pool = redis.ConnectionPool.from_url(
                self._url,
                max_connections=self._config.max_connections,
                max_idle_connections=self._config.max_idle_connections,
                socket_keepalive=self._config.socket_keepalive,
                socket_keepalive_options=self._config.socket_keepalive_options,
                socket_connect_timeout=self._config.socket_connect_timeout,
                socket_timeout=self._config.socket_timeout,
                retry_on_timeout=self._config.retry_on_timeout,
                health_check_interval=int(self._config.health_check_interval),
            )
            self._client = redis.Redis(connection_pool=self._pool)
            self._connected = True
            logger.info(
                "Redis pool connected (url=%s, max_connections=%d)",
                self._url,
                self._config.max_connections,
            )

    async def disconnect(self) -> None:
        """Close the connection pool gracefully."""
        async with self._lock:
            if not self._connected:
                return

            if self._client:
                await self._client.aclose()
            if self._pool:
                await self._pool.aclose()

            self._client = None
            self._pool = None
            self._connected = False
            logger.info("Redis pool disconnected (url=%s)", self._url)

    @asynccontextmanager
    async def client(self) -> AsyncIterator[redis.Redis]:
        """
        Acquire a Redis client from the pool.

        Yields:
            An async Redis client instance.

        Example:
            pool = RedisConnectionPool("redis://localhost:6379/0")
            async with pool.client() as redis:
                await redis.set("key", "value")
                value = await redis.get("key")
        """
        if not self._connected:
            await self.connect()

        client = self._client
        if client is None:
            raise RuntimeError("Redis client not initialized")

        yield client

    async def get_client(self) -> redis.Redis:
        """
        Get a Redis client (non-context manager usage).

        Returns:
            The shared Redis client instance.
        """
        if not self._connected:
            await self.connect()
        client = self._client
        if client is None:
            raise RuntimeError("Redis client not initialized")
        return client

    async def health_check(self) -> bool:
        """
        Perform a health check on the connection.

        Returns:
            True if the connection is healthy, False otherwise.
        """
        try:
            client = await self.get_client()
            await client.ping()  # type: ignore[await-only]
            return True
        except Exception as e:
            logger.warning("Redis health check failed: %s", e)
            return False

    @property
    def pool_stats(self) -> dict[str, Any] | None:
        """Return current pool statistics if available."""
        if self._pool is None:
            return None
        return {
            "max_connections": self._config.max_connections,
            "max_idle_connections": self._config.max_idle_connections,
            "connected": self._connected,
        }


_POOLS: dict[str, RedisConnectionPool] = {}
_POOLS_LOCK = asyncio.Lock()


async def get_redis_pool(
    url: str,
    *,
    config: PoolConfig | None = None,
) -> RedisConnectionPool:
    """
    Get or create a singleton Redis connection pool by URL.

    Args:
        url: Redis connection URL.
        config: Optional pool configuration.

    Returns:
        The Redis connection pool instance.
    """
    async with _POOLS_LOCK:
        if url not in _POOLS:
            _POOLS[url] = RedisConnectionPool(url, config=config)
            await _POOLS[url].connect()
        return _POOLS[url]


async def close_all_pools() -> None:
    """Close all registered Redis connection pools."""
    async with _POOLS_LOCK:
        for _, pool in _POOLS.items():
            await pool.disconnect()
        _POOLS.clear()
        logger.info("All Redis pools closed")
