"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Centralized, type-safe environment variable resolution.

``EnvVarField`` descriptors declare env vars; ``Settings.from_env()`` resolves
them and returns an immutable typed object.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, ClassVar, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _csv_list(raw: str) -> list[str]:
    """Parse a comma-separated string into a list of stripped, non-empty tokens."""
    return [s.strip() for s in raw.split(",") if s.strip()]


def _bool(raw: str) -> bool:
    """Parse a boolean from a raw string (raises on invalid input)."""
    if raw.strip().lower() in ("1", "true", "yes", "y", "on"):
        return True
    if raw.strip().lower() in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean value: {raw!r}")


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean from an environment variable (returns default if unset)."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return _bool(raw)


def _float(raw: str) -> float:
    return float(raw)


def _int(raw: str) -> int:
    return int(raw)


# ---------------------------------------------------------------------------
# EnvVarField descriptor
# ---------------------------------------------------------------------------


class EnvVarField:
    """
    A class-level descriptor that binds a settings field to an environment variable.

    Usage::

        class MySettings:
            host: str  = EnvVarField("MY_HOST", default="localhost")
            port: int  = EnvVarField("MY_PORT", default=8000, parser=_int)
            debug: bool = EnvVarField("DEBUG", default=False, parser=_bool)
    """

    __slots__ = ("env_name", "default", "parser")

    def __init__(
        self,
        env_name: str,
        default: Any,
        parser: Callable[[str], Any] = lambda s: s,
    ) -> None:
        self.env_name = env_name
        self.default = default
        self.parser = parser

    def get(self) -> Any:
        raw = os.getenv(self.env_name)
        if raw is None:
            return self.default
        if raw == "":
            return self.default
        return self.parser(raw)


# ---------------------------------------------------------------------------
# Settings base
# ---------------------------------------------------------------------------


class Settings:
    """
    Base class for settings that load from environment variables.

    Subclasses declare ``EnvVarField`` class variables.  ``from_env()`` resolves
    each from the environment and returns an instance of the subclass.
    """

    __slots__: ClassVar[list[str]] = []

    @classmethod
    def from_env(cls: type[T]) -> T:
        kwargs = {}
        for name in dir(cls):
            if name.startswith("_"):
                continue
            val = getattr(cls, name, None)
            if not isinstance(val, EnvVarField):
                continue
            kwargs[name] = val.get()
        instance = object.__new__(cls)
        for k, v in kwargs.items():
            object.__setattr__(instance, k, v)
        return instance  # type: ignore[return-value]

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")


# ---------------------------------------------------------------------------
# Concrete env bindings
# ---------------------------------------------------------------------------


class MCPServerEnv(Settings):
    """Environment variable bindings for the MCP server."""

    AFK_CORS_ORIGINS: list[str] = EnvVarField(
        "AFK_CORS_ORIGINS", default=[], parser=_csv_list
    )
    AFK_MCP_NAME: str = EnvVarField("AFK_MCP_NAME", default="afk-mcp-server")
    AFK_MCP_VERSION: str = EnvVarField("AFK_MCP_VERSION", default="1.0.0")
    AFK_MCP_HOST: str = EnvVarField("AFK_MCP_HOST", default="127.0.0.1")
    AFK_MCP_PORT: int = EnvVarField("AFK_MCP_PORT", default=8000, parser=_int)
    AFK_MCP_INSTRUCTIONS: str | None = EnvVarField(
        "AFK_MCP_INSTRUCTIONS", default=None
    )
    AFK_MCP_PATH: str = EnvVarField("AFK_MCP_PATH", default="/mcp")
    AFK_MCP_SSE_PATH: str = EnvVarField("AFK_MCP_SSE_PATH", default="/mcp/sse")
    AFK_MCP_HEALTH_PATH: str = EnvVarField("AFK_MCP_HEALTH_PATH", default="/health")
    AFK_MCP_ENABLE_SSE: bool = EnvVarField(
        "AFK_MCP_ENABLE_SSE", default=True, parser=_bool
    )
    AFK_MCP_ENABLE_HEALTH: bool = EnvVarField(
        "AFK_MCP_ENABLE_HEALTH", default=True, parser=_bool
    )
    AFK_MCP_ALLOW_BATCH: bool = EnvVarField(
        "AFK_MCP_ALLOW_BATCH", default=True, parser=_bool
    )


class RunnerEnv(Settings):
    """Environment variable bindings for runner behavior."""

    AFK_ALLOWED_COMMANDS: list[str] = EnvVarField(
        "AFK_ALLOWED_COMMANDS", default=[], parser=_csv_list
    )


class MemoryEnv(Settings):
    """Environment variable bindings for memory stores and queues."""

    AFK_MEMORY_BACKEND: str = EnvVarField("AFK_MEMORY_BACKEND", default="memory")
    AFK_SQLITE_PATH: str = EnvVarField("AFK_SQLITE_PATH", default="afk_memory.sqlite3")
    AFK_REDIS_URL: str | None = EnvVarField("AFK_REDIS_URL", default=None)
    AFK_REDIS_HOST: str = EnvVarField("AFK_REDIS_HOST", default="localhost")
    AFK_REDIS_PORT: int = EnvVarField("AFK_REDIS_PORT", default=6379, parser=_int)
    AFK_REDIS_DB: int = EnvVarField("AFK_REDIS_DB", default=0, parser=_int)
    AFK_REDIS_PASSWORD: str = EnvVarField("AFK_REDIS_PASSWORD", default="")
    AFK_REDIS_EVENTS_MAX: int = EnvVarField(
        "AFK_REDIS_EVENTS_MAX", default=2000, parser=_int
    )
    AFK_PG_DSN: str | None = EnvVarField("AFK_PG_DSN", default=None)
    AFK_PG_HOST: str = EnvVarField("AFK_PG_HOST", default="localhost")
    AFK_PG_PORT: int = EnvVarField("AFK_PG_PORT", default=5432, parser=_int)
    AFK_PG_USER: str = EnvVarField("AFK_PG_USER", default="postgres")
    AFK_PG_PASSWORD: str = EnvVarField("AFK_PG_PASSWORD", default="")
    AFK_PG_DB: str = EnvVarField("AFK_PG_DB", default="afk")
    AFK_PG_SSL: bool = EnvVarField("AFK_PG_SSL", default=False, parser=_bool)
    AFK_PG_POOL_MIN: int = EnvVarField("AFK_PG_POOL_MIN", default=1, parser=_int)
    AFK_PG_POOL_MAX: int = EnvVarField("AFK_PG_POOL_MAX", default=10, parser=_int)
    AFK_VECTOR_DIM: int | None = EnvVarField("AFK_VECTOR_DIM", default=None, parser=_int)
    AFK_QUEUE_BACKEND: str = EnvVarField("AFK_QUEUE_BACKEND", default="inmemory")
    AFK_QUEUE_RETRY_BACKOFF_BASE_S: float = EnvVarField(
        "AFK_QUEUE_RETRY_BACKOFF_BASE_S", default=0.5, parser=_float
    )
    AFK_QUEUE_RETRY_BACKOFF_MAX_S: float = EnvVarField(
        "AFK_QUEUE_RETRY_BACKOFF_MAX_S", default=30.0, parser=_float
    )
    AFK_QUEUE_RETRY_BACKOFF_JITTER_S: float = EnvVarField(
        "AFK_QUEUE_RETRY_BACKOFF_JITTER_S", default=0.2, parser=_float
    )
    AFK_QUEUE_REDIS_PREFIX: str = EnvVarField(
        "AFK_QUEUE_REDIS_PREFIX", default="afk:queue"
    )
    AFK_QUEUE_REDIS_URL: str | None = EnvVarField("AFK_QUEUE_REDIS_URL", default=None)
    AFK_QUEUE_REDIS_HOST: str = EnvVarField("AFK_QUEUE_REDIS_HOST", default="localhost")
    AFK_QUEUE_REDIS_PORT: int = EnvVarField(
        "AFK_QUEUE_REDIS_PORT", default=6379, parser=_int
    )
    AFK_QUEUE_REDIS_DB: int = EnvVarField("AFK_QUEUE_REDIS_DB", default=0, parser=_int)
    AFK_QUEUE_REDIS_PASSWORD: str = EnvVarField(
        "AFK_QUEUE_REDIS_PASSWORD", default=""
    )


class LLMEnv(Settings):
    """Environment variable bindings for LLM providers."""

    AFK_LLM_PROVIDER: str = EnvVarField("AFK_LLM_PROVIDER", default="litellm")
    AFK_LLM_PROVIDER_ORDER: list[str] = EnvVarField(
        "AFK_LLM_PROVIDER_ORDER", default=[], parser=_csv_list
    )
    AFK_LLM_MODEL: str = EnvVarField("AFK_LLM_MODEL", default="gpt-4.1-mini")
    AFK_EMBED_MODEL: str | None = EnvVarField("AFK_EMBED_MODEL", default=None)
    AFK_LLM_API_BASE_URL: str | None = EnvVarField(
        "AFK_LLM_API_BASE_URL", default=None
    )
    AFK_LLM_API_KEY: str | None = EnvVarField("AFK_LLM_API_KEY", default=None)
    AFK_LLM_TIMEOUT_S: float = EnvVarField(
        "AFK_LLM_TIMEOUT_S", default=30.0, parser=_float
    )
    AFK_LLM_MAX_RETRIES: int = EnvVarField(
        "AFK_LLM_MAX_RETRIES", default=3, parser=_int
    )
    AFK_LLM_BACKOFF_BASE_S: float = EnvVarField(
        "AFK_LLM_BACKOFF_BASE_S", default=0.5, parser=_float
    )
    AFK_LLM_BACKOFF_JITTER_S: float = EnvVarField(
        "AFK_LLM_BACKOFF_JITTER_S", default=0.15, parser=_float
    )
    AFK_LLM_JSON_MAX_RETRIES: int = EnvVarField(
        "AFK_LLM_JSON_MAX_RETRIES", default=2, parser=_int
    )
    AFK_LLM_MAX_INPUT_CHARS: int = EnvVarField(
        "AFK_LLM_MAX_INPUT_CHARS", default=200000, parser=_int
    )
    AFK_LLM_STREAM_IDLE_TIMEOUT_S: float = EnvVarField(
        "AFK_LLM_STREAM_IDLE_TIMEOUT_S", default=45.0, parser=_float
    )
