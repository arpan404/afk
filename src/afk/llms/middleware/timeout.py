"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Module: llms/middleware/timeout.py

Request timeout middleware for LLM client.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from ..types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)

logger = logging.getLogger("afk.middleware.timeout")


class LLMChatMiddleware(Protocol):
    """Middleware protocol for non-streaming chat requests."""

    async def __call__(self, call_next: Any, req: LLMRequest) -> LLMResponse: ...


class LLMEmbedMiddleware(Protocol):
    """Middleware protocol for embedding requests."""

    async def __call__(self, call_next: Any, req: EmbeddingRequest) -> EmbeddingResponse: ...


class LLMStreamMiddleware(Protocol):
    """Middleware protocol for streaming chat requests."""

    def __call__(self, call_next: Any, req: LLMRequest) -> AsyncIterator[LLMStreamEvent]: ...


@dataclass
class TimeoutConfig:
    """Configuration for timeout middleware."""

    default_timeout_s: float = 30.0
    chat_timeout_s: float | None = None
    embed_timeout_s: float | None = None
    stream_timeout_s: float | None = None


class TimeoutMiddleware(LLMChatMiddleware):
    """
    Middleware that applies timeouts to LLM chat requests.

    Usage:
        from afk.llms.middleware.timeout import TimeoutMiddleware, TimeoutConfig

        config = TimeoutConfig(default_timeout_s=30.0)
        middleware = TimeoutMiddleware(config)

        stack = MiddlewareStack(chat=[middleware])
    """

    def __init__(self, config: TimeoutConfig | None = None) -> None:
        self._config = config or TimeoutConfig()

    async def __call__(
        self,
        call_next: Any,
        req: LLMRequest,
    ) -> LLMResponse:
        """Apply timeout to chat request."""
        timeout_s = self._config.chat_timeout_s or self._config.default_timeout_s
        policy = req.timeout_policy if req.timeout_policy else None
        if policy and policy.request_timeout_s:
            timeout_s = policy.request_timeout_s

        try:
            return await asyncio.wait_for(
                call_next(req),
                timeout=timeout_s,
            )
        except TimeoutError:
            logger.warning(
                "Chat request timed out after %.2fs (model=%s)",
                timeout_s,
                req.model,
            )
            raise


class EmbedTimeoutMiddleware(LLMEmbedMiddleware):
    """Middleware that applies timeouts to embedding requests."""

    def __init__(self, config: TimeoutConfig | None = None) -> None:
        self._config = config or TimeoutConfig()

    async def __call__(
        self,
        call_next: Any,
        req: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Apply timeout to embedding request."""
        timeout_s = self._config.embed_timeout_s or self._config.default_timeout_s

        try:
            return await asyncio.wait_for(
                call_next(req),
                timeout=timeout_s,
            )
        except TimeoutError:
            logger.warning(
                "Embedding request timed out after %.2fs",
                timeout_s,
            )
            raise


class StreamTimeoutMiddleware(LLMStreamMiddleware):
    """
    Middleware that applies timeouts to streaming LLM requests.

    Note: Streaming timeouts apply to each chunk, not the entire stream.
    """

    def __init__(self, config: TimeoutConfig | None = None) -> None:
        self._config = config or TimeoutConfig()

    def __call__(
        self,
        call_next: Any,
        req: LLMRequest,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Apply timeout to streaming request."""
        timeout_s = self._config.stream_timeout_s or self._config.default_timeout_s

        async def timeout_wrapper() -> AsyncIterator[LLMStreamEvent]:
            stream = call_next(req)
            try:
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            stream.__anext__(),
                            timeout=timeout_s,
                        )
                    except StopAsyncIteration:
                        break
                    yield chunk
            except TimeoutError:
                logger.warning(
                    "Stream chunk timed out after %.2fs (model=%s)",
                    timeout_s,
                    req.model,
                )
                raise

        return timeout_wrapper()


def create_timeout_middleware(
    config: TimeoutConfig | None = None,
) -> list[LLMChatMiddleware]:
    """Create a list containing the timeout middleware for chat."""
    return [TimeoutMiddleware(config)]


def create_embed_timeout_middleware(
    config: TimeoutConfig | None = None,
) -> list[LLMEmbedMiddleware]:
    """Create a list containing the timeout middleware for embeddings."""
    return [EmbedTimeoutMiddleware(config)]


def create_stream_timeout_middleware(
    config: TimeoutConfig | None = None,
) -> list[LLMStreamMiddleware]:
    """Create a list containing the timeout middleware for streaming."""
    return [StreamTimeoutMiddleware(config)]
