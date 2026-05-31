"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Module defining middleware protocols and stack for LLM.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from ..types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)

LLMChatNext = Callable[[LLMRequest], Awaitable[LLMResponse]]
LLMEmbedNext = Callable[[EmbeddingRequest], Awaitable[EmbeddingResponse]]
LLMChatStreamNext = Callable[[LLMRequest], AsyncIterator[LLMStreamEvent]]


class LLMChatMiddleware(Protocol):
    """Middleware protocol for non-streaming chat requests."""

    async def __call__(
        self, call_next: LLMChatNext, req: LLMRequest
    ) -> LLMResponse: ...


class LLMEmbedMiddleware(Protocol):
    """Middleware protocol for embedding requests."""

    async def __call__(
        self, call_next: LLMEmbedNext, req: EmbeddingRequest
    ) -> EmbeddingResponse: ...


class LLMStreamMiddleware(Protocol):
    """Middleware protocol for streaming chat requests."""

    def __call__(
        self, call_next: LLMChatStreamNext, req: LLMRequest
    ) -> AsyncIterator[LLMStreamEvent]: ...


@dataclass
class MiddlewareStack:
    """Container for configured chat/embed/stream middleware pipelines."""

    chat: list[LLMChatMiddleware]
    embed: list[LLMEmbedMiddleware]
    stream: list[LLMStreamMiddleware]

    def __init__(
        self,
        chat: list[LLMChatMiddleware] | None = None,
        embed: list[LLMEmbedMiddleware] | None = None,
        stream: list[LLMStreamMiddleware] | None = None,
    ) -> None:
        self.chat = chat or []
        self.embed = embed or []
        self.stream = stream or []
