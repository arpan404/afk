"""Primary failing model and backup reasoning-aware model stubs."""

from collections.abc import AsyncIterator

from afk.llms import LLM
from afk.llms.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
)


class FailingLLM(LLM):
    @property
    def provider_id(self) -> str:
        return "primary-down"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(chat=True, streaming=False, tool_calling=False)

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = req
        _ = response_model
        raise RuntimeError("primary model unavailable")

    async def _chat_stream_core(self, req: LLMRequest, *, response_model=None) -> AsyncIterator:
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


class ReasoningAwareLLM(LLM):
    @property
    def provider_id(self) -> str:
        return "backup-reasoner"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(chat=True, streaming=False, tool_calling=False)

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = response_model
        return LLMResponse(
            text=(
                "fallback engaged; "
                f"thinking={req.thinking}; "
                f"effort={req.thinking_effort}; "
                f"max_tokens={req.max_thinking_tokens}"
            )
        )

    async def _chat_stream_core(self, req: LLMRequest, *, response_model=None) -> AsyncIterator:
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError
