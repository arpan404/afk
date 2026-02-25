"""Deterministic LLM used for local eval suite demonstrations."""

from collections.abc import AsyncIterator

from afk.llms import LLM
from afk.llms.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
)


class EvalStaticLLM(LLM):
    @property
    def provider_id(self) -> str:
        return "eval-static"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(chat=True, streaming=False, tool_calling=False)

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = response_model
        content = " ".join(str(m.content) for m in req.messages if m.role == "user")
        return LLMResponse(text=f"ok: {content}")

    async def _chat_stream_core(self, req: LLMRequest, *, response_model=None) -> AsyncIterator:
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError
