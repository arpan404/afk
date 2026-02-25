"""LLM stub that invokes an MCP-backed tool."""

from collections.abc import AsyncIterator

from afk.llms import LLM
from afk.llms.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
    ToolCall,
)


class MCPToolLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def provider_id(self) -> str:
        return "mcp-stub"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(chat=True, streaming=False, tool_calling=True)

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = response_model
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="tc_mcp_add",
                        tool_name="calc__add",
                        arguments={"a": 5, "b": 7},
                    )
                ],
                model=req.model,
            )
        return LLMResponse(text="mcp-tool-finished", model=req.model)

    async def _chat_stream_core(self, req: LLMRequest, *, response_model=None) -> AsyncIterator:
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError
