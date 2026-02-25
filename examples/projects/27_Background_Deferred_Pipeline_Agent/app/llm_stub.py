"""LLM stub that drives deferred and normal tool execution."""

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


class BackgroundAwareLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def provider_id(self) -> str:
        return "background-stub"

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
                        id="tc_build_27",
                        tool_name="compile_report",
                        arguments={},
                    )
                ],
            )

        for message in req.messages:
            if (
                message.role == "tool"
                and message.name == "compile_report"
                and isinstance(message.content, str)
                and '"status": "ok"' in message.content
            ):
                return LLMResponse(text="Background report completed and summary finalized.")

        return LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id=f"tc_docs_27_{self.calls}",
                    tool_name="draft_executive_summary",
                    arguments={},
                )
            ],
        )

    async def _chat_stream_core(self, req: LLMRequest, *, response_model=None) -> AsyncIterator:
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError
