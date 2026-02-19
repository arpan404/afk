from __future__ import annotations

import asyncio
from typing import AsyncIterator

from pydantic import BaseModel

from afk.agents import Agent
from afk.core.runner import Runner
from afk.llms import LLM
from afk.llms.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
    StreamCompletedEvent,
    StreamTextDeltaEvent,
    ToolCall,
)
from afk.tools import tool


class _AddArgs(BaseModel):
    a: int
    b: int


@tool(args_model=_AddArgs, name="add_numbers")
def add_numbers(args: _AddArgs) -> dict[str, int]:
    return {"result": args.a + args.b}


class _StreamingLLM(LLM):
    @property
    def provider_id(self) -> str:
        return "streaming"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            chat=True,
            streaming=True,
            tool_calling=True,
            structured_output=True,
            interrupt=True,
        )

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = req
        _ = response_model
        return LLMResponse(text="hello")

    async def _chat_stream_core(
        self, req: LLMRequest, *, response_model=None
    ) -> AsyncIterator:
        _ = req
        _ = response_model

        async def _iter():
            yield StreamTextDeltaEvent(delta="hel")
            yield StreamTextDeltaEvent(delta="lo")
            yield StreamCompletedEvent(response=LLMResponse(text="hello"))

        return _iter()

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


class _NonStreamingLLM(LLM):
    @property
    def provider_id(self) -> str:
        return "non-streaming"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            chat=True,
            streaming=False,
            tool_calling=True,
            structured_output=True,
        )

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = req
        _ = response_model
        return LLMResponse(text="fallback stream output.")

    async def _chat_stream_core(
        self, req: LLMRequest, *, response_model=None
    ) -> AsyncIterator:
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


class _ToolThenTextLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def provider_id(self) -> str:
        return "tool-text"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(
            chat=True,
            streaming=False,
            tool_calling=True,
            structured_output=True,
        )

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = response_model
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="tc_add_1",
                        tool_name="add_numbers",
                        arguments={"a": 3, "b": 4},
                    )
                ],
                model=req.model,
            )
        return LLMResponse(text="done", model=req.model)

    async def _chat_stream_core(
        self, req: LLMRequest, *, response_model=None
    ) -> AsyncIterator:
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


def test_run_stream_emits_true_text_deltas_for_streaming_llm():
    async def _scenario() -> list[str]:
        runner = Runner()
        handle = await runner.run_stream(Agent(model=_StreamingLLM(), instructions="x"))
        deltas: list[str] = []
        async for event in handle:
            if event.type == "text_delta" and event.text_delta:
                deltas.append(event.text_delta)
        return deltas

    deltas = asyncio.run(_scenario())
    assert deltas[:2] == ["hel", "lo"]


def test_run_stream_fallback_emits_non_empty_text_deltas():
    async def _scenario() -> list[str]:
        runner = Runner()
        handle = await runner.run_stream(
            Agent(model=_NonStreamingLLM(), instructions="x")
        )
        deltas: list[str] = []
        async for event in handle:
            if event.type == "text_delta" and event.text_delta:
                deltas.append(event.text_delta)
        return deltas

    deltas = asyncio.run(_scenario())
    assert deltas
    assert "".join(deltas).strip() == "fallback stream output."


def test_run_stream_tool_completed_contains_tool_details():
    async def _scenario():
        runner = Runner()
        handle = await runner.run_stream(
            Agent(model=_ToolThenTextLLM(), instructions="x", tools=[add_numbers]),
            user_message="compute",
        )
        seen = []
        async for event in handle:
            if event.type == "tool_completed":
                seen.append(event)
        return seen

    events = asyncio.run(_scenario())
    assert events
    tool_event = events[0]
    assert tool_event.tool_name == "add_numbers"
    assert tool_event.tool_call_id == "tc_add_1"
    assert tool_event.tool_success is True
    assert tool_event.tool_output == {"result": 7}
