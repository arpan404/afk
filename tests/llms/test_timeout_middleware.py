from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from afk.llms.middleware.timeout import (
    EmbedTimeoutMiddleware,
    StreamTimeoutMiddleware,
    TimeoutConfig,
    TimeoutMiddleware,
    create_embed_timeout_middleware,
    create_stream_timeout_middleware,
    create_timeout_middleware,
)
from afk.llms.runtime import TimeoutPolicy
from afk.llms.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMRequest,
    LLMResponse,
    Message,
    StreamTextDeltaEvent,
)


def run_async(coro):
    return asyncio.run(coro)


def _request(**kwargs) -> LLMRequest:
    return LLMRequest(
        model=kwargs.pop("model", "demo"),
        messages=[Message(role="user", content="hello")],
        **kwargs,
    )


def test_chat_timeout_middleware_returns_response():
    async def call_next(req: LLMRequest) -> LLMResponse:
        return LLMResponse(text=f"ok:{req.model}")

    middleware = TimeoutMiddleware(TimeoutConfig(default_timeout_s=1.0))

    result = run_async(middleware(call_next, _request(model="m1")))

    assert result.text == "ok:m1"


def test_chat_timeout_middleware_uses_request_policy_override():
    seen = {}

    async def call_next(req: LLMRequest) -> LLMResponse:
        seen["model"] = req.model
        await asyncio.sleep(0.02)
        return LLMResponse(text="late")

    middleware = TimeoutMiddleware(TimeoutConfig(default_timeout_s=1.0))
    req = _request(timeout_policy=TimeoutPolicy(request_timeout_s=0.001))

    with pytest.raises(TimeoutError):
        run_async(middleware(call_next, req))
    assert seen["model"] == "demo"


def test_embed_timeout_middleware_returns_response():
    async def call_next(req: EmbeddingRequest) -> EmbeddingResponse:
        return EmbeddingResponse(embeddings=[[1.0, 2.0]], model=req.model)

    middleware = EmbedTimeoutMiddleware(TimeoutConfig(embed_timeout_s=1.0))

    result = run_async(middleware(call_next, EmbeddingRequest(model="embed")))

    assert result.embeddings == [[1.0, 2.0]]
    assert result.model == "embed"


def test_embed_timeout_middleware_raises_on_slow_call():
    async def call_next(req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        await asyncio.sleep(0.02)
        return EmbeddingResponse(embeddings=[])

    middleware = EmbedTimeoutMiddleware(TimeoutConfig(embed_timeout_s=0.001))

    with pytest.raises(TimeoutError):
        run_async(middleware(call_next, EmbeddingRequest(model="embed")))


def test_stream_timeout_middleware_yields_chunks():
    async def call_next(req: LLMRequest):
        yield StreamTextDeltaEvent(delta=f"first:{req.model}")
        yield StreamTextDeltaEvent(delta="second")

    middleware = StreamTimeoutMiddleware(TimeoutConfig(stream_timeout_s=1.0))

    async def scenario():
        return [event async for event in middleware(call_next, _request(model="stream"))]

    events = run_async(scenario())

    assert [event.delta for event in events] == ["first:stream", "second"]


def test_stream_timeout_middleware_applies_timeout_per_chunk():
    async def call_next(req: LLMRequest):
        _ = req
        await asyncio.sleep(0.02)
        yield StreamTextDeltaEvent(delta="late")

    middleware = StreamTimeoutMiddleware(TimeoutConfig(stream_timeout_s=0.001))

    async def scenario():
        return [event async for event in middleware(call_next, _request())]

    with pytest.raises(TimeoutError):
        run_async(scenario())


def test_timeout_middleware_factories_return_expected_types():
    config = TimeoutConfig(default_timeout_s=2.0)

    assert isinstance(create_timeout_middleware(config)[0], TimeoutMiddleware)
    assert isinstance(create_embed_timeout_middleware(config)[0], EmbedTimeoutMiddleware)
    assert isinstance(create_stream_timeout_middleware(config)[0], StreamTimeoutMiddleware)


def test_timeout_config_can_be_reused_with_modified_request():
    req = _request()
    updated = replace(req, timeout_policy=TimeoutPolicy(request_timeout_s=0.5))

    assert updated.timeout_policy is not None
    assert updated.timeout_policy.request_timeout_s == 0.5
