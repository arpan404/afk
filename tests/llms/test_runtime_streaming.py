from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from afk.llms import (
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
    LLMSettings,
    Message,
    StreamCompletedEvent,
    StreamTextDeltaEvent,
    create_llm_client,
    register_llm_provider,
)
from afk.llms.errors import LLMInvalidResponseError
from afk.llms.providers import ProviderSettingsSchema
from afk.llms.runtime import CachePolicy


class _FakeStreamHandle:
    def __init__(self, events):
        self._events = events
        self.interrupt_calls = 0
        self.cancel_calls = 0

    @property
    def events(self):
        async def _iter():
            for event in self._events:
                yield event

        return _iter()

    async def cancel(self) -> None:
        self.cancel_calls += 1

    async def interrupt(self) -> None:
        self.interrupt_calls += 1

    async def await_result(self):
        return None


@dataclass
class _Transport:
    provider_id: str
    fail_stream: bool = False

    def __post_init__(self):
        self.capabilities = LLMCapabilities(
            chat=True,
            streaming=True,
            tool_calling=False,
            structured_output=False,
            embeddings=False,
            interrupt=True,
            idempotency=True,
        )
        self.stream_handle_calls = 0

    async def chat(self, req, *, response_model=None):
        _ = req
        _ = response_model
        return LLMResponse(text="ok")

    async def chat_stream(self, req, *, response_model=None):
        _ = req
        _ = response_model
        if self.fail_stream:
            raise RuntimeError("stream setup failed")

        async def _iter():
            yield StreamTextDeltaEvent(delta=f"from:{self.provider_id}")
            yield StreamCompletedEvent(response=LLMResponse(text="done"))

        return _iter()

    async def chat_stream_handle(self, req, *, response_model=None):
        _ = req
        _ = response_model
        self.stream_handle_calls += 1
        return _FakeStreamHandle(
            [
                StreamTextDeltaEvent(delta=f"from:{self.provider_id}"),
                StreamCompletedEvent(response=LLMResponse(text="done")),
            ]
        )

    async def embed(self, req):
        _ = req
        raise RuntimeError("unused")

    def start_session(self, *, session_token=None, checkpoint_token=None):
        _ = session_token
        _ = checkpoint_token
        raise RuntimeError("unused")


class _Provider:
    settings_schema = ProviderSettingsSchema()

    def __init__(self, provider_id: str, *, fail_stream: bool = False) -> None:
        self.provider_id = provider_id
        self._transport = _Transport(provider_id=provider_id, fail_stream=fail_stream)

    def create_transport(self, **kwargs):
        _ = kwargs
        return self._transport


def run_async(coro):
    return asyncio.run(coro)


def test_chat_stream_falls_back_to_next_provider():
    register_llm_provider(_Provider("stream_fail", fail_stream=True), overwrite=True)
    register_llm_provider(_Provider("stream_ok", fail_stream=False), overwrite=True)

    llm = create_llm_client(provider="stream_fail", settings=LLMSettings())
    req = LLMRequest(
        model="demo",
        messages=[Message(role="user", content="hello")],
    )
    req = dataclass_replace_route(req, ("stream_fail", "stream_ok"))

    async def scenario():
        stream = await llm.chat_stream(req)
        return [event async for event in stream]

    events = run_async(scenario())
    assert any(
        isinstance(e, StreamTextDeltaEvent) and e.delta == "from:stream_ok"
        for e in events
    )


def test_chat_cache_key_is_scoped_by_session_namespace_and_response_model():
    register_llm_provider(_Provider("cache_scope"), overwrite=True)
    llm = create_llm_client(provider="cache_scope", settings=LLMSettings())
    base = LLMRequest(
        model="demo",
        messages=[Message(role="user", content="hello")],
    )
    session_req = dataclass_replace_session(base, "session_1")
    other_session_req = dataclass_replace_session(base, "session_2")
    tenant_req = dataclass_replace_metadata(base, {"tenant_id": "tenant_a"})

    key_base = llm._cache_key("cache_scope", base)
    key_session = llm._cache_key("cache_scope", session_req)
    key_other_session = llm._cache_key("cache_scope", other_session_req)
    key_tenant = llm._cache_key("cache_scope", tenant_req)
    key_model = llm._cache_key("cache_scope", base, response_model=dict)
    key_namespace = llm._cache_key(
        "cache_scope",
        base,
        cache_policy=CachePolicy(namespace="customer_a"),
    )

    assert len(
        {
            key_base,
            key_session,
            key_other_session,
            key_tenant,
            key_model,
            key_namespace,
        }
    ) == 6


def test_chat_stream_handle_interrupt_uses_single_underlying_handle():
    provider = _Provider("stream_handle_ok")
    register_llm_provider(provider, overwrite=True)

    llm = create_llm_client(provider="stream_handle_ok", settings=LLMSettings())
    req = LLMRequest(model="demo", messages=[Message(role="user", content="hello")])

    async def scenario():
        handle = await llm.chat_stream_handle(req)
        await handle.interrupt()

    run_async(scenario())
    assert provider._transport.stream_handle_calls == 1


def dataclass_replace_route(req: LLMRequest, order: tuple[str, ...]) -> LLMRequest:
    from dataclasses import replace

    from afk.llms.runtime import RoutePolicy

    return replace(req, route_policy=RoutePolicy(provider_order=order))


def dataclass_replace_session(req: LLMRequest, session_token: str) -> LLMRequest:
    from dataclasses import replace

    return replace(req, session_token=session_token)


def dataclass_replace_metadata(req: LLMRequest, metadata: dict[str, object]) -> LLMRequest:
    from dataclasses import replace

    return replace(req, metadata=metadata)


def test_stream_handle_rejects_double_completion_events():
    provider = _Provider("stream_double")

    async def _bad_stream_handle(self, req, *, response_model=None):
        _ = req
        _ = response_model
        self.stream_handle_calls += 1
        return _FakeStreamHandle(
            [
                StreamTextDeltaEvent(delta="x"),
                StreamCompletedEvent(response=LLMResponse(text="a")),
                StreamCompletedEvent(response=LLMResponse(text="b")),
            ]
        )

    provider._transport.chat_stream_handle = _bad_stream_handle.__get__(
        provider._transport, type(provider._transport)
    )
    register_llm_provider(provider, overwrite=True)

    llm = create_llm_client(provider="stream_double", settings=LLMSettings())
    req = LLMRequest(model="demo", messages=[Message(role="user", content="hello")])

    async def scenario():
        handle = await llm.chat_stream_handle(req)
        with pytest.raises(LLMInvalidResponseError):
            async for _ in handle.events:
                pass

    run_async(scenario())
