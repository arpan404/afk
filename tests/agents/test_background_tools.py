from __future__ import annotations

import asyncio

from pydantic import BaseModel

from afk.agents import Agent, FailSafeConfig
from afk.agents.errors import AgentExecutionError
from afk.core import Runner, RunnerConfig
from afk.llms import LLM
from afk.llms.types import (
    EmbeddingRequest,
    EmbeddingResponse,
    LLMCapabilities,
    LLMRequest,
    LLMResponse,
    ToolCall,
)
from afk.memory import InMemoryMemoryStore
from afk.tools import ToolDeferredHandle, ToolResult, tool


class _NoArgs(BaseModel):
    pass


@tool(args_model=_NoArgs, name="write_docs")
def write_docs(args: _NoArgs) -> dict[str, str]:
    _ = args
    return {"status": "docs_written"}


@tool(args_model=_NoArgs, name="build_project")
async def build_project(args: _NoArgs) -> ToolResult[dict[str, str]]:
    _ = args
    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, str]] = loop.create_future()

    async def _finish() -> None:
        await asyncio.sleep(0.001)
        if not future.done():
            future.set_result({"status": "ok", "artifact": "dist/app"})

    asyncio.create_task(_finish())
    return ToolResult(
        output=None,
        success=True,
        deferred=ToolDeferredHandle(
            ticket_id="build-ticket-1",
            tool_name="build_project",
            status="running",
            summary="Build started",
            resume_hint="Continue docs while build runs",
            poll_after_s=0.01,
        ),
        metadata={"background_future": future},
    )


class _BackgroundAwareLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def provider_id(self) -> str:
        return "bg-llm"

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
                        id="tc_build_1",
                        tool_name="build_project",
                        arguments={},
                    )
                ],
                model=req.model,
            )

        has_build_result = False
        for message in req.messages:
            if message.role != "tool" or message.name != "build_project":
                continue
            if isinstance(message.content, str) and '"status": "ok"' in message.content:
                has_build_result = True
                break

        if has_build_result:
            return LLMResponse(text="Build done, applied fixes and finalized docs.")

        return LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id=f"tc_docs_{self.calls}",
                    tool_name="write_docs",
                    arguments={},
                )
            ],
            model=req.model,
        )

    async def _chat_stream_core(self, req: LLMRequest, *, response_model=None):
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


@tool(args_model=_NoArgs, name="build_project_external")
def build_project_external(args: _NoArgs) -> ToolResult[dict[str, str]]:
    _ = args
    return ToolResult(
        output=None,
        success=True,
        deferred=ToolDeferredHandle(
            ticket_id="build-ticket-ext",
            tool_name="build_project_external",
            status="running",
            summary="External build queued",
            resume_hint="Await external worker completion",
            poll_after_s=0.01,
        ),
    )


class _ResumeAwareLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def provider_id(self) -> str:
        return "resume-bg-llm"

    @property
    def capabilities(self) -> LLMCapabilities:
        return LLMCapabilities(chat=True, streaming=False, tool_calling=True)

    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        _ = response_model
        self.calls += 1
        for message in req.messages:
            if (
                message.role == "tool"
                and message.name == "build_project_external"
                and isinstance(message.content, str)
                and '"status": "ok"' in message.content
            ):
                return LLMResponse(text="Build resumed and completed after restart.")
        if self.calls == 1:
            return LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        id="tc_build_ext_1",
                        tool_name="build_project_external",
                        arguments={},
                    )
                ],
                model=req.model,
            )
        return LLMResponse(
            text="",
            tool_calls=[
                ToolCall(
                    id=f"tc_docs_resume_{self.calls}",
                    tool_name="write_docs",
                    arguments={},
                )
            ],
            model=req.model,
        )

    async def _chat_stream_core(self, req: LLMRequest, *, response_model=None):
        _ = req
        _ = response_model
        raise NotImplementedError

    async def _embed_core(self, req: EmbeddingRequest) -> EmbeddingResponse:
        _ = req
        raise NotImplementedError


class _SlowResumeAwareLLM(_ResumeAwareLLM):
    async def _chat_core(self, req: LLMRequest, *, response_model=None) -> LLMResponse:
        await asyncio.sleep(0.03)
        return await super()._chat_core(req, response_model=response_model)


def test_background_tool_defer_resolve_and_stream_events():
    async def _scenario():
        runner = Runner(
            config=RunnerConfig(
                sanitize_tool_output=False,
                background_tools_enabled=True,
                background_tool_poll_interval_s=0.01,
                background_tool_result_ttl_s=5.0,
                background_tool_interrupt_on_resolve=True,
            )
        )
        agent = Agent(
            model=_BackgroundAwareLLM(),
            instructions="x",
            tools=[build_project, write_docs],
            fail_safe=FailSafeConfig(max_steps=20),
        )

        stream = await runner.run_stream(agent, user_message="build and document")
        event_types: list[str] = []
        errors: list[str] = []
        async for event in stream:
            event_types.append(event.type)
            if event.type == "error" and event.error:
                errors.append(event.error)

        result = stream.result
        assert result is not None, f"stream errors={errors!r}, events={event_types!r}"
        return result, event_types

    result, event_types = asyncio.run(_scenario())
    assert "tool_deferred" in event_types
    assert "tool_background_resolved" in event_types
    assert any(row.tool_name == "build_project" for row in result.tool_executions)
    assert "finalized docs" in result.final_text


def test_background_pending_ticket_survives_resume_after_restart(monkeypatch):
    async def _scenario():
        memory = InMemoryMemoryStore()
        cfg = RunnerConfig(
            sanitize_tool_output=False,
            background_tools_enabled=True,
            background_tool_poll_interval_s=0.01,
            background_tool_result_ttl_s=5.0,
        )
        runner_a = Runner(memory_store=memory, config=cfg)
        agent = Agent(
            model=_ResumeAwareLLM(),
            instructions="x",
            tools=[build_project_external, write_docs],
            fail_safe=FailSafeConfig(max_steps=30),
        )
        original_persist = runner_a._persist_checkpoint
        crash_once = {"raised": False}

        async def crash_after_pending_runtime(
            *,
            memory,
            thread_id,
            run_id,
            step,
            phase,
            payload,
        ):
            if (
                phase == "step_started"
                and step == 2
                and not crash_once["raised"]
            ):
                crash_once["raised"] = True
                raise RuntimeError("simulated restart before background resolution")
            return await original_persist(
                memory=memory,
                thread_id=thread_id,
                run_id=run_id,
                step=step,
                phase=phase,
                payload=payload,
            )

        monkeypatch.setattr(runner_a, "_persist_checkpoint", crash_after_pending_runtime)

        handle = await runner_a.run_handle(agent, user_message="run external build")
        run_id: str | None = None
        thread_id: str | None = None
        ticket_id: str | None = None
        async for event in handle.events:
            if run_id is None:
                run_id = event.run_id
                thread_id = event.thread_id
            if event.type == "tool_deferred":
                maybe_ticket = event.data.get("ticket_id")
                if isinstance(maybe_ticket, str):
                    ticket_id = maybe_ticket
        try:
            _ = await handle.await_result()
            assert False, "expected simulated crash"
        except AgentExecutionError:
            pass
        assert run_id and thread_id and ticket_id

        state = await memory.list_state(thread_id)
        runtime_key = f"checkpoint:{run_id}:1:runtime_state"
        latest_key = f"checkpoint:{run_id}:latest"
        assert runtime_key in state
        await memory.put_state(thread_id, latest_key, state[runtime_key])

        await memory.put_state(
            thread_id,
            f"bgtool:{run_id}:{ticket_id}:state",
            {
                "run_id": run_id,
                "thread_id": thread_id,
                "ticket_id": ticket_id,
                "tool_name": "build_project_external",
                "status": "completed",
                "output": {"status": "ok", "artifact": "dist/resumed"},
            },
        )

        runner_b = Runner(memory_store=memory, config=cfg)
        resumed = await runner_b.resume(
            agent,
            run_id=run_id,
            thread_id=thread_id,
        )
        return resumed

    result = asyncio.run(_scenario())
    assert "resumed and completed" in result.final_text
    assert any(
        row.tool_name == "build_project_external"
        and isinstance(row.output, dict)
        and row.output.get("status") == "ok"
        for row in result.tool_executions
    )


def test_background_tool_ttl_expiry_emits_background_failed_event():
    async def _scenario():
        runner = Runner(
            config=RunnerConfig(
                sanitize_tool_output=False,
                background_tools_enabled=True,
                background_tool_poll_interval_s=0.005,
                background_tool_result_ttl_s=1.01,
            )
        )
        agent = Agent(
            model=_SlowResumeAwareLLM(),
            instructions="x",
            tools=[build_project_external, write_docs],
            fail_safe=FailSafeConfig(max_steps=80),
        )
        handle = await runner.run_handle(agent, user_message="run external build")
        seen_failed = False
        async for event in handle.events:
            if event.type == "tool_background_failed":
                seen_failed = True
                await handle.cancel()
                break
        _ = await handle.await_result()
        return seen_failed

    seen = asyncio.run(_scenario())
    assert seen is True


def test_runner_background_ticket_helpers_list_resolve_fail():
    async def _scenario():
        memory = InMemoryMemoryStore()
        runner = Runner(
            memory_store=memory,
            config=RunnerConfig(
                sanitize_tool_output=False,
                background_tools_enabled=True,
                background_tool_poll_interval_s=0.01,
                background_tool_result_ttl_s=5.0,
            ),
        )
        agent = Agent(
            model=_ResumeAwareLLM(),
            instructions="x",
            tools=[build_project_external, write_docs],
            fail_safe=FailSafeConfig(max_steps=20),
        )
        handle = await runner.run_handle(agent, user_message="run external build")
        run_id: str | None = None
        thread_id: str | None = None
        ticket_id: str | None = None
        async for event in handle.events:
            if run_id is None:
                run_id = event.run_id
                thread_id = event.thread_id
            if event.type == "tool_deferred":
                maybe = event.data.get("ticket_id")
                if isinstance(maybe, str):
                    ticket_id = maybe
                await handle.cancel()
                break
        _ = await handle.await_result()
        assert run_id and thread_id and ticket_id

        pending = await runner.list_background_tools(
            thread_id=thread_id,
            run_id=run_id,
        )
        assert any(row.get("ticket_id") == ticket_id for row in pending)

        await runner.resolve_background_tool(
            thread_id=thread_id,
            run_id=run_id,
            ticket_id=ticket_id,
            output={"status": "ok"},
            tool_name="build_project_external",
        )
        resolved = await runner.list_background_tools(
            thread_id=thread_id,
            run_id=run_id,
            include_resolved=True,
        )
        assert any(
            row.get("ticket_id") == ticket_id and row.get("status") == "completed"
            for row in resolved
        )

        await runner.fail_background_tool(
            thread_id=thread_id,
            run_id=run_id,
            ticket_id=ticket_id,
            error="manual fail",
            tool_name="build_project_external",
        )
        failed = await runner.list_background_tools(
            thread_id=thread_id,
            run_id=run_id,
            include_resolved=True,
        )
        assert any(
            row.get("ticket_id") == ticket_id
            and row.get("status") == "failed"
            and row.get("error") == "manual fail"
            for row in failed
        )

    asyncio.run(_scenario())


def test_background_grace_can_resolve_before_next_step_when_interrupt_hint_enabled():
    async def _scenario():
        runner = Runner(
            config=RunnerConfig(
                sanitize_tool_output=False,
                background_tools_enabled=True,
                background_tool_default_grace_s=0.02,
                background_tool_interrupt_on_resolve=True,
                background_tool_poll_interval_s=0.005,
                background_tool_result_ttl_s=5.0,
            )
        )
        agent = Agent(
            model=_BackgroundAwareLLM(),
            instructions="x",
            tools=[build_project, write_docs],
            fail_safe=FailSafeConfig(max_steps=20),
        )
        stream = await runner.run_stream(agent, user_message="build and document")
        sequence: list[str] = []
        async for event in stream:
            sequence.append(event.type)
        return sequence

    sequence = asyncio.run(_scenario())
    assert "tool_deferred" in sequence
    assert "tool_background_resolved" in sequence
    assert sequence.index("tool_background_resolved") < sequence.index("step_started", 1)
