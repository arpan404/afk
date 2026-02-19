"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Shared runtime types for AFK runner internals.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import AsyncIterator

from ...agents.types import (
    AgentResult,
    AgentRunEvent,
    AgentRunHandle,
    DecisionKind,
    InteractionMode,
)
from ...tools.security import (
    SandboxProfile,
    SandboxProfileProvider,
    SecretScopeProvider,
)


_RUN_END = object()


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    """
    Runtime configuration for runner behavior and safety defaults.

    Attributes:
        interaction_mode: Interaction strategy (`headless`, `interactive`,
            `external`).
        approval_timeout_s: Timeout for deferred approval decisions.
        input_timeout_s: Timeout for deferred user-input decisions.
        approval_fallback: Fallback decision when approval times out.
        input_fallback: Fallback decision when user input times out.
        sanitize_tool_output: Enable sanitizer for model-visible tool output.
        untrusted_tool_preamble: Inject untrusted-data warning preamble.
        tool_output_max_chars: Max tool output characters forwarded to model.
        default_sandbox_profile: Optional default sandbox profile.
        sandbox_profile_provider: Optional runtime sandbox profile resolver.
        secret_scope_provider: Optional secret-scope resolver per tool call.
        default_allowlisted_commands: Default allowlisted shell commands for
            runtime/skill command tools.
        max_parallel_subagents_global: Global cap across all runs for
            concurrently executing subagent tasks.
        max_parallel_subagents_per_parent: Per-parent-run cap for concurrent
            subagent fanout.
        max_parallel_subagents_per_target_agent: Per-target-agent cap to avoid
            overloading one specialist under broad fanout.
        subagent_queue_backpressure_limit: Maximum pending subagent nodes per
            parent run before backpressure is raised.
        checkpoint_async_writes: Enable asynchronous checkpoint/state writes.
        checkpoint_queue_maxsize: Maximum queued checkpoint writes.
        checkpoint_flush_timeout_s: Timeout for terminal checkpoint flush.
        checkpoint_coalesce_runtime_state: Coalesce runtime-state writes by key.
    """

    interaction_mode: InteractionMode = "headless"
    approval_timeout_s: float = 300.0
    input_timeout_s: float = 300.0
    approval_fallback: DecisionKind = "deny"
    input_fallback: DecisionKind = "deny"
    sanitize_tool_output: bool = True
    untrusted_tool_preamble: bool = True
    tool_output_max_chars: int = 12_000
    default_sandbox_profile: "SandboxProfile | None" = None
    sandbox_profile_provider: "SandboxProfileProvider | None" = None
    secret_scope_provider: "SecretScopeProvider | None" = None
    default_allowlisted_commands: tuple[str, ...] = (
        "ls",
        "cat",
        "head",
        "tail",
        "rg",
        "find",
        "pwd",
        "echo",
    )
    max_parallel_subagents_global: int = 64
    max_parallel_subagents_per_parent: int = 8
    max_parallel_subagents_per_target_agent: int = 4
    subagent_queue_backpressure_limit: int = 512
    checkpoint_async_writes: bool = True
    checkpoint_queue_maxsize: int = 1024
    checkpoint_flush_timeout_s: float = 10.0
    checkpoint_coalesce_runtime_state: bool = True


class _RunHandle(AgentRunHandle):
    """
    Concrete async run handle used by the runner implementation.

    The handle is single-consumer for events and supports cooperative pause,
    cancel, and interrupt controls.
    """

    def __init__(self) -> None:
        """Initialize queue, result future, and lifecycle flags."""
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._result_fut: asyncio.Future[AgentResult | None] = (
            asyncio.get_running_loop().create_future()
        )
        self._task: asyncio.Task[None] | None = None
        self._events_consumed = False
        self._paused_by_user = False
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._cancel_requested = False
        self._interrupt_requested = False
        self._interrupt_cb = None
        self._stream_text_deltas = False

    def attach_task(self, task: asyncio.Task[None]) -> None:
        """
        Attach the underlying execution task.

        Args:
            task: Background task executing the run loop.
        """
        self._task = task

    def set_interrupt_callback(self, callback) -> None:
        """
        Register provider-specific interrupt callback.

        Args:
            callback: Callable invoked on `interrupt()` before cancellation.
        """
        self._interrupt_cb = callback

    @property
    def events(self) -> AsyncIterator[AgentRunEvent]:
        """
        Return run event stream.

        Returns:
            Async iterator of `AgentRunEvent`.

        Raises:
            RuntimeError: If events stream is requested by multiple consumers.
        """
        if self._events_consumed:
            raise RuntimeError("AgentRunHandle.events supports a single consumer")
        self._events_consumed = True
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[AgentRunEvent]:
        """Internal async generator that yields queued events until run end marker."""
        while True:
            item = await self._queue.get()
            if item is _RUN_END:
                break
            yield item  # type: ignore[misc]

    async def emit(self, event: AgentRunEvent) -> None:
        """
        Push event into handle stream.

        Args:
            event: Runtime event to publish.
        """
        await self._queue.put(event)

    async def pause(self) -> None:
        """Pause cooperative execution at safe boundaries."""
        self._paused_by_user = True
        self._resume_event.clear()

    async def resume(self) -> None:
        """Resume a paused run."""
        self._paused_by_user = False
        self._resume_event.set()

    async def cancel(self) -> None:
        """
        Request cancellation and resolve handle with `None`.

        Cancellation propagates to execution task and awaits task cleanup.
        """
        self._cancel_requested = True
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        if not self._result_fut.done():
            await self.set_result(None)

    async def interrupt(self) -> None:
        """
        Request interruption and invoke interrupt callback if available.

        Interruption sets interrupt flag, invokes callback (sync or async), and
        cancels running task.
        """
        self._interrupt_requested = True
        callback = self._interrupt_cb
        if callback is not None:
            maybe = callback()
            if inspect.isawaitable(maybe):
                await maybe
        if self._task is not None and not self._task.done():
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def await_result(self) -> AgentResult | None:
        """
        Await terminal run result.

        Returns:
            `AgentResult` on completion, or `None` when cancelled.
        """
        return await self._result_fut

    async def wait_if_paused(self) -> None:
        """Block until resume is requested."""
        await self._resume_event.wait()

    def is_cancel_requested(self) -> bool:
        """Return `True` when cancellation has been requested."""
        return self._cancel_requested

    def is_interrupt_requested(self) -> bool:
        """Return `True` when interruption has been requested."""
        return self._interrupt_requested

    async def set_result(self, result: AgentResult | None) -> None:
        """
        Set terminal result and close event stream.

        Args:
            result: Final result payload, or `None` for cancelled runs.
        """
        if not self._result_fut.done():
            self._result_fut.set_result(result)
        await self._queue.put(_RUN_END)

    def enable_stream_text_deltas(self) -> None:
        """Enable emission of incremental text-delta run events."""
        self._stream_text_deltas = True

    def stream_text_deltas_enabled(self) -> bool:
        """Return whether incremental text deltas are enabled for this run."""
        return self._stream_text_deltas

    async def set_exception(self, exc: Exception) -> None:
        """
        Set terminal exception and close event stream.

        Args:
            exc: Exception to propagate from `await_result()`.
        """
        if not self._result_fut.done():
            self._result_fut.set_exception(exc)
        await self._queue.put(_RUN_END)
