"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Public runner API and lifecycle entrypoints.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from ...agents.core.base import BaseAgent
from ...agents.errors import (
    AgentCancelledError,
    AgentCheckpointCorruptionError,
    AgentConfigurationError,
)
from ...agents.policy.engine import PolicyEngine
from ...agents.lifecycle.runtime import EffectJournal, checkpoint_latest_key
from ...agents.types import AgentResult, AgentRunHandle, json_value_from_tool_result
from ...llms.types import JSONValue
from ...memory import (
    MemoryCompactionResult,
    MemoryStore,
    RetentionPolicy,
    StateRetentionPolicy,
    compact_thread_memory,
    now_ms,
)
from ...observability.backends import create_telemetry_sink
from ..interaction import HeadlessInteractionProvider, InteractionProvider
from ..telemetry import TelemetrySink
from .types import RunnerConfig, _RunHandle

if TYPE_CHECKING:
    from ..streaming import AgentStreamHandle


class RunnerAPIMixin:
    """
    Public API surface for running, resuming, and compacting agent threads.

    This mixin owns dependency wiring (memory, interaction, policy, telemetry)
    and exposes the stable entrypoints used by `Agent.call(...)` and external
    runtime integrations.
    """

    def __init__(
        self,
        *,
        memory_store: MemoryStore | None = None,
        interaction_provider: InteractionProvider | None = None,
        policy_engine: PolicyEngine | None = None,
        telemetry: str | TelemetrySink | None = None,
        telemetry_config: Mapping[str, JSONValue] | None = None,
        config: RunnerConfig | None = None,
    ) -> None:
        """
        Initialize a runner API surface with optional runtime dependencies.

        Args:
            memory_store: Memory backend. When `None`, runtime resolves from
                environment on first use and may fallback to in-memory.
            interaction_provider: Human-in-the-loop provider. Required when
                `config.interaction_mode` is not `headless`.
            policy_engine: Optional deterministic policy engine shared across runs.
            telemetry: Telemetry sink instance or backend id.
            telemetry_config: Optional backend-specific sink config.
            config: Runner configuration. Defaults to `RunnerConfig()`.

        Raises:
            AgentConfigurationError: If interaction mode requires provider but
                none is supplied.
        """
        self.config = config or RunnerConfig()
        self._memory_store = memory_store
        self._owns_memory_store = memory_store is None
        self._memory_fallback_reason: str | None = None
        if interaction_provider is None:
            if self.config.interaction_mode == "headless":
                self._interaction = HeadlessInteractionProvider(
                    approval_fallback=self.config.approval_fallback,
                    input_fallback=self.config.input_fallback,
                )
            else:
                raise AgentConfigurationError(
                    "interaction_provider is required when interaction_mode is not 'headless'"
                )
        else:
            self._interaction = interaction_provider
        self._policy_engine = policy_engine
        self._telemetry = create_telemetry_sink(
            telemetry,
            config=telemetry_config,
        )
        self._effect_journal = EffectJournal()
        self._active_runs = 0
        self._memory_store_lock = asyncio.Lock()
        self._checkpoint_queue = None
        self._checkpoint_writer_task = None
        self._checkpoint_pending_count = 0
        self._checkpoint_pending_event = asyncio.Event()
        self._checkpoint_pending_event.set()
        self._checkpoint_coalesce_buffer = {}
        self._checkpoint_coalesce_keys = set()
        self._background_lock = asyncio.Lock()
        self._background_pending = {}
        self._background_ready = {}
        self._background_poller_task = None
        self._background_interrupt_hints = set()

    async def compact_thread(
        self,
        *,
        thread_id: str,
        event_policy: RetentionPolicy | None = None,
        state_policy: StateRetentionPolicy | None = None,
    ) -> MemoryCompactionResult:
        """
        Compact retained memory records for a thread.

        Args:
            thread_id: Thread identifier whose memory should be compacted.
            event_policy: Optional override for event retention compaction.
            state_policy: Optional override for state retention compaction.

        Returns:
            Memory compaction summary for the thread.

        Raises:
            AgentConfigurationError: If `thread_id` is empty.
        """
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise AgentConfigurationError("thread_id must be a non-empty string")
        memory = await self._ensure_memory_store()
        return await compact_thread_memory(
            memory,
            thread_id=thread_id,
            event_policy=event_policy,
            state_policy=state_policy,
        )

    async def list_background_tools(
        self,
        *,
        thread_id: str,
        run_id: str,
        include_resolved: bool = False,
    ) -> list[dict[str, JSONValue]]:
        """
        List persisted background-tool ticket rows for a run.

        Args:
            thread_id: Thread identifier.
            run_id: Run identifier.
            include_resolved: Include completed/failed rows.

        Returns:
            Sorted list of JSON-safe ticket rows.
        """
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise AgentConfigurationError("thread_id must be a non-empty string")
        if not isinstance(run_id, str) or not run_id.strip():
            raise AgentConfigurationError("run_id must be a non-empty string")
        memory = await self._ensure_memory_store()
        rows = await memory.list_state(thread_id, prefix=f"bgtool:{run_id}:")
        out: list[dict[str, JSONValue]] = []
        for key, value in sorted(rows.items(), key=lambda item: item[0]):
            if key.endswith(":latest"):
                continue
            if not isinstance(value, dict):
                continue
            status = value.get("status")
            if not include_resolved and status in {"completed", "failed"}:
                continue
            out.append({str(k): json_value_from_tool_result(v) for k, v in value.items()})
        return out

    async def resolve_background_tool(
        self,
        *,
        thread_id: str,
        run_id: str,
        ticket_id: str,
        output: JSONValue | None = None,
        tool_name: str | None = None,
    ) -> None:
        """
        Mark a background ticket as completed.
        """
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise AgentConfigurationError("thread_id must be a non-empty string")
        if not isinstance(run_id, str) or not run_id.strip():
            raise AgentConfigurationError("run_id must be a non-empty string")
        if not isinstance(ticket_id, str) or not ticket_id.strip():
            raise AgentConfigurationError("ticket_id must be a non-empty string")
        memory = await self._ensure_memory_store()
        state_key = self._background_state_key(run_id, ticket_id)
        prior = await memory.get_state(thread_id, state_key)
        prior_tool_name = (
            prior.get("tool_name")
            if isinstance(prior, dict) and isinstance(prior.get("tool_name"), str)
            else None
        )
        row = {
            "run_id": run_id,
            "thread_id": thread_id,
            "ticket_id": ticket_id,
            "tool_name": tool_name or prior_tool_name or "",
            "status": "completed",
            "output": output,
            "error": None,
            "resolved_at_ms": now_ms(),
        }
        await memory.put_state(thread_id, state_key, row)  # type: ignore[arg-type]
        await memory.put_state(
            thread_id,
            self._background_latest_key(run_id),
            row,  # type: ignore[arg-type]
        )

    async def fail_background_tool(
        self,
        *,
        thread_id: str,
        run_id: str,
        ticket_id: str,
        error: str,
        tool_name: str | None = None,
    ) -> None:
        """
        Mark a background ticket as failed.
        """
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise AgentConfigurationError("thread_id must be a non-empty string")
        if not isinstance(run_id, str) or not run_id.strip():
            raise AgentConfigurationError("run_id must be a non-empty string")
        if not isinstance(ticket_id, str) or not ticket_id.strip():
            raise AgentConfigurationError("ticket_id must be a non-empty string")
        if not isinstance(error, str) or not error.strip():
            raise AgentConfigurationError("error must be a non-empty string")
        memory = await self._ensure_memory_store()
        state_key = self._background_state_key(run_id, ticket_id)
        prior = await memory.get_state(thread_id, state_key)
        prior_tool_name = (
            prior.get("tool_name")
            if isinstance(prior, dict) and isinstance(prior.get("tool_name"), str)
            else None
        )
        row = {
            "run_id": run_id,
            "thread_id": thread_id,
            "ticket_id": ticket_id,
            "tool_name": tool_name or prior_tool_name or "",
            "status": "failed",
            "output": None,
            "error": error.strip(),
            "resolved_at_ms": now_ms(),
        }
        await memory.put_state(thread_id, state_key, row)  # type: ignore[arg-type]
        await memory.put_state(
            thread_id,
            self._background_latest_key(run_id),
            row,  # type: ignore[arg-type]
        )

    async def run(
        self,
        agent: BaseAgent,
        *,
        user_message: str | None = None,
        context: dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> AgentResult:
        """
        Execute an agent run and wait for terminal result.

        Args:
            agent: Agent definition to execute.
            user_message: Optional initial user message.
            context: Optional JSON-like run context.
            thread_id: Optional thread id for memory continuity.

        Returns:
            Terminal agent result.

        Raises:
            AgentCancelledError: If run is cancelled before completion.
        """
        handle = await self.run_handle(
            agent,
            user_message=user_message,
            context=context,
            thread_id=thread_id,
        )
        result = await handle.await_result()
        if result is None:
            raise AgentCancelledError("Run cancelled")
        return result

    def run_sync(
        self,
        agent: BaseAgent,
        *,
        user_message: str | None = None,
        context: dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> AgentResult:
        """
        Execute an agent run synchronously (blocking).

        Convenience wrapper around :meth:`run` for scripts and CLIs that
        do not have their own event loop. Raises ``RuntimeError`` if called
        from inside an already-running async event loop.

        Args:
            agent: Agent definition to execute.
            user_message: Optional initial user message.
            context: Optional JSON-like run context.
            thread_id: Optional thread id for memory continuity.

        Returns:
            Terminal agent result.

        Raises:
            AgentCancelledError: If run is cancelled before completion.
            RuntimeError: If called inside a running event loop.
        """
        from ...llms.utils import run_sync as _run_sync

        return _run_sync(
            self.run(
                agent,
                user_message=user_message,
                context=context,
                thread_id=thread_id,
            )
        )

    async def resume(
        self,
        agent: BaseAgent,
        *,
        run_id: str,
        thread_id: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Resume a previously checkpointed run and wait for completion.

        Args:
            agent: Agent definition to resume.
            run_id: Existing run identifier.
            thread_id: Existing thread identifier.
            context: Optional context overlay for resumed execution.

        Returns:
            Terminal agent result.

        Raises:
            AgentCancelledError: If resumed run is cancelled.
        """
        handle = await self.resume_handle(
            agent,
            run_id=run_id,
            thread_id=thread_id,
            context=context,
        )
        result = await handle.await_result()
        if result is None:
            raise AgentCancelledError("Run cancelled")
        return result

    async def resume_handle(
        self,
        agent: BaseAgent,
        *,
        run_id: str,
        thread_id: str,
        context: dict[str, Any] | None = None,
    ) -> AgentRunHandle:
        """
        Resume a run and return a live handle for lifecycle control.

        If the latest checkpoint already contains a terminal result, this method
        returns a handle pre-populated with that result.

        Args:
            agent: Agent definition used for continued execution.
            run_id: Existing run identifier.
            thread_id: Existing thread identifier.
            context: Optional context overlay for resumed execution.

        Returns:
            Active run handle or pre-resolved terminal handle.

        Raises:
            AgentConfigurationError: If `run_id`/`thread_id` is invalid.
            AgentCheckpointCorruptionError: If checkpoint chain is missing or
                invalid for the given run.
        """
        if not isinstance(run_id, str) or not run_id.strip():
            raise AgentConfigurationError("run_id must be a non-empty string")
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise AgentConfigurationError("thread_id must be a non-empty string")

        memory = await self._ensure_memory_store()
        latest = await memory.get_state(thread_id, checkpoint_latest_key(run_id))
        if not isinstance(latest, dict):
            raise AgentCheckpointCorruptionError(
                f"No checkpoint found for run_id={run_id} thread_id={thread_id}"
            )
        latest = self._normalize_checkpoint_record(latest)

        payload = latest.get("payload")
        phase = latest.get("phase")
        if isinstance(payload, dict):
            terminal = payload.get("terminal_result")
            if phase == "run_terminal" and isinstance(terminal, dict):
                handle = _RunHandle()
                result = self._deserialize_agent_result(terminal)
                await handle.set_result(result)
                return handle

        resume_snapshot = await self._load_latest_runtime_snapshot(
            memory=memory,
            thread_id=thread_id,
            run_id=run_id,
            latest=latest,
        )

        return await self.run_handle(
            agent,
            context=context,
            thread_id=thread_id,
            _resume_run_id=run_id,
            _resume_snapshot=resume_snapshot,
        )

    async def run_handle(
        self,
        agent: BaseAgent,
        *,
        user_message: str | None = None,
        context: dict[str, Any] | None = None,
        thread_id: str | None = None,
        _depth: int = 0,
        _lineage: tuple[int, ...] = (),
        _resume_run_id: str | None = None,
        _resume_snapshot: dict[str, Any] | None = None,
        _stream_text_deltas: bool = False,
    ) -> AgentRunHandle:
        """
        Start execution and return an async run handle.

        Args:
            agent: Agent definition to execute.
            user_message: Optional initial user message.
            context: Optional JSON-like run context.
            thread_id: Optional thread id for memory continuity.
            _depth: Internal recursion depth for subagent execution.
            _lineage: Internal lineage tuple used for tracing nested runs.
            _resume_run_id: Internal run id for resume continuation.
            _resume_snapshot: Internal restored snapshot payload.

        Returns:
            Handle exposing event stream and run lifecycle controls.
        """
        handle = _RunHandle()
        if _stream_text_deltas:
            handle.enable_stream_text_deltas()
        task = asyncio.create_task(
            self._execute(
                handle,
                agent,
                user_message=user_message,
                context=context,
                thread_id=thread_id,
                depth=_depth,
                lineage=_lineage,
                resume_run_id=_resume_run_id,
                resume_snapshot=_resume_snapshot,
            )
        )
        handle.attach_task(task)
        return handle

    async def run_stream(
        self,
        agent: BaseAgent,
        *,
        user_message: str | None = None,
        context: dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> AgentStreamHandle:
        """
        Start an agent run and return a stream handle for real-time events.

        The stream yields ``AgentStreamEvent`` instances including text deltas,
        tool lifecycle events, and a terminal ``completed`` event with the
        final ``AgentResult``.

        Usage::

            handle = await runner.run_stream(agent, user_message="Hi")
            async for event in handle:
                if event.type == "text_delta":
                    print(event.text_delta, end="", flush=True)
            result = handle.result

        Args:
            agent: Agent definition to execute.
            user_message: Optional initial user message.
            context: Optional JSON-like run context.
            thread_id: Optional thread id for memory continuity.

        Returns:
            ``AgentStreamHandle`` for consuming stream events.
        """
        from ..streaming import (
            AgentStreamEvent,
            AgentStreamHandle,
            stream_completed,
            stream_error,
            text_delta,
            tool_deferred as _tool_deferred,
            tool_started as _tool_started,
            step_started as _step_started,
        )

        run_handle = await self.run_handle(
            agent,
            user_message=user_message,
            context=context,
            thread_id=thread_id,
            _stream_text_deltas=True,
        )
        stream = AgentStreamHandle()

        async def _bridge() -> None:
            """Bridge run events → stream events."""
            saw_text_delta = False

            def _chunk_text(value: str) -> list[str]:
                chunks: list[str] = []
                current = ""
                for line in value.splitlines(keepends=True):
                    current += line
                    if line.endswith((".", "!", "?", "\n")) and current.strip():
                        chunks.append(current)
                        current = ""
                if current.strip():
                    chunks.append(current)
                return chunks if chunks else [value]

            try:
                async for event in run_handle.events:
                    # Map known event types to stream events
                    if event.type == "text_delta" and event.data:
                        delta = event.data.get("delta")
                        if isinstance(delta, str) and delta:
                            saw_text_delta = True
                            await stream.emit(
                                text_delta(delta, step=event.step)
                            )
                    elif event.type == "llm_completed" and event.data:
                        response_text = event.data.get("text", "")
                        if (
                            isinstance(response_text, str)
                            and response_text
                            and not saw_text_delta
                        ):
                            saw_text_delta = True
                            for chunk in _chunk_text(response_text):
                                await stream.emit(text_delta(chunk, step=event.step))
                    elif event.type == "tool_batch_started" and event.data:
                        tool_names = event.data.get("tool_names")
                        tool_ids = event.data.get("tool_call_ids")
                        if isinstance(tool_names, list):
                            for idx, tn in enumerate(tool_names):
                                call_id = None
                                if isinstance(tool_ids, list) and idx < len(tool_ids):
                                    maybe_id = tool_ids[idx]
                                    if isinstance(maybe_id, str):
                                        call_id = maybe_id
                                await stream.emit(
                                    _tool_started(
                                        str(tn),
                                        tool_call_id=call_id,
                                        step=event.step,
                                    )
                                )
                    elif event.type == "tool_completed" and event.data:
                        await stream.emit(
                            AgentStreamEvent(
                                type="tool_completed",
                                tool_name=str(event.data.get("tool_name", "")),
                                tool_call_id=event.data.get("tool_call_id")
                                if isinstance(event.data.get("tool_call_id"), str)
                                else None,
                                tool_success=bool(event.data.get("success", False)),
                                tool_output=event.data.get("output"),
                                tool_error=event.data.get("error")
                                if isinstance(event.data.get("error"), str)
                                else None,
                                step=event.step,
                                data=dict(event.data),
                            )
                        )
                    elif event.type == "tool_deferred" and event.data:
                        await stream.emit(
                            _tool_deferred(
                                str(event.data.get("tool_name", "")),
                                tool_call_id=event.data.get("tool_call_id")
                                if isinstance(event.data.get("tool_call_id"), str)
                                else None,
                                ticket_id=event.data.get("ticket_id")
                                if isinstance(event.data.get("ticket_id"), str)
                                else None,
                                step=event.step,
                                data=dict(event.data),
                            )
                        )
                    elif event.type in (
                        "tool_background_resolved",
                        "tool_background_failed",
                    ) and event.data:
                        await stream.emit(
                            AgentStreamEvent(
                                type=event.type,
                                tool_name=str(event.data.get("tool_name", "")),
                                tool_call_id=event.data.get("tool_call_id")
                                if isinstance(event.data.get("tool_call_id"), str)
                                else None,
                                tool_ticket_id=event.data.get("ticket_id")
                                if isinstance(event.data.get("ticket_id"), str)
                                else None,
                                tool_success=bool(event.data.get("success", False)),
                                tool_output=event.data.get("output"),
                                tool_error=event.data.get("error")
                                if isinstance(event.data.get("error"), str)
                                else None,
                                step=event.step,
                                data=dict(event.data),
                            )
                        )
                    elif event.type == "step_started":
                        await stream.emit(
                            _step_started(
                                step=int(event.step or 0),
                                state=event.state,
                            )
                        )
                    elif event.type in ("run_failed", "run_interrupted"):
                        error_msg = (
                            event.data.get("error", str(event.type))
                            if event.data
                            else str(event.type)
                        )
                        await stream.emit(stream_error(str(error_msg)))

                # Run completed — emit terminal event
                result = await run_handle.await_result()
                if result is not None:
                    if result.final_text and not saw_text_delta:
                        for chunk in _chunk_text(result.final_text):
                            await stream.emit(text_delta(chunk))
                    await stream.emit(stream_completed(result))
            except Exception as exc:
                await stream.emit(stream_error(str(exc)))
            finally:
                await stream.close()

        asyncio.create_task(_bridge())
        return stream
