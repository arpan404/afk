"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Runner helpers for persistence, serialization, budgeting, and replay logic.
"""

from __future__ import annotations

import time
import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ...agents.errors import (
    AgentBudgetExceededError,
    AgentCancelledError,
    AgentCheckpointCorruptionError,
    AgentInterruptedError,
    AgentLoopLimitError,
)
from ...agents.model.resolution import resolve_model_to_llm
from ...agents.lifecycle.runtime import (
    checkpoint_latest_key,
    checkpoint_state_key,
    effect_state_key,
    json_hash,
    state_snapshot,
    validate_state_transition,
)
from ...agents.lifecycle.versioning import (
    CHECKPOINT_SCHEMA_VERSION,
    check_checkpoint_schema_version,
    migrate_checkpoint_record,
)
from ...agents.types import (
    AgentResult,
    AgentRunEvent,
    AgentState,
    CommandExecutionRecord,
    FailSafeConfig,
    SkillReadRecord,
    SubagentExecutionRecord,
    ToolExecutionRecord,
    UsageAggregate,
    json_value_from_tool_result,
)
from ...llms.types import JSONValue, LLMRequest, LLMResponse, Message, ToolCall, Usage
from ...llms.types import StreamCompletedEvent, StreamTextDeltaEvent
from ...memory import (
    InMemoryMemoryStore,
    MemoryEvent,
    MemoryStore,
    create_memory_store_from_env,
    new_id,
    now_ms,
)
from ...observability import contracts as obs_contracts
from ...tools import ToolResult
from ..telemetry import TelemetryEvent, TelemetrySpan
from .types import _RunHandle


@dataclass(slots=True)
class _CheckpointWrite:
    thread_id: str
    key: str
    value: dict[str, Any]
    coalesce: bool = False


@dataclass(slots=True, frozen=True)
class _CheckpointCoalesceToken:
    thread_id: str
    key: str


StreamTextDeltaSink = Callable[[str], Awaitable[None]]


class RunnerInternalsMixin:
    """Shared internal helpers used by the execution and API mixins."""

    def _enforce_budget(
        self,
        *,
        fail_safe: FailSafeConfig,
        step: int,
        llm_calls: int,
        tool_calls: int,
        started_at_s: float,
        total_cost_usd: float,
    ) -> None:
        """
        Enforce runtime budget and loop guards.

        Args:
            fail_safe: Active fail-safe configuration.
            step: Current step index.
            llm_calls: Number of LLM calls made so far.
            tool_calls: Number of tool calls made so far.
            started_at_s: Run start timestamp in seconds.
            total_cost_usd: Aggregated run cost.

        Raises:
            AgentBudgetExceededError: If time/call/cost budget exceeded.
            AgentLoopLimitError: If step limit exceeded.
        """
        elapsed = time.time() - started_at_s
        if elapsed > fail_safe.max_wall_time_s:
            raise AgentBudgetExceededError(
                f"Exceeded max_wall_time_s={fail_safe.max_wall_time_s}"
            )
        if llm_calls > fail_safe.max_llm_calls:
            raise AgentBudgetExceededError(
                f"Exceeded max_llm_calls={fail_safe.max_llm_calls}"
            )
        if tool_calls > fail_safe.max_tool_calls:
            raise AgentBudgetExceededError(
                f"Exceeded max_tool_calls={fail_safe.max_tool_calls}"
            )
        if step > fail_safe.max_steps:
            raise AgentLoopLimitError(f"Exceeded max_steps={fail_safe.max_steps}")
        if (
            fail_safe.max_total_cost_usd is not None
            and total_cost_usd > fail_safe.max_total_cost_usd
        ):
            raise AgentBudgetExceededError(
                f"Exceeded max_total_cost_usd={fail_safe.max_total_cost_usd}"
            )

    async def _emit(
        self,
        handle: _RunHandle,
        memory: MemoryStore,
        event: AgentRunEvent,
        *,
        user_id: str | None,
    ) -> None:
        """
        Emit lifecycle event to handle, interaction provider, memory, telemetry.

        Args:
            handle: Active run handle.
            memory: Memory backend.
            event: Event payload to emit.
            user_id: Optional user id for memory event.
        """
        await handle.emit(event)
        await self._interaction.notify(event)
        payload = {
            "schema_version": event.schema_version,
            "type": event.type,
            "state": event.state,
            "step": event.step or 0,
            "message": event.message or "",
            "data": event.data,
        }
        mem_event = MemoryEvent(
            id=self._new_id("evt"),
            thread_id=event.thread_id,
            user_id=user_id,
            type="trace",
            timestamp=now_ms(),
            payload=payload,  # type: ignore[arg-type]
            tags=["agent_run"],
        )
        await self._append_event_with_retry(memory, mem_event)
        try:
            self._telemetry.record_event(
                TelemetryEvent(
                    name=obs_contracts.AGENT_RUN_EVENT,
                    timestamp_ms=now_ms(),
                    attributes={
                        "event_type": event.type,
                        "run_id": event.run_id,
                        "thread_id": event.thread_id,
                        "state": event.state,
                        "step": event.step,
                        "user_id": user_id,
                    },
                )
            )
            self._telemetry.increment_counter(
                obs_contracts.AGENT_RUN_EVENTS_TOTAL,
                value=1,
                attributes={
                    "event_type": event.type,
                    "state": event.state,
                },
            )
        except Exception:
            # Telemetry failures should never break runtime execution.
            pass

    async def _append_event_with_retry(
        self,
        memory: MemoryStore,
        event: MemoryEvent,
    ) -> None:
        """
        Append memory event with best-effort retry.

        Args:
            memory: Memory backend.
            event: Event to append.
        """
        last_error: Exception | None = None
        for _ in range(3):
            try:
                await memory.append_event(event)
                return
            except Exception as e:
                last_error = e
                await asyncio.sleep(0)
        if last_error is not None:
            self._memory_fallback_reason = f"{self._memory_fallback_reason or 'memory_append_failed'}: {last_error}"

    async def _ensure_memory_store(self) -> MemoryStore:
        """
        Ensure memory store is initialized and ready.

        Returns:
            Ready-to-use memory store instance.
        """
        async with self._memory_store_lock:
            if self._memory_store is None:
                try:
                    self._memory_store = self._create_memory_store_from_env()
                    self._memory_fallback_reason = None
                except Exception as e:
                    self._memory_store = InMemoryMemoryStore()
                    self._memory_fallback_reason = str(e)
                self._owns_memory_store = True

            await self._memory_store.setup()
            return self._memory_store

    def _transition_state(self, current: AgentState, target: AgentState) -> AgentState:
        """
        Validate and return next state transition.

        Args:
            current: Current run state.
            target: Desired next state.

        Returns:
            Validated target state.
        """
        return validate_state_transition(current, target)

    async def _persist_checkpoint(
        self,
        *,
        memory: MemoryStore,
        thread_id: str,
        run_id: str,
        step: int,
        phase: str,
        payload: dict[str, Any],
    ) -> None:
        """
        Persist checkpoint for step/phase and update latest pointer.

        Args:
            memory: Memory backend.
            thread_id: Thread identifier.
            run_id: Run identifier.
            step: Current step.
            phase: Checkpoint phase label.
            payload: JSON-like checkpoint payload.
        """
        data = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": run_id,
            "step": step,
            "phase": phase,
            "timestamp_ms": now_ms(),
            "payload": {
                str(k): json_value_from_tool_result(v) for k, v in payload.items()
            },
        }
        state_key = checkpoint_state_key(run_id, step, phase)
        latest_key = checkpoint_latest_key(run_id)
        if self.config.checkpoint_async_writes:
            await self._ensure_checkpoint_writer()
            await self._queue_checkpoint_write(
                _CheckpointWrite(
                    thread_id=thread_id,
                    key=state_key,
                    value=data,
                    coalesce=(
                        self.config.checkpoint_coalesce_runtime_state
                        and phase == "runtime_state"
                    ),
                )
            )
            await self._queue_checkpoint_write(
                _CheckpointWrite(
                    thread_id=thread_id,
                    key=latest_key,
                    value=data,
                    coalesce=False,
                )
            )
            return
        await self._write_checkpoint_now(
            memory=memory,
            thread_id=thread_id,
            key=state_key,
            value=data,
        )
        await self._write_checkpoint_now(
            memory=memory,
            thread_id=thread_id,
            key=latest_key,
            value=data,
        )

    async def _write_checkpoint_now(
        self,
        *,
        memory: MemoryStore,
        thread_id: str,
        key: str,
        value: dict[str, Any],
    ) -> None:
        await memory.put_state(
            thread_id,
            key,
            value,  # type: ignore[arg-type]
        )

    async def _ensure_checkpoint_writer(self) -> None:
        if self._checkpoint_writer_task is not None and not self._checkpoint_writer_task.done():
            return
        if self._checkpoint_queue is None:
            self._checkpoint_queue = asyncio.Queue(
                maxsize=max(1, int(self.config.checkpoint_queue_maxsize))
            )
        self._checkpoint_writer_task = asyncio.create_task(self._checkpoint_writer_loop())

    async def _queue_checkpoint_write(self, write: _CheckpointWrite) -> None:
        queue = self._checkpoint_queue
        if queue is None:
            await self._ensure_checkpoint_writer()
            queue = self._checkpoint_queue
        assert queue is not None
        if write.coalesce:
            coalesce_key = (write.thread_id, write.key)
            self._checkpoint_coalesce_buffer[coalesce_key] = write
            if coalesce_key in self._checkpoint_coalesce_keys:
                return
            self._checkpoint_coalesce_keys.add(coalesce_key)
            item: _CheckpointWrite | _CheckpointCoalesceToken = _CheckpointCoalesceToken(
                thread_id=write.thread_id,
                key=write.key,
            )
        else:
            item = write
        self._checkpoint_pending_count += 1
        self._checkpoint_pending_event.clear()
        await queue.put(item)

    async def _checkpoint_writer_loop(self) -> None:
        queue = self._checkpoint_queue
        if queue is None:
            return
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            write: _CheckpointWrite | None
            if isinstance(item, _CheckpointCoalesceToken):
                coalesce_key = (item.thread_id, item.key)
                write = self._checkpoint_coalesce_buffer.pop(coalesce_key, None)
                self._checkpoint_coalesce_keys.discard(coalesce_key)
            else:
                write = item
            try:
                if write is not None:
                    memory = await self._ensure_memory_store()
                    await self._write_checkpoint_now(
                        memory=memory,
                        thread_id=write.thread_id,
                        key=write.key,
                        value=write.value,
                    )
            finally:
                self._checkpoint_pending_count = max(0, self._checkpoint_pending_count - 1)
                if self._checkpoint_pending_count == 0:
                    self._checkpoint_pending_event.set()
                queue.task_done()

    async def _flush_checkpoint_writes(self) -> None:
        if not self.config.checkpoint_async_writes:
            return
        timeout = max(0.0, float(self.config.checkpoint_flush_timeout_s))
        await asyncio.wait_for(self._checkpoint_pending_event.wait(), timeout=timeout)

    async def _stop_checkpoint_writer(self) -> None:
        if self._checkpoint_queue is None:
            return
        if self._checkpoint_writer_task is None:
            return
        if not self._checkpoint_writer_task.done():
            await self._checkpoint_queue.put(None)
            await asyncio.gather(self._checkpoint_writer_task, return_exceptions=True)
        self._checkpoint_writer_task = None

    def _build_llm_candidates(
        self,
        *,
        primary,
        fallback_chain: list[str],
        resolver,
    ) -> list[Any]:
        """
        Build ordered unique list of primary + fallback LLM candidates.

        Args:
            primary: Primary resolved model candidate.
            fallback_chain: Fallback model strings.
            resolver: Optional model resolver override.

        Returns:
            Ordered list of unique resolved candidates.
        """
        candidates = [primary]
        seen = {(primary.adapter, primary.normalized_model)}
        for model in fallback_chain:
            if not isinstance(model, str) or not model.strip():
                continue
            resolved = resolve_model_to_llm(model, resolver=resolver)
            key = (resolved.adapter, resolved.normalized_model)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(resolved)
        return candidates

    def _apply_llm_failure_policy(self, policy: str) -> str:
        """
        Normalize LLM failure policy to runtime action.

        Args:
            policy: Configured failure policy string.

        Returns:
            Runtime action (`degrade` or `fail`).
        """
        if policy in {"retry_then_degrade"}:
            return "degrade"
        if policy in {
            "continue",
            "continue_with_error",
            "retry_then_continue",
            "skip_action",
        }:
            return "degrade"
        return "fail"

    def _apply_tool_failure_policy(self, policy: str) -> str:
        """
        Normalize tool failure policy to runtime action.

        Args:
            policy: Configured failure policy string.

        Returns:
            Runtime action (`degrade`, `fail`, or `continue`).
        """
        if policy == "retry_then_degrade":
            return "degrade"
        if policy in {"retry_then_fail", "fail_fast", "fail_run"}:
            return "fail"
        return "continue"

    def _apply_subagent_failure_policy(self, policy: str) -> str:
        """
        Normalize subagent failure policy to runtime action.

        Args:
            policy: Configured failure policy string.

        Returns:
            Runtime action (`degrade`, `fail`, or `continue`).
        """
        if policy == "retry_then_degrade":
            return "degrade"
        if policy in {"retry_then_fail", "fail_fast", "fail_run"}:
            return "fail"
        return "continue"

    def _apply_approval_denial_policy(self, policy: str) -> str:
        """
        Normalize approval denial policy to runtime action.

        Args:
            policy: Configured failure policy string.

        Returns:
            Runtime action (`degrade`, `fail`, or `continue`).
        """
        if policy == "retry_then_degrade":
            return "degrade"
        if policy in {"retry_then_fail", "fail_fast", "fail_run"}:
            return "fail"
        return "continue"

    def _resolve_subagent_parallel(
        self,
        *,
        agent_parallelism_mode: str,
        router_parallel: bool,
    ) -> bool:
        """
        Resolve effective subagent parallelism mode.

        Args:
            agent_parallelism_mode: Agent-level mode (`single`, `parallel`,
                `configurable`).
            router_parallel: Router-provided parallelism hint.

        Returns:
            `True` when subagents should execute in parallel.
        """
        if agent_parallelism_mode == "single":
            return False
        if agent_parallelism_mode == "parallel":
            return True
        return bool(router_parallel)

    def _build_subagent_context(
        self,
        *,
        context: dict[str, Any],
        inherit_keys: list[str],
    ) -> dict[str, Any]:
        """
        Build subagent context from inherited key allowlist.

        Args:
            context: Parent context.
            inherit_keys: Keys allowed to flow into subagent.

        Returns:
            Filtered context dictionary for child agent.
        """
        if not inherit_keys:
            return {}
        out: dict[str, Any] = {}
        for key in inherit_keys:
            if key in context:
                out[key] = context[key]
        return out

    def _accumulate_cost(self, current_cost: float, response) -> float:
        """
        Add response cost (if available) to current aggregate cost.

        Args:
            current_cost: Existing aggregated run cost.
            response: LLM response object with optional raw usage/cost payload.

        Returns:
            Updated aggregated cost.
        """
        raw = response.raw if isinstance(getattr(response, "raw", None), dict) else {}

        direct = raw.get("total_cost_usd")
        if isinstance(direct, (float, int)):
            return current_cost + float(direct)

        usage = raw.get("usage")
        if isinstance(usage, dict):
            nested = usage.get("total_cost_usd")
            if isinstance(nested, (float, int)):
                return current_cost + float(nested)
        return current_cost

    async def _resolve_effect_replay_result(
        self,
        *,
        memory: MemoryStore,
        thread_id: str,
        run_id: str,
        step: int,
        tool_call_id: str | None,
        tool_name: str,
        call_args: dict[str, Any],
    ) -> ToolResult[Any] | None:
        """
        Resolve replayable tool result from effect journal/checkpoint state.

        Args:
            memory: Memory backend.
            thread_id: Thread identifier.
            run_id: Run identifier.
            step: Current step index.
            tool_call_id: Tool-call identifier.
            tool_name: Tool name.
            call_args: Tool arguments.

        Returns:
            Replayed successful tool result when available; otherwise `None`.

        Raises:
            AgentCheckpointCorruptionError: If input hash does not match stored
                effect record.
        """
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return None

        key = effect_state_key(run_id, step, tool_call_id)
        row = self._effect_journal.get(run_id, step, tool_call_id)
        if row is None:
            loaded = await memory.get_state(thread_id, key)
            if isinstance(loaded, dict):
                row = dict(loaded)
                in_hash = row.get("input_hash")
                out_hash = row.get("output_hash")
                output = row.get("output")
                success = bool(row.get("success", False))
                if isinstance(in_hash, str) and isinstance(out_hash, str):
                    self._effect_journal.put(
                        run_id,
                        step,
                        tool_call_id,
                        in_hash,
                        out_hash,
                        output=output
                        if isinstance(
                            output, (dict, list, str, int, float, bool, type(None))
                        )
                        else None,
                        success=success,
                    )

        if not isinstance(row, dict):
            return None

        expected_input_hash = json_hash({"tool_name": tool_name, "args": call_args})
        actual_input_hash = row.get("input_hash")
        if (
            isinstance(actual_input_hash, str)
            and actual_input_hash != expected_input_hash
        ):
            raise AgentCheckpointCorruptionError(
                f"Effect journal integrity conflict for tool_call_id={tool_call_id}"
            )
        if bool(row.get("success", False)):
            return ToolResult(
                output=row.get("output"),
                success=True,
            )
        return None

    async def _persist_effect_result(
        self,
        *,
        memory: MemoryStore,
        thread_id: str,
        run_id: str,
        step: int,
        tool_call_id: str,
        input_hash: str,
        output_hash: str,
        output: Any,
        success: bool,
    ) -> None:
        """
        Persist tool effect journal row for idempotent replay.

        Args:
            memory: Memory backend.
            thread_id: Thread identifier.
            run_id: Run identifier.
            step: Current step.
            tool_call_id: Tool-call id.
            input_hash: Deterministic hash of tool input payload.
            output_hash: Deterministic hash of tool output payload.
            output: Raw output payload.
            success: Whether tool execution succeeded.
        """
        output_json = json_value_from_tool_result(output)
        self._effect_journal.put(
            run_id,
            step,
            tool_call_id,
            input_hash,
            output_hash,
            output=output_json,
            success=success,
        )
        row = {
            "input_hash": input_hash,
            "output_hash": output_hash,
            "output": output_json,
            "success": success,
        }
        await memory.put_state(
            thread_id,
            effect_state_key(run_id, step, tool_call_id),
            row,  # type: ignore[arg-type]
        )

    async def _chat_with_interrupt_support(
        self,
        *,
        handle: _RunHandle,
        llm: Any,
        req: LLMRequest,
        on_text_delta: StreamTextDeltaSink | None = None,
    ):
        """
        Execute chat call with optional provider interrupt wiring.

        Args:
            handle: Active run handle.
            llm: LLM adapter instance.
            req: Normalized LLM request.

        Returns:
            LLM response object.

        Raises:
            AgentInterruptedError: If run is interrupted during streaming call.
            AgentCancelledError: If run is cancelled before response completion.
        """
        if llm.capabilities.streaming and (
            llm.capabilities.interrupt or on_text_delta is not None
        ):
            stream_handle = await llm.chat_stream_handle(req)
            if llm.capabilities.interrupt:
                handle.set_interrupt_callback(stream_handle.interrupt)
            else:
                handle.set_interrupt_callback(None)
            try:
                if on_text_delta is not None:
                    async for event in stream_handle.events:
                        if isinstance(event, StreamTextDeltaEvent) and event.delta:
                            await on_text_delta(event.delta)
                        if isinstance(event, StreamCompletedEvent):
                            break
                response = await stream_handle.await_result()
            finally:
                handle.set_interrupt_callback(None)
            if response is None:
                if handle.is_interrupt_requested():
                    raise AgentInterruptedError(
                        "Run interrupted during LLM streaming call"
                    )
                raise AgentCancelledError("Run cancelled during LLM streaming call")
            return response

        handle.set_interrupt_callback(None)
        return await llm.chat(req)

    async def _persist_runtime_snapshot(
        self,
        *,
        memory: MemoryStore,
        thread_id: str,
        run_id: str,
        step: int,
        state: AgentState,
        context: dict[str, Any],
        messages: list[Message],
        llm_calls: int,
        tool_calls: int,
        started_at_s: float,
        usage: UsageAggregate,
        total_cost_usd: float,
        session_token: str | None,
        checkpoint_token: str | None,
        requested_model: str | None,
        normalized_model: str | None,
        provider_adapter: str | None,
        tool_execs: list[ToolExecutionRecord],
        sub_execs: list[SubagentExecutionRecord],
        skill_reads: list[SkillReadRecord],
        skill_cmd_execs: list[CommandExecutionRecord],
        final_text: str,
        final_structured: dict[str, Any] | None,
        pending_llm_response: LLMResponse | None,
        final_response: LLMResponse | None,
        replayed_effect_count: int,
    ) -> None:
        """
        Persist checkpoint payload containing full runtime snapshot.

        Args:
            memory: Memory backend.
            thread_id: Thread identifier.
            run_id: Run identifier.
            step: Current step.
            state: Current run state.
            context: Runtime context snapshot.
            messages: Transcript messages.
            llm_calls: LLM call counter.
            tool_calls: Tool call counter.
            started_at_s: Run start timestamp.
            usage: Aggregated usage snapshot.
            total_cost_usd: Aggregated cost.
            session_token: Optional provider session token.
            checkpoint_token: Optional provider checkpoint token.
            requested_model: Requested model string.
            normalized_model: Effective model string.
            provider_adapter: Provider adapter id.
            tool_execs: Tool execution records.
            sub_execs: Subagent execution records.
            skill_reads: Skill read records.
            skill_cmd_execs: Skill command records.
            final_text: Current final text buffer.
            final_structured: Current structured payload.
            pending_llm_response: In-flight LLM response for replay.
            final_response: Terminal LLM response when available.
            replayed_effect_count: Number of replayed tool effects.
        """
        runtime_payload = {
            "thread_id": thread_id,
            "step": step,
            "state": state,
            "context": {
                str(k): json_value_from_tool_result(v) for k, v in context.items()
            },
            "messages": self._serialize_messages(messages),
            "llm_calls": llm_calls,
            "tool_calls": tool_calls,
            "started_at_s": started_at_s,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            },
            "total_cost_usd": total_cost_usd,
            "session_token": session_token,
            "checkpoint_token": checkpoint_token,
            "requested_model": requested_model,
            "normalized_model": normalized_model,
            "provider_adapter": provider_adapter,
            "tool_executions": self._serialize_tool_records(tool_execs),
            "subagent_executions": self._serialize_subagent_records(sub_execs),
            "skill_reads": self._serialize_skill_reads(skill_reads),
            "skill_command_executions": self._serialize_command_records(
                skill_cmd_execs
            ),
            "final_text": final_text,
            "final_structured": (
                {
                    str(k): json_value_from_tool_result(v)
                    for k, v in final_structured.items()
                }
                if isinstance(final_structured, dict)
                else None
            ),
            "pending_llm_response": self._serialize_llm_response(pending_llm_response),
            "final_response": self._serialize_llm_response(final_response),
            "replayed_effect_count": replayed_effect_count,
        }
        await self._persist_checkpoint(
            memory=memory,
            thread_id=thread_id,
            run_id=run_id,
            step=step,
            phase="runtime_state",
            payload=runtime_payload,
        )

    def _restore_runtime_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """
        Extract runtime payload from normalized checkpoint snapshot.

        Args:
            snapshot: Normalized checkpoint record.

        Returns:
            Runtime payload dictionary.

        Raises:
            AgentCheckpointCorruptionError: If payload is missing/invalid.
        """
        payload = snapshot.get("payload")
        if not isinstance(payload, dict):
            raise AgentCheckpointCorruptionError("Checkpoint payload is invalid")
        return payload

    def _new_id(self, prefix: str) -> str:
        """
        Generate unique id via runner module indirection.

        Args:
            prefix: Id prefix.

        Returns:
            Generated id string.
        """
        return new_id(prefix)

    def _create_memory_store_from_env(self):
        """
        Create memory store from environment configuration.

        Returns:
            Memory store instance selected by environment.
        """
        return create_memory_store_from_env()

    async def _load_latest_runtime_snapshot(
        self,
        *,
        memory: MemoryStore,
        thread_id: str,
        run_id: str,
        latest: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Load most recent runtime_state checkpoint for resume.

        Args:
            memory: Memory backend.
            thread_id: Thread identifier.
            run_id: Run identifier.
            latest: Latest checkpoint pointer record.

        Returns:
            Normalized runtime_state checkpoint.

        Raises:
            AgentCheckpointCorruptionError: If no runtime snapshot can be found.
        """
        normalized_latest = self._normalize_checkpoint_record(latest)
        phase = normalized_latest.get("phase")
        if phase == "runtime_state":
            return normalized_latest

        step_value = normalized_latest.get("step")
        max_step = step_value if isinstance(step_value, int) and step_value >= 0 else 0
        for step in range(max_step, -1, -1):
            state = await memory.get_state(
                thread_id, checkpoint_state_key(run_id, step, "runtime_state")
            )
            if isinstance(state, dict):
                return self._normalize_checkpoint_record(state)

        payload = normalized_latest.get("payload")
        if isinstance(payload, dict) and "messages" in payload:
            return normalized_latest

        raise AgentCheckpointCorruptionError(
            f"No runtime_state checkpoint found for run_id={run_id} thread_id={thread_id}"
        )

    def _normalize_checkpoint_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """
        Migrate and validate checkpoint record.

        Args:
            record: Raw checkpoint record from memory store.

        Returns:
            Migrated and schema-validated checkpoint record.

        Raises:
            AgentCheckpointCorruptionError: If migration/validation fails.
        """
        try:
            migrated = migrate_checkpoint_record(record).migrated
        except Exception as e:
            raise AgentCheckpointCorruptionError(
                f"Checkpoint migration failed: {e}"
            ) from e
        version = migrated.get("schema_version")
        check = check_checkpoint_schema_version(
            version if isinstance(version, str) else None
        )
        if not check.compatible:
            raise AgentCheckpointCorruptionError(check.message)
        return migrated

    def _serialize_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """
        Serialize message objects into checkpoint-friendly dictionaries.

        Args:
            messages: Message list.

        Returns:
            JSON-like serialized message payload.
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.content
            if isinstance(content, list):
                serialized_content = []
                for part in content:
                    if isinstance(part, dict):
                        serialized_content.append(
                            {
                                str(k): json_value_from_tool_result(v)
                                for k, v in part.items()
                            }
                        )
                    else:
                        serialized_content.append(json_value_from_tool_result(part))
                content_value = serialized_content
            else:
                content_value = content
            row: dict[str, Any] = {
                "role": msg.role,
                "content": json_value_from_tool_result(content_value),
            }
            if msg.name is not None:
                row["name"] = msg.name
            out.append(row)
        return out

    def _deserialize_messages(self, value: Any) -> list[Message]:
        """
        Deserialize messages from checkpoint payload.

        Args:
            value: Raw serialized message payload.

        Returns:
            List of normalized `Message` objects.
        """
        if not isinstance(value, list):
            return []
        out: list[Message] = []
        for row in value:
            if not isinstance(row, dict):
                continue
            role = row.get("role")
            content = row.get("content")
            name = row.get("name")
            if isinstance(role, str) and (isinstance(content, (str, list))):
                out.append(
                    Message(
                        role=role,  # type: ignore[arg-type]
                        content=content,
                        name=name if isinstance(name, str) else None,
                    )
                )
        return out

    def _serialize_llm_response(
        self, value: LLMResponse | None
    ) -> dict[str, Any] | None:
        """
        Serialize optional LLM response for checkpoint persistence.

        Args:
            value: LLM response.

        Returns:
            Serialized response dictionary or `None`.
        """
        if value is None:
            return None
        return {
            "text": value.text,
            "request_id": value.request_id,
            "provider_request_id": value.provider_request_id,
            "session_token": value.session_token,
            "checkpoint_token": value.checkpoint_token,
            "structured_response": value.structured_response,
            "tool_calls": [
                {
                    "id": call.id,
                    "tool_name": call.tool_name,
                    "arguments": call.arguments,
                }
                for call in value.tool_calls
            ],
            "finish_reason": value.finish_reason,
            "usage": {
                "input_tokens": value.usage.input_tokens,
                "output_tokens": value.usage.output_tokens,
                "total_tokens": value.usage.total_tokens,
            },
            "raw": json_value_from_tool_result(value.raw),
            "model": value.model,
        }

    def _deserialize_llm_response(self, value: Any) -> LLMResponse | None:
        """
        Deserialize optional LLM response from checkpoint payload.

        Args:
            value: Serialized response payload.

        Returns:
            Reconstructed `LLMResponse` or `None`.
        """
        if not isinstance(value, dict):
            return None
        tool_calls_value = value.get("tool_calls")
        tool_calls: list[ToolCall] = []
        if isinstance(tool_calls_value, list):
            for row in tool_calls_value:
                if not isinstance(row, dict):
                    continue
                args = row.get("arguments")
                tool_calls.append(
                    ToolCall(
                        id=row.get("id") if isinstance(row.get("id"), str) else None,
                        tool_name=row.get("tool_name")
                        if isinstance(row.get("tool_name"), str)
                        else "",
                        arguments=args if isinstance(args, dict) else {},
                    )
                )

        usage_row = value.get("usage")
        usage = Usage()
        if isinstance(usage_row, dict):
            usage = Usage(
                input_tokens=usage_row.get("input_tokens")
                if isinstance(usage_row.get("input_tokens"), int)
                else None,
                output_tokens=usage_row.get("output_tokens")
                if isinstance(usage_row.get("output_tokens"), int)
                else None,
                total_tokens=usage_row.get("total_tokens")
                if isinstance(usage_row.get("total_tokens"), int)
                else None,
            )

        raw = value.get("raw")
        return LLMResponse(
            text=value.get("text") if isinstance(value.get("text"), str) else "",
            request_id=value.get("request_id")
            if isinstance(value.get("request_id"), str)
            else None,
            provider_request_id=value.get("provider_request_id")
            if isinstance(value.get("provider_request_id"), str)
            else None,
            session_token=value.get("session_token")
            if isinstance(value.get("session_token"), str)
            else None,
            checkpoint_token=value.get("checkpoint_token")
            if isinstance(value.get("checkpoint_token"), str)
            else None,
            structured_response=value.get("structured_response")
            if isinstance(value.get("structured_response"), dict)
            else None,
            tool_calls=tool_calls,
            finish_reason=value.get("finish_reason")
            if isinstance(value.get("finish_reason"), str)
            else None,
            usage=usage,
            raw=raw if isinstance(raw, dict) else {},
            model=value.get("model") if isinstance(value.get("model"), str) else None,
        )

    def _serialize_tool_records(
        self, rows: list[ToolExecutionRecord]
    ) -> list[dict[str, Any]]:
        """
        Serialize tool execution records.

        Args:
            rows: Tool execution record list.

        Returns:
            Serialized record list.
        """
        return [
            {
                "tool_name": row.tool_name,
                "tool_call_id": row.tool_call_id,
                "success": row.success,
                "output": row.output,
                "error": row.error,
                "latency_ms": row.latency_ms,
                "agent_name": row.agent_name,
                "agent_depth": row.agent_depth,
                "agent_path": row.agent_path,
            }
            for row in rows
        ]

    def _deserialize_tool_records(self, value: Any) -> list[ToolExecutionRecord]:
        """
        Deserialize tool execution records.

        Args:
            value: Serialized tool record payload.

        Returns:
            Parsed tool execution records.
        """
        if not isinstance(value, list):
            return []
        out: list[ToolExecutionRecord] = []
        for row in value:
            if not isinstance(row, dict):
                continue
            name = row.get("tool_name")
            if not isinstance(name, str):
                continue
            out.append(
                ToolExecutionRecord(
                    tool_name=name,
                    tool_call_id=row.get("tool_call_id")
                    if isinstance(row.get("tool_call_id"), str)
                    else None,
                    success=bool(row.get("success", False)),
                    output=row.get("output"),
                    error=row.get("error")
                    if isinstance(row.get("error"), str)
                    else None,
                    latency_ms=row.get("latency_ms")
                    if isinstance(row.get("latency_ms"), (float, int))
                    else None,
                    agent_name=row.get("agent_name")
                    if isinstance(row.get("agent_name"), str)
                    else None,
                    agent_depth=int(row.get("agent_depth"))
                    if isinstance(row.get("agent_depth"), int)
                    else None,
                    agent_path=row.get("agent_path")
                    if isinstance(row.get("agent_path"), str)
                    else None,
                )
            )
        return out

    def _serialize_subagent_records(
        self,
        rows: list[SubagentExecutionRecord],
    ) -> list[dict[str, Any]]:
        """
        Serialize subagent execution records.

        Args:
            rows: Subagent execution records.

        Returns:
            Serialized record list.
        """
        return [
            {
                "subagent_name": row.subagent_name,
                "success": row.success,
                "output_text": row.output_text,
                "error": row.error,
                "latency_ms": row.latency_ms,
            }
            for row in rows
        ]

    def _deserialize_subagent_records(
        self, value: Any
    ) -> list[SubagentExecutionRecord]:
        """
        Deserialize subagent execution records.

        Args:
            value: Serialized subagent record payload.

        Returns:
            Parsed subagent execution records.
        """
        if not isinstance(value, list):
            return []
        out: list[SubagentExecutionRecord] = []
        for row in value:
            if not isinstance(row, dict):
                continue
            name = row.get("subagent_name")
            if not isinstance(name, str):
                continue
            out.append(
                SubagentExecutionRecord(
                    subagent_name=name,
                    success=bool(row.get("success", False)),
                    output_text=row.get("output_text")
                    if isinstance(row.get("output_text"), str)
                    else None,
                    error=row.get("error")
                    if isinstance(row.get("error"), str)
                    else None,
                    latency_ms=row.get("latency_ms")
                    if isinstance(row.get("latency_ms"), (float, int))
                    else None,
                )
            )
        return out

    def _serialize_skill_reads(
        self, rows: list[SkillReadRecord]
    ) -> list[dict[str, Any]]:
        """
        Serialize skill read records.

        Args:
            rows: Skill read records.

        Returns:
            Serialized record list.
        """
        return [
            {
                "skill_name": row.skill_name,
                "path": row.path,
                "checksum": row.checksum,
            }
            for row in rows
        ]

    def _deserialize_skill_reads(self, value: Any) -> list[SkillReadRecord]:
        """
        Deserialize skill read records.

        Args:
            value: Serialized skill-read payload.

        Returns:
            Parsed skill read records.
        """
        if not isinstance(value, list):
            return []
        out: list[SkillReadRecord] = []
        for row in value:
            if not isinstance(row, dict):
                continue
            skill_name = row.get("skill_name")
            path = row.get("path")
            if not isinstance(skill_name, str) or not isinstance(path, str):
                continue
            out.append(
                SkillReadRecord(
                    skill_name=skill_name,
                    path=path,
                    checksum=row.get("checksum")
                    if isinstance(row.get("checksum"), str)
                    else None,
                )
            )
        return out

    def _serialize_command_records(
        self,
        rows: list[CommandExecutionRecord],
    ) -> list[dict[str, Any]]:
        """
        Serialize command execution records.

        Args:
            rows: Command execution records.

        Returns:
            Serialized record list.
        """
        return [
            {
                "command": list(row.command),
                "exit_code": row.exit_code,
                "stdout": row.stdout,
                "stderr": row.stderr,
                "denied": row.denied,
            }
            for row in rows
        ]

    def _deserialize_command_records(self, value: Any) -> list[CommandExecutionRecord]:
        """
        Deserialize command execution records.

        Args:
            value: Serialized command record payload.

        Returns:
            Parsed command execution records.
        """
        if not isinstance(value, list):
            return []
        out: list[CommandExecutionRecord] = []
        for row in value:
            if not isinstance(row, dict):
                continue
            cmd = row.get("command")
            if not isinstance(cmd, list):
                continue
            out.append(
                CommandExecutionRecord(
                    command=[str(item) for item in cmd],
                    exit_code=int(row.get("exit_code", 1)),
                    stdout=row.get("stdout")
                    if isinstance(row.get("stdout"), str)
                    else "",
                    stderr=row.get("stderr")
                    if isinstance(row.get("stderr"), str)
                    else "",
                    denied=bool(row.get("denied", False)),
                )
            )
        return out

    def _build_terminal_result(
        self,
        *,
        run_id: str,
        thread_id: str,
        state: AgentState,
        final_text: str,
        requested_model: str | None,
        normalized_model: str | None,
        provider_adapter: str | None,
        final_structured: dict[str, Any] | None,
        llm_response: LLMResponse | None,
        tool_execs: list[ToolExecutionRecord],
        sub_execs: list[SubagentExecutionRecord],
        skills: list[Any],
        skill_reads: list[SkillReadRecord],
        skill_cmd_execs: list[CommandExecutionRecord],
        usage: UsageAggregate,
        total_cost_usd: float,
        session_token: str | None,
        checkpoint_token: str | None,
        step: int,
        llm_calls: int,
        tool_calls: int,
        started_at_s: float,
        replayed_effect_count: int,
    ) -> AgentResult:
        """
        Build terminal `AgentResult` from accumulated runtime state.

        Args:
            run_id: Run identifier.
            thread_id: Thread identifier.
            state: Terminal state.
            final_text: Final textual assistant output.
            requested_model: Requested model string.
            normalized_model: Effective model string.
            provider_adapter: Provider adapter id.
            final_structured: Final structured output payload.
            llm_response: Final LLM response.
            tool_execs: Tool execution records.
            sub_execs: Subagent execution records.
            skills: Resolved skill refs.
            skill_reads: Skill read records.
            skill_cmd_execs: Skill command execution records.
            usage: Aggregated usage.
            total_cost_usd: Aggregated cost.
            session_token: Provider session token.
            checkpoint_token: Provider checkpoint token.
            step: Final step count.
            llm_calls: LLM call count.
            tool_calls: Tool call count.
            started_at_s: Run start timestamp.
            replayed_effect_count: Replayed effect count.

        Returns:
            Normalized terminal agent result.
        """
        return AgentResult(
            run_id=run_id,
            thread_id=thread_id,
            state=state,
            final_text=final_text,
            requested_model=requested_model,
            normalized_model=normalized_model,
            provider_adapter=provider_adapter,
            final_structured=final_structured,
            llm_response=llm_response,
            tool_executions=list(tool_execs),
            subagent_executions=list(sub_execs),
            skills_used=[s.name for s in skills if hasattr(s, "name")],
            skill_reads=list(skill_reads),
            skill_command_executions=list(skill_cmd_execs),
            usage_aggregate=usage,
            total_cost_usd=total_cost_usd if total_cost_usd > 0 else None,
            session_token=session_token,
            checkpoint_token=checkpoint_token,
            state_snapshot=state_snapshot(
                state=state,
                step=step,
                llm_calls=llm_calls,
                tool_calls=tool_calls,
                started_at_s=started_at_s,
                requested_model=requested_model,
                normalized_model=normalized_model,
                provider_adapter=provider_adapter,
                total_cost_usd=total_cost_usd if total_cost_usd > 0 else None,
                replayed_effect_count=replayed_effect_count,
            ),
        )

    async def _persist_terminal_checkpoint(
        self,
        *,
        memory: Any,
        thread_id: str,
        run_id: str,
        step: int,
        state: AgentState,
        message: str,
        final_text: str,
        requested_model: str | None,
        normalized_model: str | None,
        provider_adapter: str | None,
        final_structured: dict[str, Any] | None,
        llm_response: LLMResponse | None,
        tool_execs: list[ToolExecutionRecord],
        sub_execs: list[SubagentExecutionRecord],
        skills: list[Any],
        skill_reads: list[SkillReadRecord],
        skill_cmd_execs: list[CommandExecutionRecord],
        usage: UsageAggregate,
        total_cost_usd: float,
        session_token: str | None,
        checkpoint_token: str | None,
        llm_calls: int,
        tool_calls: int,
        started_at_s: float,
        replayed_effect_count: int,
    ) -> None:
        """Build, serialize, and persist a terminal checkpoint in one call."""
        result = self._build_terminal_result(
            run_id=run_id,
            thread_id=thread_id,
            state=state,
            final_text=final_text,
            requested_model=requested_model,
            normalized_model=normalized_model,
            provider_adapter=provider_adapter,
            final_structured=final_structured,
            llm_response=llm_response,
            tool_execs=tool_execs,
            sub_execs=sub_execs,
            skills=skills,
            skill_reads=skill_reads,
            skill_cmd_execs=skill_cmd_execs,
            usage=usage,
            total_cost_usd=total_cost_usd,
            session_token=session_token,
            checkpoint_token=checkpoint_token,
            step=step,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            started_at_s=started_at_s,
            replayed_effect_count=replayed_effect_count,
        )
        await self._persist_checkpoint(
            memory=memory,
            thread_id=thread_id,
            run_id=run_id,
            step=step,
            phase="run_terminal",
            payload={
                "state": state,
                "message": message,
                "terminal_result": self._serialize_agent_result(result),
            },
        )

    def _serialize_agent_result(self, result: AgentResult) -> dict[str, Any]:
        """
        Serialize `AgentResult` for checkpoint storage.

        Args:
            result: Agent result.

        Returns:
            Serialized result payload.
        """
        return {
            "run_id": result.run_id,
            "thread_id": result.thread_id,
            "state": result.state,
            "final_text": result.final_text,
            "requested_model": result.requested_model,
            "normalized_model": result.normalized_model,
            "provider_adapter": result.provider_adapter,
            "final_structured": result.final_structured,
            "llm_response": self._serialize_llm_response(result.llm_response),
            "tool_executions": self._serialize_tool_records(result.tool_executions),
            "subagent_executions": self._serialize_subagent_records(
                result.subagent_executions
            ),
            "skills_used": list(result.skills_used),
            "skill_reads": self._serialize_skill_reads(result.skill_reads),
            "skill_command_executions": self._serialize_command_records(
                result.skill_command_executions
            ),
            "usage_aggregate": {
                "input_tokens": result.usage_aggregate.input_tokens,
                "output_tokens": result.usage_aggregate.output_tokens,
                "total_tokens": result.usage_aggregate.total_tokens,
            },
            "total_cost_usd": result.total_cost_usd,
            "session_token": result.session_token,
            "checkpoint_token": result.checkpoint_token,
            "state_snapshot": result.state_snapshot,
        }

    def _deserialize_agent_result(self, value: dict[str, Any]) -> AgentResult:
        """
        Deserialize `AgentResult` from checkpoint payload.

        Args:
            value: Serialized result payload.

        Returns:
            Reconstructed agent result.
        """
        usage_row = value.get("usage_aggregate")
        usage = UsageAggregate()
        if isinstance(usage_row, dict):
            usage = UsageAggregate(
                input_tokens=int(usage_row.get("input_tokens", 0)),
                output_tokens=int(usage_row.get("output_tokens", 0)),
                total_tokens=int(usage_row.get("total_tokens", 0)),
            )
        return AgentResult(
            run_id=value.get("run_id") if isinstance(value.get("run_id"), str) else "",
            thread_id=value.get("thread_id")
            if isinstance(value.get("thread_id"), str)
            else "",
            state=value.get("state")
            if isinstance(value.get("state"), str)
            else "failed",
            final_text=value.get("final_text")
            if isinstance(value.get("final_text"), str)
            else "",
            requested_model=value.get("requested_model")
            if isinstance(value.get("requested_model"), str)
            else None,
            normalized_model=value.get("normalized_model")
            if isinstance(value.get("normalized_model"), str)
            else None,
            provider_adapter=value.get("provider_adapter")
            if isinstance(value.get("provider_adapter"), str)
            else None,
            final_structured=value.get("final_structured")
            if isinstance(value.get("final_structured"), dict)
            else None,
            llm_response=self._deserialize_llm_response(value.get("llm_response")),
            tool_executions=self._deserialize_tool_records(
                value.get("tool_executions")
            ),
            subagent_executions=self._deserialize_subagent_records(
                value.get("subagent_executions")
            ),
            skills_used=(
                [str(item) for item in value.get("skills_used")]
                if isinstance(value.get("skills_used"), list)
                else []
            ),
            skill_reads=self._deserialize_skill_reads(value.get("skill_reads")),
            skill_command_executions=self._deserialize_command_records(
                value.get("skill_command_executions")
            ),
            usage_aggregate=usage,
            total_cost_usd=value.get("total_cost_usd")
            if isinstance(value.get("total_cost_usd"), (int, float))
            else None,
            session_token=value.get("session_token")
            if isinstance(value.get("session_token"), str)
            else None,
            checkpoint_token=value.get("checkpoint_token")
            if isinstance(value.get("checkpoint_token"), str)
            else None,
            state_snapshot=value.get("state_snapshot")
            if isinstance(value.get("state_snapshot"), dict)
            else {},
        )

    def _maybe_str(self, value: Any) -> str | None:
        """
        Return string value when input is a string.

        Args:
            value: Arbitrary value.

        Returns:
            String value or `None`.
        """
        return value if isinstance(value, str) else None

    def _telemetry_start_span(
        self,
        name: str,
        *,
        attributes: dict[str, JSONValue] | None = None,
    ) -> TelemetrySpan | None:
        """
        Safely start telemetry span.

        Args:
            name: Span name.
            attributes: Optional span attributes.

        Returns:
            Span object or `None` when telemetry fails.
        """
        try:
            return self._telemetry.start_span(name, attributes=attributes)
        except Exception:
            return None

    def _telemetry_end_span(
        self,
        span: TelemetrySpan | None,
        *,
        status: str,
        error: str | None = None,
        attributes: dict[str, JSONValue] | None = None,
    ) -> None:
        """
        Safely end telemetry span.

        Args:
            span: Span object.
            status: Terminal status string.
            error: Optional error detail.
            attributes: Optional final attributes.
        """
        try:
            self._telemetry.end_span(
                span,
                status=status,
                error=error,
                attributes=attributes,
            )
        except Exception:
            return None

    def _telemetry_counter(
        self,
        name: str,
        *,
        value: int = 1,
        attributes: dict[str, JSONValue] | None = None,
    ) -> None:
        """
        Safely record telemetry counter.

        Args:
            name: Counter name.
            value: Increment value.
            attributes: Optional attributes.
        """
        try:
            self._telemetry.increment_counter(name, value=value, attributes=attributes)
        except Exception:
            return None

    def _telemetry_histogram(
        self,
        name: str,
        *,
        value: float,
        attributes: dict[str, JSONValue] | None = None,
    ) -> None:
        """
        Safely record telemetry histogram value.

        Args:
            name: Histogram name.
            value: Numeric measurement.
            attributes: Optional attributes.
        """
        try:
            self._telemetry.record_histogram(name, value, attributes=attributes)
        except Exception:
            return None
