"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Core execution loop for the AFK runner.

Lifecycle Overview
==================

The ``RunnerExecutionMixin`` implements the main agent execution loop:

1. **Step Loop** — The runner iterates ``max_steps`` times. Each step:
   a. Sends the conversation history to the LLM adapter.
   b. Receives an ``LLMResponse`` containing text and/or tool calls.
   c. If tool calls are present, dispatches them (see Tool Batching below).
   d. Appends assistant + tool results to the message history.
   e. Checks fail-safe limits (wall time, LLM calls, tool calls, cost).

2. **Tool Batching** — Multiple tool calls from a single LLM response
   are executed concurrently up to ``max_parallel_tools``. Each tool
   invocation goes through:
   - Policy evaluation (``PolicyEngine`` / ``PolicyRole``)
   - Approval flow (when policy defers to human-in-the-loop)
   - Tool execution with timeout enforcement
   - Result serialization back into the conversation

3. **Circuit Breaker** — Consecutive failures (LLM errors, tool errors)
   are tracked per failure-key. After ``breaker_failure_threshold``
   consecutive failures, the breaker opens and subsequent calls fail
   fast until the ``breaker_cooldown_s`` window expires.

4. **Checkpoint Persistence** — At terminal states (completed, failed,
   cancelled, degraded), the runner persists a checkpoint snapshot to
   the memory store, enabling ``Runner.resume()`` to continue from
   the last known good state.

5. **Subagent Routing** — When the agent has subagents configured, the
   router selects targets and the runner recursively executes them,
   respecting ``max_subagent_depth`` for bounded recursion.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from ...agents.core.base import BaseAgent
from ...agents.errors import (
    AgentBudgetExceededError,
    AgentCancelledError,
    AgentCheckpointCorruptionError,
    AgentError,
    AgentExecutionError,
    AgentInterruptedError,
    AgentLoopLimitError,
    SubagentExecutionError,
    SubagentRoutingError,
)
from ...agents.model.resolution import resolve_model_to_llm
from ...agents.lifecycle.runtime import (
    CircuitBreaker,
    build_skill_manifest_prompt,
    json_hash,
    resolve_skills,
    state_snapshot,
)
from ...agents.security import (
    UNTRUSTED_TOOL_PREAMBLE,
    render_untrusted_tool_message,
    trusted_system_channel_header,
)
from ...agents.types import (
    AgentResult,
    AgentRunEvent,
    AgentState,
    CommandExecutionRecord,
    FailSafeConfig,
    PolicyEvent,
    SkillReadRecord,
    SubagentExecutionRecord,
    ToolExecutionRecord,
    UsageAggregate,
    json_value_from_tool_result,
    tool_record_from_result,
)
from ...llms.types import LLMRequest, LLMResponse, Message
from ...mcp import get_mcp_store
from ...observability import contracts as obs_contracts
from ...tools import ToolContext, ToolResult
from ...tools.prebuilts import build_runtime_tools, build_skill_tools
from ...tools.security import (
    apply_tool_output_limits,
    resolve_sandbox_profile,
    validate_tool_args_against_sandbox,
)
from .types import _RunHandle


class RunnerExecutionMixin:
    """Implements the main agent loop, tool orchestration, and recovery flow."""

    async def _execute(
        self,
        handle: _RunHandle,
        agent: BaseAgent,
        *,
        user_message: str | None,
        context: dict[str, Any] | None,
        thread_id: str | None,
        depth: int,
        lineage: tuple[int, ...],
        resume_run_id: str | None = None,
        resume_snapshot: dict[str, Any] | None = None,
    ) -> None:
        """
        Execute one agent run end-to-end and resolve the run handle.

        Args:
            handle: Active run handle that receives events and terminal result.
            agent: Agent definition being executed.
            user_message: Optional initial user message.
            context: Optional run context overlay.
            thread_id: Optional thread identifier.
            depth: Current recursion depth (subagent execution).
            lineage: Current lineage path for nested runs.
            resume_run_id: Existing run id when resuming checkpoints.
            resume_snapshot: Restored runtime snapshot for resume.

        Returns:
            `None`. Terminal state is written into `handle`.
        """
        run_id = resume_run_id or self._new_id("run")
        t_id = thread_id or self._new_id("thread")
        ctx: dict[str, Any] = {**agent.context_defaults, **(context or {})}
        state: AgentState = "pending"
        llm_calls = 0
        tool_calls = 0
        step = 0
        started_at_s = time.time()
        usage = UsageAggregate()
        tool_execs: list[ToolExecutionRecord] = []
        sub_execs: list[SubagentExecutionRecord] = []
        skill_reads: list[SkillReadRecord] = []
        skill_cmd_execs: list[CommandExecutionRecord] = []
        session_token: str | None = None
        checkpoint_token: str | None = None
        final_text = ""
        final_structured = None
        final_resp = None
        total_cost_usd = 0.0
        requested_model: str | None = None
        normalized_model: str | None = None
        provider_adapter: str | None = None
        messages: list[Message] = []
        pending_llm_response: LLMResponse | None = None
        reuse_current_step = False
        replayed_effect_count = 0
        run_span = self._telemetry_start_span(
            obs_contracts.SPAN_AGENT_RUN,
            attributes={
                "run_id": run_id,
                "thread_id": t_id,
                "agent_name": agent.name,
                "depth": depth,
            },
        )
        run_started_s = time.time()
        run_span_status = "error"
        run_span_error: str | None = None

        if isinstance(resume_snapshot, dict):
            restored = self._restore_runtime_snapshot(resume_snapshot)
            restored_thread_id = restored.get("thread_id")
            if isinstance(restored_thread_id, str) and restored_thread_id:
                t_id = restored_thread_id

            restored_ctx = restored.get("context")
            if isinstance(restored_ctx, dict):
                ctx = {
                    **agent.context_defaults,
                    **restored_ctx,
                    **(context or {}),
                }

            restored_state = restored.get("state")
            if isinstance(restored_state, str):
                state = restored_state  # type: ignore[assignment]
            llm_calls = int(restored.get("llm_calls", llm_calls))
            tool_calls = int(restored.get("tool_calls", tool_calls))
            step = int(restored.get("step", step))
            started_at_s = float(restored.get("started_at_s", started_at_s))
            total_cost_usd = float(restored.get("total_cost_usd", total_cost_usd))
            session_token = (
                self._maybe_str(restored.get("session_token")) or session_token
            )
            checkpoint_token = (
                self._maybe_str(restored.get("checkpoint_token")) or checkpoint_token
            )
            requested_model = (
                self._maybe_str(restored.get("requested_model")) or requested_model
            )
            normalized_model = (
                self._maybe_str(restored.get("normalized_model")) or normalized_model
            )
            provider_adapter = (
                self._maybe_str(restored.get("provider_adapter")) or provider_adapter
            )
            final_text = self._maybe_str(restored.get("final_text")) or final_text
            restored_final_structured = restored.get("final_structured")
            if isinstance(restored_final_structured, dict):
                final_structured = restored_final_structured

            usage_row = restored.get("usage")
            if isinstance(usage_row, dict):
                usage = UsageAggregate(
                    input_tokens=int(usage_row.get("input_tokens", 0)),
                    output_tokens=int(usage_row.get("output_tokens", 0)),
                    total_tokens=int(usage_row.get("total_tokens", 0)),
                )

            tool_execs = self._deserialize_tool_records(restored.get("tool_executions"))
            sub_execs = self._deserialize_subagent_records(
                restored.get("subagent_executions")
            )
            skill_reads = self._deserialize_skill_reads(restored.get("skill_reads"))
            skill_cmd_execs = self._deserialize_command_records(
                restored.get("skill_command_executions")
            )
            messages = self._deserialize_messages(restored.get("messages"))
            pending_llm_response = self._deserialize_llm_response(
                restored.get("pending_llm_response")
            )
            final_resp = self._deserialize_llm_response(restored.get("final_response"))
            if pending_llm_response is not None:
                reuse_current_step = True

        fail_safe: FailSafeConfig = agent.fail_safe
        breaker = CircuitBreaker(fail_safe)
        memory = await self._ensure_memory_store()
        self._active_runs += 1

        try:
            if depth > fail_safe.max_subagent_depth:
                raise SubagentRoutingError(
                    f"Subagent depth exceeded max_subagent_depth={fail_safe.max_subagent_depth}"
                )
            if id(agent) in lineage:
                raise SubagentRoutingError(
                    f"Subagent cycle detected for '{agent.name}'"
                )

            is_resume = isinstance(resume_snapshot, dict)
            state = self._transition_state(state, "running")
            await self._emit(
                handle,
                memory,
                AgentRunEvent(
                    type="run_started",
                    run_id=run_id,
                    thread_id=t_id,
                    state=state,
                    message=(
                        f"Run resumed for agent '{agent.name}'"
                        if is_resume
                        else f"Run started for agent '{agent.name}'"
                    ),
                    data={"agent_name": agent.name, "resumed": is_resume},
                ),
                user_id=self._maybe_str(ctx.get("user_id")),
            )
            await self._persist_checkpoint(
                memory=memory,
                thread_id=t_id,
                run_id=run_id,
                step=0,
                phase="run_started",
                payload={"agent_name": agent.name, "resumed": is_resume},
            )
            if self._memory_fallback_reason:
                await self._emit(
                    handle,
                    memory,
                    AgentRunEvent(
                        type="warning",
                        run_id=run_id,
                        thread_id=t_id,
                        state=state,
                        message=(
                            "Memory store env resolution failed; using in-memory store. "
                            f"reason={self._memory_fallback_reason}"
                        ),
                    ),
                    user_id=self._maybe_str(ctx.get("user_id")),
                )

            resolved = resolve_model_to_llm(
                agent.model,
                resolver=agent.model_resolver,
            )
            llm = resolved.llm
            model_name = resolved.normalized_model
            requested_model = resolved.requested_model
            normalized_model = resolved.normalized_model
            provider_adapter = resolved.adapter

            skills = resolve_skills(
                skill_names=agent.skills,
                skills_dir=agent.skills_dir,
                cwd=Path.cwd(),
            )
            extra_tools = []
            if agent.enable_skill_tools and skills.resolved_skills:
                skill_policy = agent.skill_tool_policy
                if not skill_policy.command_allowlist:
                    from dataclasses import replace

                    skill_policy = replace(
                        skill_policy,
                        command_allowlist=list(
                            self.config.default_allowlisted_commands
                        ),
                    )
                extra_tools.extend(
                    build_skill_tools(
                        skills=skills.resolved_skills,
                        policy=skill_policy,
                    )
                )
            if agent.enable_mcp_tools and agent.mcp_servers:
                try:
                    mcp_store = get_mcp_store()
                    extra_tools.extend(
                        await mcp_store.tools_from_servers(agent.mcp_servers)
                    )
                except Exception as e:
                    raise AgentExecutionError(
                        f"Failed to load MCP tools for agent '{agent.name}': {e}"
                    ) from e
            extra_tools.extend(build_runtime_tools(root_dir=Path.cwd()))
            registry = agent.build_tool_registry(
                extra_tools=extra_tools,
            )
            llm_tools = (
                registry.to_openai_function_tools() if registry.names() else None
            )

            if not messages:
                sys_chunks: list[str] = []
                if self.config.untrusted_tool_preamble:
                    sys_chunks.append(
                        f"{trusted_system_channel_header()}\n{UNTRUSTED_TOOL_PREAMBLE}"
                    )
                base_inst = await agent.resolve_instructions(ctx)
                if base_inst:
                    sys_chunks.append(base_inst)
                skill_manifest = build_skill_manifest_prompt(skills.resolved_skills)
                if skill_manifest:
                    sys_chunks.append(skill_manifest)
                if sys_chunks:
                    messages.append(
                        Message(role="system", content="\n\n".join(sys_chunks))
                    )
                if isinstance(user_message, str) and user_message.strip():
                    messages.append(Message(role="user", content=user_message.strip()))

            while True:
                await handle.wait_if_paused()
                if handle.is_cancel_requested():
                    raise AgentCancelledError("Run cancelled by caller")
                if handle.is_interrupt_requested():
                    raise AgentInterruptedError("Run interrupted by caller")

                if reuse_current_step:
                    reuse_current_step = False
                else:
                    step += 1
                self._enforce_budget(
                    fail_safe=fail_safe,
                    step=step,
                    llm_calls=llm_calls,
                    tool_calls=tool_calls,
                    started_at_s=started_at_s,
                    total_cost_usd=total_cost_usd,
                )
                if step > fail_safe.max_steps:
                    raise AgentLoopLimitError(
                        f"Exceeded max_steps={fail_safe.max_steps} for run {run_id}"
                    )

                await self._emit(
                    handle,
                    memory,
                    AgentRunEvent(
                        type="step_started",
                        run_id=run_id,
                        thread_id=t_id,
                        state=state,
                        step=step,
                    ),
                    user_id=self._maybe_str(ctx.get("user_id")),
                )
                await self._persist_checkpoint(
                    memory=memory,
                    thread_id=t_id,
                    run_id=run_id,
                    step=step,
                    phase="step_started",
                    payload={"state": state, "message_count": len(messages)},
                )
                await self._persist_runtime_snapshot(
                    memory=memory,
                    thread_id=t_id,
                    run_id=run_id,
                    step=step,
                    state=state,
                    context=ctx,
                    messages=messages,
                    llm_calls=llm_calls,
                    tool_calls=tool_calls,
                    started_at_s=started_at_s,
                    usage=usage,
                    total_cost_usd=total_cost_usd,
                    session_token=session_token,
                    checkpoint_token=checkpoint_token,
                    requested_model=requested_model,
                    normalized_model=normalized_model,
                    provider_adapter=provider_adapter,
                    tool_execs=tool_execs,
                    sub_execs=sub_execs,
                    skill_reads=skill_reads,
                    skill_cmd_execs=skill_cmd_execs,
                    final_text=final_text,
                    final_structured=final_structured,
                    pending_llm_response=pending_llm_response,
                    final_response=final_resp,
                    replayed_effect_count=replayed_effect_count,
                )

                if agent.subagent_router and agent.subagents:
                    r_decision = await self._call_router(
                        agent,
                        run_id=run_id,
                        thread_id=t_id,
                        step=step,
                        context=ctx,
                        messages=messages,
                    )
                    if r_decision.targets:
                        if (
                            len(r_decision.targets)
                            > fail_safe.max_subagent_fanout_per_step
                        ):
                            raise SubagentRoutingError(
                                "Subagent fanout exceeded guard "
                                f"({len(r_decision.targets)}>{fail_safe.max_subagent_fanout_per_step})"
                            )
                        await self._persist_checkpoint(
                            memory=memory,
                            thread_id=t_id,
                            run_id=run_id,
                            step=step,
                            phase="pre_subagent_batch",
                            payload={
                                "targets": [str(t) for t in r_decision.targets],
                                "router_parallel": bool(r_decision.parallel),
                            },
                        )
                        parallel_mode = self._resolve_subagent_parallel(
                            agent_parallelism_mode=agent.subagent_parallelism_mode,
                            router_parallel=r_decision.parallel,
                        )
                        subagent_batch_started_s = time.time()
                        subagent_span = self._telemetry_start_span(
                            obs_contracts.SPAN_AGENT_SUBAGENT_BATCH,
                            attributes={
                                "run_id": run_id,
                                "thread_id": t_id,
                                "step": step,
                                "target_count": len(r_decision.targets),
                                "parallel": parallel_mode,
                            },
                        )
                        records, bridge, child_tool_execs = await self._run_subagents(
                            agent=agent,
                            targets=r_decision.targets,
                            parallel=parallel_mode,
                            router_metadata=r_decision.metadata,
                            context=ctx,
                            thread_id=t_id,
                            depth=depth + 1,
                            lineage=(*lineage, id(agent)),
                            run_id=run_id,
                            step=step,
                            handle=handle,
                            memory=memory,
                            user_id=self._maybe_str(ctx.get("user_id")),
                        )
                        subagent_latency_ms = (
                            time.time() - subagent_batch_started_s
                        ) * 1000.0
                        self._telemetry_histogram(
                            obs_contracts.METRIC_AGENT_SUBAGENT_BATCH_LATENCY_MS,
                            value=subagent_latency_ms,
                            attributes={
                                "parallel": parallel_mode,
                                "target_count": len(r_decision.targets),
                            },
                        )
                        self._telemetry_counter(
                            obs_contracts.METRIC_AGENT_SUBAGENT_BATCHES_TOTAL,
                            value=1,
                            attributes={
                                "result": "success"
                                if all(record.success for record in records)
                                else "partial_or_error",
                                "parallel": parallel_mode,
                            },
                        )
                        self._telemetry_end_span(
                            subagent_span,
                            status="ok"
                            if all(record.success for record in records)
                            else "error",
                            attributes={
                                "latency_ms": subagent_latency_ms,
                                "success_count": sum(
                                    1 for record in records if record.success
                                ),
                                "failure_count": sum(
                                    1 for record in records if not record.success
                                ),
                            },
                        )
                        sub_execs.extend(records)
                        tool_execs.extend(child_tool_execs)
                        await self._persist_checkpoint(
                            memory=memory,
                            thread_id=t_id,
                            run_id=run_id,
                            step=step,
                            phase="post_subagent_batch",
                            payload={
                                "success_count": sum(
                                    1 for record in records if record.success
                                ),
                                "failure_count": sum(
                                    1 for record in records if not record.success
                                ),
                            },
                        )
                        if any(not record.success for record in records):
                            sub_policy_outcome = self._apply_subagent_failure_policy(
                                fail_safe.subagent_failure_policy
                            )
                            if sub_policy_outcome == "fail":
                                raise SubagentExecutionError(
                                    "Subagent execution failed and policy requires run failure"
                                )
                            if sub_policy_outcome == "degrade":
                                state = self._transition_state(state, "degraded")
                                final_text = (
                                    "Subagent execution failed under degrade policy"
                                )
                                break
                        if bridge:
                            messages.append(Message(role="assistant", content=bridge))

                request_id = f"{run_id}:step:{step}"
                if pending_llm_response is None:
                    llm_calls += 1
                    await self._persist_checkpoint(
                        memory=memory,
                        thread_id=t_id,
                        run_id=run_id,
                        step=step,
                        phase="pre_llm",
                        payload={
                            "model": model_name,
                            "provider": llm.provider_id,
                            "message_count": len(messages),
                        },
                    )
                    llm_policy_decision = await self._evaluate_policy(
                        agent=agent,
                        event=PolicyEvent(
                            event_type="llm_before_execute",
                            run_id=run_id,
                            thread_id=t_id,
                            step=step,
                            context={
                                str(k): json_value_from_tool_result(v)
                                for k, v in ctx.items()
                            },
                            metadata={
                                "model": model_name,
                                "provider": llm.provider_id,
                            },
                        ),
                        handle=handle,
                        memory=memory,
                        user_id=self._maybe_str(ctx.get("user_id")),
                        state=state,
                    )
                    if llm_policy_decision.action == "deny":
                        llm_outcome = self._apply_llm_failure_policy(
                            fail_safe.llm_failure_policy
                        )
                        if llm_outcome == "degrade":
                            state = self._transition_state(state, "degraded")
                            final_text = (
                                llm_policy_decision.reason
                                or "LLM call denied by policy under degrade setting"
                            )
                            break
                        raise AgentExecutionError(
                            llm_policy_decision.reason or "LLM call denied by policy"
                        )
                    if llm_policy_decision.action in {"defer", "request_approval"}:
                        approved = await self._request_approval(
                            handle=handle,
                            memory=memory,
                            run_id=run_id,
                            thread_id=t_id,
                            step=step,
                            reason=(
                                llm_policy_decision.reason
                                or f"Approval requested before LLM call for model '{model_name}'"
                            ),
                            payload=llm_policy_decision.request_payload,
                            user_id=self._maybe_str(ctx.get("user_id")),
                        )
                        if not approved:
                            denial_outcome = self._apply_approval_denial_policy(
                                fail_safe.approval_denial_policy
                            )
                            if denial_outcome == "degrade":
                                state = self._transition_state(state, "degraded")
                                final_text = (
                                    "LLM call approval denied under degrade setting"
                                )
                                break
                            raise AgentExecutionError("LLM call approval denied")
                    if llm_policy_decision.action == "request_user_input":
                        llm_input_decision = await self._request_user_input(
                            handle=handle,
                            memory=memory,
                            run_id=run_id,
                            thread_id=t_id,
                            step=step,
                            prompt=self._resolve_user_input_prompt(
                                tool_name="llm_call",
                                decision=llm_policy_decision,
                            ),
                            payload=llm_policy_decision.request_payload,
                            user_id=self._maybe_str(ctx.get("user_id")),
                        )
                        if llm_input_decision.kind != "allow":
                            denial_outcome = self._apply_approval_denial_policy(
                                fail_safe.approval_denial_policy
                            )
                            if denial_outcome == "degrade":
                                state = self._transition_state(state, "degraded")
                                final_text = (
                                    "LLM user-input gate denied under degrade setting"
                                )
                                break
                            raise AgentExecutionError(
                                llm_input_decision.reason
                                or "LLM user-input gate denied"
                            )
                        if (
                            isinstance(llm_input_decision.value, str)
                            and llm_input_decision.value
                        ):
                            messages.append(
                                Message(
                                    role="user",
                                    content=llm_input_decision.value,
                                )
                            )
                    req = LLMRequest(
                        model=model_name,
                        request_id=request_id,
                        messages=list(messages),
                        tools=llm_tools,
                        tool_choice="auto" if llm_tools else None,
                        idempotency_key=f"{run_id}:step:{step}",
                        session_token=session_token,
                        checkpoint_token=checkpoint_token,
                        metadata={
                            "run_id": run_id,
                            "thread_id": t_id,
                            "agent_name": agent.name,
                            "content_channels": {
                                "system": "trusted_system",
                                "tool": "untrusted_tool_output",
                            },
                            "tool_output_sanitized": self.config.sanitize_tool_output,
                        },
                    )
                    await self._emit(
                        handle,
                        memory,
                        AgentRunEvent(
                            type="llm_called",
                            run_id=run_id,
                            thread_id=t_id,
                            state=state,
                            step=step,
                            data={"model": model_name, "provider": llm.provider_id},
                        ),
                        user_id=self._maybe_str(ctx.get("user_id")),
                    )
                    llm_candidates = self._build_llm_candidates(
                        primary=resolved,
                        fallback_chain=fail_safe.fallback_model_chain,
                        resolver=agent.model_resolver,
                    )
                    llm_error: Exception | None = None
                    resp = None
                    for candidate in llm_candidates:
                        candidate_key = (
                            f"llm:{candidate.adapter}:{candidate.normalized_model}"
                        )
                        llm_call_started_s: float | None = None
                        llm_span = None
                        try:
                            await breaker.ensure_closed(candidate_key)
                            candidate_req = LLMRequest(
                                model=candidate.normalized_model,
                                request_id=req.request_id,
                                idempotency_key=req.idempotency_key,
                                session_token=req.session_token,
                                checkpoint_token=req.checkpoint_token,
                                messages=list(req.messages),
                                tools=req.tools,
                                tool_choice=req.tool_choice,
                                max_tokens=req.max_tokens,
                                temperature=req.temperature,
                                top_p=req.top_p,
                                stop=req.stop,
                                thinking=req.thinking,
                                thinking_effort=req.thinking_effort,
                                max_thinking_tokens=req.max_thinking_tokens,
                                timeout_s=req.timeout_s,
                                metadata=dict(req.metadata),
                                extra=dict(req.extra),
                            )
                            llm_call_started_s = time.time()
                            llm_span = self._telemetry_start_span(
                                obs_contracts.SPAN_AGENT_LLM_CALL,
                                attributes={
                                    "run_id": run_id,
                                    "thread_id": t_id,
                                    "step": step,
                                    "provider": candidate.llm.provider_id,
                                    "adapter": candidate.adapter,
                                    "model": candidate.normalized_model,
                                },
                            )
                            response = await self._chat_with_interrupt_support(
                                handle=handle,
                                llm=candidate.llm,
                                req=candidate_req,
                                on_text_delta=(
                                    lambda delta: self._emit(
                                        handle,
                                        memory,
                                        AgentRunEvent(
                                            type="text_delta",
                                            run_id=run_id,
                                            thread_id=t_id,
                                            state=state,
                                            step=step,
                                            data={"delta": delta},
                                        ),
                                        user_id=self._maybe_str(ctx.get("user_id")),
                                    )
                                )
                                if handle.stream_text_deltas_enabled()
                                else None,
                            )
                            llm_latency_ms = (time.time() - llm_call_started_s) * 1000.0
                            self._telemetry_histogram(
                                obs_contracts.METRIC_AGENT_LLM_LATENCY_MS,
                                value=llm_latency_ms,
                                attributes={
                                    "provider": candidate.llm.provider_id,
                                    "adapter": candidate.adapter,
                                    "model": candidate.normalized_model,
                                },
                            )
                            self._telemetry_counter(
                                obs_contracts.METRIC_AGENT_LLM_CALLS_TOTAL,
                                value=1,
                                attributes={
                                    "result": "success",
                                    "provider": candidate.llm.provider_id,
                                    "adapter": candidate.adapter,
                                    "model": candidate.normalized_model,
                                },
                            )
                            self._telemetry_end_span(
                                llm_span,
                                status="ok",
                                attributes={
                                    "latency_ms": llm_latency_ms,
                                },
                            )
                            resp = response
                            final_resp = response
                            breaker.record_success(candidate_key)
                            requested_model = (
                                requested_model or candidate.requested_model
                            )
                            normalized_model = candidate.normalized_model
                            provider_adapter = candidate.adapter
                            model_name = candidate.normalized_model
                            llm = candidate.llm
                            break
                        except Exception as e:
                            if llm_call_started_s is not None:
                                llm_latency_ms = (
                                    time.time() - llm_call_started_s
                                ) * 1000.0
                                self._telemetry_histogram(
                                    obs_contracts.METRIC_AGENT_LLM_LATENCY_MS,
                                    value=llm_latency_ms,
                                    attributes={
                                        "provider": candidate.llm.provider_id,
                                        "adapter": candidate.adapter,
                                        "model": candidate.normalized_model,
                                    },
                                )
                                self._telemetry_end_span(
                                    llm_span,
                                    status="error",
                                    error=str(e),
                                    attributes={"latency_ms": llm_latency_ms},
                                )
                            self._telemetry_counter(
                                obs_contracts.METRIC_AGENT_LLM_CALLS_TOTAL,
                                value=1,
                                attributes={
                                    "result": "error",
                                    "provider": candidate.llm.provider_id,
                                    "adapter": candidate.adapter,
                                    "model": candidate.normalized_model,
                                },
                            )
                            llm_error = e
                            await breaker.record_failure(candidate_key)
                            continue

                    if resp is None:
                        llm_outcome = self._apply_llm_failure_policy(
                            fail_safe.llm_failure_policy,
                        )
                        if llm_outcome == "degrade":
                            state = self._transition_state(state, "degraded")
                            final_text = f"LLM call failed: {llm_error}"
                            await self._emit(
                                handle,
                                memory,
                                AgentRunEvent(
                                    type="warning",
                                    run_id=run_id,
                                    thread_id=t_id,
                                    state=state,
                                    step=step,
                                    message=final_text,
                                ),
                                user_id=self._maybe_str(ctx.get("user_id")),
                            )
                            break
                        if isinstance(llm_error, Exception):
                            raise llm_error
                        raise AgentExecutionError("LLM call failed with unknown error")

                    usage = usage.add_usage(resp.usage)
                    total_cost_usd = self._accumulate_cost(total_cost_usd, resp)
                    if (
                        fail_safe.max_total_cost_usd is not None
                        and total_cost_usd > fail_safe.max_total_cost_usd
                    ):
                        raise AgentBudgetExceededError(
                            f"Exceeded max_total_cost_usd={fail_safe.max_total_cost_usd}"
                        )
                    session_token = resp.session_token or session_token
                    checkpoint_token = resp.checkpoint_token or checkpoint_token
                    pending_llm_response = resp
                    await self._persist_checkpoint(
                        memory=memory,
                        thread_id=t_id,
                        run_id=run_id,
                        step=step,
                        phase="post_llm",
                        payload={
                            "model": model_name,
                            "provider": llm.provider_id,
                            "finish_reason": resp.finish_reason or "",
                            "tool_call_count": len(resp.tool_calls),
                            "session_token": session_token or "",
                            "checkpoint_token": checkpoint_token or "",
                            "total_cost_usd": total_cost_usd,
                        },
                    )
                    await self._persist_runtime_snapshot(
                        memory=memory,
                        thread_id=t_id,
                        run_id=run_id,
                        step=step,
                        state=state,
                        context=ctx,
                        messages=messages,
                        llm_calls=llm_calls,
                        tool_calls=tool_calls,
                        started_at_s=started_at_s,
                        usage=usage,
                        total_cost_usd=total_cost_usd,
                        session_token=session_token,
                        checkpoint_token=checkpoint_token,
                        requested_model=requested_model,
                        normalized_model=normalized_model,
                        provider_adapter=provider_adapter,
                        tool_execs=tool_execs,
                        sub_execs=sub_execs,
                        skill_reads=skill_reads,
                        skill_cmd_execs=skill_cmd_execs,
                        final_text=final_text,
                        final_structured=final_structured,
                        pending_llm_response=pending_llm_response,
                        final_response=final_resp,
                        replayed_effect_count=replayed_effect_count,
                    )
                else:
                    resp = pending_llm_response
                    pending_llm_response = None
                    if resp is None:
                        raise AgentCheckpointCorruptionError(
                            "pending_llm_response checkpoint was invalid"
                        )

                if resp.text:
                    messages.append(Message(role="assistant", content=resp.text))

                await self._emit(
                    handle,
                    memory,
                    AgentRunEvent(
                        type="llm_completed",
                        run_id=run_id,
                        thread_id=t_id,
                        state=state,
                        step=step,
                        data={
                            "tool_call_count": len(resp.tool_calls),
                            "finish_reason": resp.finish_reason or "",
                            "text": resp.text or "",
                        },
                    ),
                    user_id=self._maybe_str(ctx.get("user_id")),
                )

                if not resp.tool_calls:
                    final_text = resp.text
                    final_structured = resp.structured_response
                    pending_llm_response = None
                    state = self._transition_state(state, "completed")
                    break

                await self._emit(
                    handle,
                    memory,
                    AgentRunEvent(
                        type="tool_batch_started",
                        run_id=run_id,
                        thread_id=t_id,
                        state=state,
                        step=step,
                        data={
                            "tool_call_count": len(resp.tool_calls),
                            "tool_names": [call.tool_name for call in resp.tool_calls],
                            "tool_call_ids": [call.id for call in resp.tool_calls],
                        },
                    ),
                    user_id=self._maybe_str(ctx.get("user_id")),
                )
                await self._persist_checkpoint(
                    memory=memory,
                    thread_id=t_id,
                    run_id=run_id,
                    step=step,
                    phase="pre_tool_batch",
                    payload={"tool_call_count": len(resp.tool_calls)},
                )

                calls: list[tuple[str, dict[str, Any]]] = []
                tool_ids: list[str | None] = []
                call_contexts: list[ToolContext] = []
                call_timeouts: list[float | None] = []
                call_sandbox_profiles: list[Any | None] = []
                for tc in resp.tool_calls:
                    tool_name = tc.tool_name
                    raw_args = dict(tc.arguments)

                    decision = await self._evaluate_policy(
                        agent=agent,
                        event=PolicyEvent(
                            event_type="tool_before_execute",
                            run_id=run_id,
                            thread_id=t_id,
                            step=step,
                            context=ctx,
                            tool_name=tool_name,
                            tool_args=raw_args,
                        ),
                        handle=handle,
                        memory=memory,
                        user_id=self._maybe_str(ctx.get("user_id")),
                        state=state,
                    )
                    if decision.action in {"deny"}:
                        record = ToolExecutionRecord(
                            tool_name=tool_name,
                            tool_call_id=tc.id,
                            success=False,
                            error=decision.reason or "Denied by policy",
                        )
                        tool_execs.append(record)
                        payload = {
                            "success": False,
                            "error": record.error or "denied",
                            "tool_call_id": tc.id,
                        }
                        messages.append(
                            Message(
                                role="tool",
                                name=tool_name,
                                content=json.dumps(payload, ensure_ascii=True),
                            )
                        )
                        denial_outcome = self._apply_approval_denial_policy(
                            fail_safe.approval_denial_policy
                        )
                        if denial_outcome == "fail":
                            raise AgentExecutionError(
                                f"Tool '{tool_name}' denied by policy and approval_denial_policy requires fail"
                            )
                        if denial_outcome == "degrade":
                            state = self._transition_state(state, "degraded")
                            final_text = f"Tool '{tool_name}' denied by policy under degrade setting"
                            break
                        continue

                    if decision.action in {
                        "defer",
                        "request_approval",
                        "request_user_input",
                    }:
                        if decision.action == "request_user_input" or (
                            decision.action == "defer"
                            and self._is_defer_user_input(decision)
                        ):
                            user_decision = await self._request_user_input(
                                handle=handle,
                                memory=memory,
                                run_id=run_id,
                                thread_id=t_id,
                                step=step,
                                prompt=self._resolve_user_input_prompt(
                                    tool_name=tool_name,
                                    decision=decision,
                                ),
                                payload=decision.request_payload,
                                user_id=self._maybe_str(ctx.get("user_id")),
                            )
                            if user_decision.kind != "allow":
                                record = ToolExecutionRecord(
                                    tool_name=tool_name,
                                    tool_call_id=tc.id,
                                    success=False,
                                    error=user_decision.reason or "User input denied",
                                )
                                tool_execs.append(record)
                                messages.append(
                                    Message(
                                        role="tool",
                                        name=tool_name,
                                        content=json.dumps(
                                            {"success": False, "error": record.error},
                                            ensure_ascii=True,
                                        ),
                                    )
                                )
                                denial_outcome = self._apply_approval_denial_policy(
                                    fail_safe.approval_denial_policy
                                )
                                if denial_outcome == "fail":
                                    raise AgentExecutionError(
                                        f"Tool '{tool_name}' blocked by user input policy"
                                    )
                                if denial_outcome == "degrade":
                                    state = self._transition_state(state, "degraded")
                                    final_text = f"Tool '{tool_name}' blocked by user input under degrade setting"
                                    break
                                continue
                            target_arg = decision.request_payload.get("target_arg")
                            if isinstance(target_arg, str) and target_arg.strip():
                                raw_args[target_arg.strip()] = user_decision.value or ""
                        else:
                            approved = await self._request_approval(
                                handle=handle,
                                memory=memory,
                                run_id=run_id,
                                thread_id=t_id,
                                step=step,
                                reason=decision.reason
                                or f"Approval requested for {tool_name}",
                                payload=decision.request_payload,
                                user_id=self._maybe_str(ctx.get("user_id")),
                            )
                            if not approved:
                                record = ToolExecutionRecord(
                                    tool_name=tool_name,
                                    tool_call_id=tc.id,
                                    success=False,
                                    error="Approval denied",
                                )
                                tool_execs.append(record)
                                messages.append(
                                    Message(
                                        role="tool",
                                        name=tool_name,
                                        content=json.dumps(
                                            {
                                                "success": False,
                                                "error": "Approval denied",
                                            },
                                            ensure_ascii=True,
                                        ),
                                    )
                                )
                                denial_outcome = self._apply_approval_denial_policy(
                                    fail_safe.approval_denial_policy
                                )
                                if denial_outcome == "fail":
                                    raise AgentExecutionError(
                                        f"Tool '{tool_name}' approval denied and policy requires fail"
                                    )
                                if denial_outcome == "degrade":
                                    state = self._transition_state(state, "degraded")
                                    final_text = f"Tool '{tool_name}' approval denied under degrade setting"
                                    break
                                continue

                    if decision.updated_tool_args is not None:
                        raw_args = dict(decision.updated_tool_args)

                    effective_sandbox_profile = resolve_sandbox_profile(
                        tool_name=tool_name,
                        tool_args=raw_args,
                        run_context=ctx,
                        default_profile=self.config.default_sandbox_profile,
                        provider=self.config.sandbox_profile_provider,
                    )
                    if effective_sandbox_profile is not None:
                        sandbox_violation = validate_tool_args_against_sandbox(
                            tool_name=tool_name,
                            tool_args=raw_args,
                            profile=effective_sandbox_profile,
                            cwd=Path.cwd(),
                        )
                        if sandbox_violation is not None:
                            record = ToolExecutionRecord(
                                tool_name=tool_name,
                                tool_call_id=tc.id,
                                success=False,
                                error=sandbox_violation,
                            )
                            tool_execs.append(record)
                            messages.append(
                                Message(
                                    role="tool",
                                    name=tool_name,
                                    content=json.dumps(
                                        {"success": False, "error": sandbox_violation},
                                        ensure_ascii=True,
                                    ),
                                )
                            )
                            denial_outcome = self._apply_tool_failure_policy(
                                fail_safe.tool_failure_policy
                            )
                            if denial_outcome == "fail":
                                raise AgentExecutionError(
                                    f"Tool '{tool_name}' blocked by sandbox policy"
                                )
                            if denial_outcome == "degrade":
                                state = self._transition_state(state, "degraded")
                                final_text = f"Tool '{tool_name}' blocked by sandbox under degrade setting"
                                break
                            continue

                    calls.append((tool_name, raw_args))
                    tool_ids.append(tc.id)
                    call_metadata: dict[str, Any] = {
                        "run_id": run_id,
                        "thread_id": t_id,
                    }
                    if effective_sandbox_profile is not None:
                        call_metadata["sandbox_profile_id"] = (
                            effective_sandbox_profile.profile_id
                        )
                    secret_provider = self.config.secret_scope_provider
                    if secret_provider is not None:
                        try:
                            secret_values = secret_provider.resolve(
                                tool_name=tool_name,
                                tool_args=raw_args,
                                run_context=ctx,
                            )
                        except Exception as e:
                            raise AgentExecutionError(
                                f"secret scope resolution failed for tool '{tool_name}': {e}"
                            ) from e
                        if secret_values:
                            call_metadata["secrets"] = {
                                str(key): str(value)
                                for key, value in secret_values.items()
                            }
                    call_contexts.append(
                        ToolContext(
                            request_id=request_id,
                            user_id=self._maybe_str(ctx.get("user_id")),
                            metadata=call_metadata,
                        )
                    )
                    call_timeouts.append(
                        effective_sandbox_profile.command_timeout_s
                        if effective_sandbox_profile is not None
                        else None
                    )
                    call_sandbox_profiles.append(effective_sandbox_profile)

                if state == "degraded":
                    break

                if calls:
                    tool_batch_started_s = time.time()
                    tool_span = self._telemetry_start_span(
                        obs_contracts.SPAN_AGENT_TOOL_BATCH,
                        attributes={
                            "run_id": run_id,
                            "thread_id": t_id,
                            "step": step,
                            "call_count": len(calls),
                        },
                    )
                    resolved_results: list[Any | None] = [None] * len(calls)
                    replayed_indices: set[int] = set()
                    execution_indices: list[int] = []
                    call_latency_ms: dict[int, float] = {}

                    for idx, (tool_name, call_args) in enumerate(calls):
                        call_id = tool_ids[idx] if idx < len(tool_ids) else None
                        replay_result = await self._resolve_effect_replay_result(
                            memory=memory,
                            thread_id=t_id,
                            run_id=run_id,
                            step=step,
                            tool_call_id=call_id,
                            tool_name=tool_name,
                            call_args=call_args,
                        )
                        if replay_result is not None:
                            resolved_results[idx] = replay_result
                            replayed_indices.add(idx)
                            replayed_effect_count += 1
                        else:
                            execution_indices.append(idx)

                    for idx in execution_indices:
                        tool_name = calls[idx][0]
                        try:
                            await breaker.ensure_closed(f"tool:{tool_name}")
                        except Exception as e:
                            tool_outcome = self._apply_tool_failure_policy(
                                fail_safe.tool_failure_policy
                            )
                            if tool_outcome == "fail":
                                raise e
                            state = self._transition_state(state, "degraded")
                            final_text = str(e)
                            break
                    if state == "degraded":
                        break

                    if execution_indices:

                        async def _exec_one(
                            mapped_idx: int,
                        ) -> tuple[ToolResult[Any] | Exception, float]:
                            """Execute one tool call for mapped index and capture exceptions."""
                            call_name, call_args = calls[mapped_idx]
                            call_ctx = call_contexts[mapped_idx]
                            call_timeout = call_timeouts[mapped_idx]
                            call_tool_id = (
                                tool_ids[mapped_idx] or f"{run_id}:{step}:{mapped_idx}"
                            )
                            started = time.time()
                            try:
                                return (
                                    await registry.call(
                                        call_name,
                                        call_args,
                                        ctx=call_ctx,
                                        timeout=call_timeout,
                                        tool_call_id=call_tool_id,
                                    ),
                                    (time.time() - started) * 1000.0,
                                )
                            except Exception as exc:
                                return exc, (time.time() - started) * 1000.0

                        execution_results = await asyncio.gather(
                            *[_exec_one(idx) for idx in execution_indices],
                            return_exceptions=False,
                        )
                        for mapped_idx, row in zip(
                            execution_indices, execution_results
                        ):
                            result, latency_ms = row
                            resolved_results[mapped_idx] = result
                            call_latency_ms[mapped_idx] = latency_ms

                    for idx, result in enumerate(resolved_results):
                        tool_name = calls[idx][0]
                        call_args = calls[idx][1]
                        call_id = tool_ids[idx] if idx < len(tool_ids) else None
                        call_profile = (
                            call_sandbox_profiles[idx]
                            if idx < len(call_sandbox_profiles)
                            else None
                        )
                        tool_calls += 1

                        source = "replay" if idx in replayed_indices else "execute"
                        tool_key = f"tool:{tool_name}"
                        if idx in replayed_indices:
                            tr = (
                                result
                                if isinstance(result, ToolResult)
                                else ToolResult(
                                    output=None,
                                    success=False,
                                    error_message="Invalid replay result",
                                )
                            )
                            if tr.success:
                                breaker.record_success(tool_key)
                            else:
                                await breaker.record_failure(tool_key)
                        elif isinstance(result, Exception):
                            await breaker.record_failure(tool_key)
                            tr = ToolResult(
                                output=None,
                                success=False,
                                error_message=str(result),
                            )
                        else:
                            if result is None:
                                tr = ToolResult(
                                    output=None,
                                    success=False,
                                    error_message="Missing tool execution result",
                                )
                                await breaker.record_failure(tool_key)
                            elif not isinstance(result, ToolResult):
                                tr = ToolResult(
                                    output=None,
                                    success=False,
                                    error_message=(
                                        f"Unexpected tool result type: {type(result).__name__}"
                                    ),
                                )
                                await breaker.record_failure(tool_key)
                            else:
                                tr = result
                            if tr.success:
                                breaker.record_success(tool_key)
                            else:
                                await breaker.record_failure(tool_key)

                        latency_ms = call_latency_ms.get(idx)
                        self._telemetry_counter(
                            obs_contracts.METRIC_AGENT_TOOL_CALLS_TOTAL,
                            value=1,
                            attributes={
                                "tool_name": tool_name,
                                "result": "success" if tr.success else "error",
                                "source": source,
                            },
                        )
                        if latency_ms is not None:
                            self._telemetry_histogram(
                                obs_contracts.METRIC_AGENT_TOOL_CALL_LATENCY_MS,
                                value=latency_ms,
                                attributes={
                                    "tool_name": tool_name,
                                    "result": "success" if tr.success else "error",
                                    "source": source,
                                },
                            )

                        tr = apply_tool_output_limits(
                            tr,
                            profile=call_profile,
                        )

                        rec = tool_record_from_result(
                            tool_name,
                            call_id,
                            tr,
                            latency_ms=latency_ms,
                            agent_name=agent.name,
                            agent_depth=depth,
                            agent_path=agent.name,
                        )
                        tool_execs.append(rec)

                        if call_id:
                            input_hash = json_hash(
                                {"tool_name": tool_name, "args": call_args}
                            )
                            output_value = json_value_from_tool_result(tr.output)
                            output_hash = json_hash({"output": output_value})
                            await self._persist_effect_result(
                                memory=memory,
                                thread_id=t_id,
                                run_id=run_id,
                                step=step,
                                tool_call_id=call_id,
                                input_hash=input_hash,
                                output_hash=output_hash,
                                output=output_value,
                                success=tr.success,
                            )

                        if tool_name == "read_skill_md" and tr.success:
                            out = tr.output if isinstance(tr.output, dict) else {}
                            skill_reads.append(
                                SkillReadRecord(
                                    skill_name=str(out.get("skill_name", "")),
                                    path=str(out.get("path", "")),
                                    checksum=(
                                        str(out.get("checksum", ""))
                                        if out.get("checksum") is not None
                                        else None
                                    ),
                                )
                            )
                        if tool_name == "run_skill_command" and isinstance(
                            tr.output, dict
                        ):
                            cmd = tr.output.get("command")
                            if isinstance(cmd, list):
                                skill_cmd_execs.append(
                                    CommandExecutionRecord(
                                        command=[str(x) for x in cmd],
                                        exit_code=int(tr.output.get("exit_code", 1)),
                                        stdout=str(tr.output.get("stdout", "")),
                                        stderr=str(tr.output.get("stderr", "")),
                                        denied=False,
                                    )
                                )

                        tool_payload = {
                            "success": tr.success,
                            "output": json_value_from_tool_result(tr.output),
                            "error": tr.error_message,
                        }
                        messages.append(
                            Message(
                                role="tool",
                                name=tool_name,
                                content=render_untrusted_tool_message(
                                    tool_name=tool_name,
                                    payload=tool_payload,
                                    max_chars=(
                                        min(
                                            self.config.tool_output_max_chars,
                                            call_profile.max_output_chars,
                                        )
                                        if call_profile is not None
                                        else self.config.tool_output_max_chars
                                    ),
                                )
                                if self.config.sanitize_tool_output
                                else json.dumps(tool_payload, ensure_ascii=True),
                            )
                        )

                        await self._emit(
                            handle,
                            memory,
                            AgentRunEvent(
                                type="tool_completed",
                                run_id=run_id,
                                thread_id=t_id,
                                state=state,
                                step=step,
                                data={
                                    "tool_name": tool_name,
                                    "tool_call_id": call_id,
                                    "success": tr.success,
                                    "output": json_value_from_tool_result(tr.output),
                                    "error": tr.error_message,
                                    "agent_name": agent.name,
                                    "agent_depth": depth,
                                    "agent_path": agent.name,
                                },
                            ),
                            user_id=self._maybe_str(ctx.get("user_id")),
                        )

                        if not tr.success:
                            tool_outcome = self._apply_tool_failure_policy(
                                fail_safe.tool_failure_policy
                            )
                            if tool_outcome == "fail":
                                raise AgentExecutionError(
                                    f"Tool '{tool_name}' failed and policy requires run failure"
                                )
                            if tool_outcome == "degrade":
                                state = self._transition_state(state, "degraded")
                                final_text = (
                                    f"Tool '{tool_name}' failed under degrade policy: "
                                    f"{tr.error_message or 'unknown error'}"
                                )
                                break

                    await self._persist_checkpoint(
                        memory=memory,
                        thread_id=t_id,
                        run_id=run_id,
                        step=step,
                        phase="post_tool_batch",
                        payload={
                            "tool_calls_total": len(calls),
                            "tool_failures": sum(
                                1
                                for record in tool_execs
                                if not record.success
                                and record.tool_call_id in tool_ids
                            ),
                        },
                    )
                    tool_batch_latency_ms = (
                        time.time() - tool_batch_started_s
                    ) * 1000.0
                    tool_failure_count = sum(
                        1
                        for record in tool_execs
                        if not record.success and record.tool_call_id in tool_ids
                    )
                    self._telemetry_histogram(
                        obs_contracts.METRIC_AGENT_TOOL_BATCH_LATENCY_MS,
                        value=tool_batch_latency_ms,
                        attributes={
                            "call_count": len(calls),
                            "failure_count": tool_failure_count,
                        },
                    )
                    self._telemetry_counter(
                        obs_contracts.METRIC_AGENT_TOOL_BATCHES_TOTAL,
                        value=1,
                        attributes={
                            "result": "success"
                            if tool_failure_count == 0
                            else "partial_or_error",
                        },
                    )
                    self._telemetry_end_span(
                        tool_span,
                        status="ok" if tool_failure_count == 0 else "error",
                        attributes={
                            "latency_ms": tool_batch_latency_ms,
                            "call_count": len(calls),
                            "failure_count": tool_failure_count,
                            "replayed_count": len(replayed_indices),
                        },
                    )

                pending_llm_response = None
                await self._persist_runtime_snapshot(
                    memory=memory,
                    thread_id=t_id,
                    run_id=run_id,
                    step=step,
                    state=state,
                    context=ctx,
                    messages=messages,
                    llm_calls=llm_calls,
                    tool_calls=tool_calls,
                    started_at_s=started_at_s,
                    usage=usage,
                    total_cost_usd=total_cost_usd,
                    session_token=session_token,
                    checkpoint_token=checkpoint_token,
                    requested_model=requested_model,
                    normalized_model=normalized_model,
                    provider_adapter=provider_adapter,
                    tool_execs=tool_execs,
                    sub_execs=sub_execs,
                    skill_reads=skill_reads,
                    skill_cmd_execs=skill_cmd_execs,
                    final_text=final_text,
                    final_structured=final_structured,
                    pending_llm_response=pending_llm_response,
                    final_response=final_resp,
                    replayed_effect_count=replayed_effect_count,
                )

                if state == "degraded":
                    break

            result = AgentResult(
                run_id=run_id,
                thread_id=t_id,
                state=state,
                final_text=final_text,
                requested_model=requested_model,
                normalized_model=normalized_model,
                provider_adapter=provider_adapter,
                final_structured=final_structured,
                llm_response=final_resp,
                tool_executions=tool_execs,
                subagent_executions=sub_execs,
                skills_used=[s.name for s in skills.resolved_skills],
                skill_reads=skill_reads,
                skill_command_executions=skill_cmd_execs,
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
            await self._emit(
                handle,
                memory,
                AgentRunEvent(
                    type="run_completed",
                    run_id=run_id,
                    thread_id=t_id,
                    state=state,
                    message="Run completed",
                ),
                user_id=self._maybe_str(ctx.get("user_id")),
            )
            await self._persist_checkpoint(
                memory=memory,
                thread_id=t_id,
                run_id=run_id,
                step=step,
                phase="run_terminal",
                payload={
                    "state": state,
                    "final_text": final_text,
                    "requested_model": requested_model or "",
                    "normalized_model": normalized_model or "",
                    "provider_adapter": provider_adapter or "",
                    "terminal_result": self._serialize_agent_result(result),
                },
            )
            try:
                await self._flush_checkpoint_writes()
            except Exception as flush_err:
                await self._emit(
                    handle,
                    memory,
                    AgentRunEvent(
                        type="warning",
                        run_id=run_id,
                        thread_id=t_id,
                        state=state,
                        step=step,
                        message=f"Checkpoint flush failed: {flush_err}",
                    ),
                    user_id=self._maybe_str(ctx.get("user_id")),
                )
            run_span_status = "ok"
            await handle.set_result(result)

        except AgentInterruptedError as e:
            run_span_status = "cancelled"
            run_span_error = str(e)
            state = self._transition_state(state, "cancelled")
            await self._emit(
                handle,
                memory,
                AgentRunEvent(
                    type="run_interrupted",
                    run_id=run_id,
                    thread_id=t_id,
                    state=state,
                    step=step,
                    message=str(e),
                ),
                user_id=self._maybe_str(ctx.get("user_id")),
            )
            await self._persist_terminal_checkpoint(
                memory=memory,
                thread_id=t_id,
                run_id=run_id,
                step=step,
                state=state,
                message=str(e),
                final_text=str(e),
                requested_model=requested_model,
                normalized_model=normalized_model,
                provider_adapter=provider_adapter,
                final_structured=final_structured,
                llm_response=final_resp,
                tool_execs=tool_execs,
                sub_execs=sub_execs,
                skills=skills.resolved_skills if "skills" in locals() else [],
                skill_reads=skill_reads,
                skill_cmd_execs=skill_cmd_execs,
                usage=usage,
                total_cost_usd=total_cost_usd,
                session_token=session_token,
                checkpoint_token=checkpoint_token,
                llm_calls=llm_calls,
                tool_calls=tool_calls,
                started_at_s=started_at_s,
                replayed_effect_count=replayed_effect_count,
            )
            try:
                await self._flush_checkpoint_writes()
            except Exception as flush_err:
                await self._emit(
                    handle,
                    memory,
                    AgentRunEvent(
                        type="warning",
                        run_id=run_id,
                        thread_id=t_id,
                        state=state,
                        step=step,
                        message=f"Checkpoint flush failed: {flush_err}",
                    ),
                    user_id=self._maybe_str(ctx.get("user_id")),
                )
            await handle.set_exception(e)
        except asyncio.CancelledError:
            if handle.is_interrupt_requested():
                run_span_status = "cancelled"
                run_span_error = "Run interrupted by caller"
                state = self._transition_state(state, "cancelled")
                interrupted = AgentInterruptedError("Run interrupted by caller")
                await self._emit(
                    handle,
                    memory,
                    AgentRunEvent(
                        type="run_interrupted",
                        run_id=run_id,
                        thread_id=t_id,
                        state=state,
                        step=step,
                        message=str(interrupted),
                    ),
                    user_id=self._maybe_str(ctx.get("user_id")),
                )
                await self._persist_terminal_checkpoint(
                    memory=memory,
                    thread_id=t_id,
                    run_id=run_id,
                    step=step,
                    state=state,
                    message=str(interrupted),
                    final_text=str(interrupted),
                    requested_model=requested_model,
                    normalized_model=normalized_model,
                    provider_adapter=provider_adapter,
                    final_structured=final_structured,
                    llm_response=final_resp,
                    tool_execs=tool_execs,
                    sub_execs=sub_execs,
                    skills=skills.resolved_skills if "skills" in locals() else [],
                    skill_reads=skill_reads,
                    skill_cmd_execs=skill_cmd_execs,
                    usage=usage,
                    total_cost_usd=total_cost_usd,
                    session_token=session_token,
                    checkpoint_token=checkpoint_token,
                    llm_calls=llm_calls,
                    tool_calls=tool_calls,
                    started_at_s=started_at_s,
                    replayed_effect_count=replayed_effect_count,
                )
                try:
                    await self._flush_checkpoint_writes()
                except Exception as flush_err:
                    await self._emit(
                        handle,
                        memory,
                        AgentRunEvent(
                            type="warning",
                            run_id=run_id,
                            thread_id=t_id,
                            state=state,
                            step=step,
                            message=f"Checkpoint flush failed: {flush_err}",
                        ),
                        user_id=self._maybe_str(ctx.get("user_id")),
                    )
                await handle.set_exception(interrupted)
            else:
                run_span_status = "cancelled"
                run_span_error = "Run cancelled"
                state = self._transition_state(state, "cancelled")
                await self._emit(
                    handle,
                    memory,
                    AgentRunEvent(
                        type="run_cancelled",
                        run_id=run_id,
                        thread_id=t_id,
                        state=state,
                        step=step,
                        message="Run cancelled",
                    ),
                    user_id=self._maybe_str(ctx.get("user_id")),
                )
                await self._persist_terminal_checkpoint(
                    memory=memory,
                    thread_id=t_id,
                    run_id=run_id,
                    step=step,
                    state=state,
                    message="Run cancelled",
                    final_text="Run cancelled",
                    requested_model=requested_model,
                    normalized_model=normalized_model,
                    provider_adapter=provider_adapter,
                    final_structured=final_structured,
                    llm_response=final_resp,
                    tool_execs=tool_execs,
                    sub_execs=sub_execs,
                    skills=skills.resolved_skills if "skills" in locals() else [],
                    skill_reads=skill_reads,
                    skill_cmd_execs=skill_cmd_execs,
                    usage=usage,
                    total_cost_usd=total_cost_usd,
                    session_token=session_token,
                    checkpoint_token=checkpoint_token,
                    llm_calls=llm_calls,
                    tool_calls=tool_calls,
                    started_at_s=started_at_s,
                    replayed_effect_count=replayed_effect_count,
                )
                try:
                    await self._flush_checkpoint_writes()
                except Exception as flush_err:
                    await self._emit(
                        handle,
                        memory,
                        AgentRunEvent(
                            type="warning",
                            run_id=run_id,
                            thread_id=t_id,
                            state=state,
                            step=step,
                            message=f"Checkpoint flush failed: {flush_err}",
                        ),
                        user_id=self._maybe_str(ctx.get("user_id")),
                    )
                await handle.set_result(None)
        except Exception as e:
            run_span_status = "error"
            run_span_error = str(e)
            state = self._transition_state(state, "failed")
            await self._emit(
                handle,
                memory,
                AgentRunEvent(
                    type="run_failed",
                    run_id=run_id,
                    thread_id=t_id,
                    state=state,
                    step=step,
                    message=str(e),
                ),
                user_id=self._maybe_str(ctx.get("user_id")),
            )
            await self._persist_terminal_checkpoint(
                memory=memory,
                thread_id=t_id,
                run_id=run_id,
                step=step,
                state=state,
                message=str(e),
                final_text=str(e),
                requested_model=requested_model,
                normalized_model=normalized_model,
                provider_adapter=provider_adapter,
                final_structured=final_structured,
                llm_response=final_resp,
                tool_execs=tool_execs,
                sub_execs=sub_execs,
                skills=skills.resolved_skills if "skills" in locals() else [],
                skill_reads=skill_reads,
                skill_cmd_execs=skill_cmd_execs,
                usage=usage,
                total_cost_usd=total_cost_usd,
                session_token=session_token,
                checkpoint_token=checkpoint_token,
                llm_calls=llm_calls,
                tool_calls=tool_calls,
                started_at_s=started_at_s,
                replayed_effect_count=replayed_effect_count,
            )
            try:
                await self._flush_checkpoint_writes()
            except Exception as flush_err:
                await self._emit(
                    handle,
                    memory,
                    AgentRunEvent(
                        type="warning",
                        run_id=run_id,
                        thread_id=t_id,
                        state=state,
                        step=step,
                        message=f"Checkpoint flush failed: {flush_err}",
                    ),
                    user_id=self._maybe_str(ctx.get("user_id")),
                )
            if isinstance(e, AgentError):
                await handle.set_exception(e)
            else:
                await handle.set_exception(AgentExecutionError(str(e)))
        finally:
            run_duration_ms = (time.time() - run_started_s) * 1000.0
            self._telemetry_histogram(
                obs_contracts.METRIC_AGENT_RUN_DURATION_MS,
                value=run_duration_ms,
                attributes={
                    "state": state,
                    "status": run_span_status,
                },
            )
            self._telemetry_counter(
                obs_contracts.METRIC_AGENT_RUNS_TOTAL,
                value=1,
                attributes={
                    "state": state,
                    "status": run_span_status,
                },
            )
            self._telemetry_end_span(
                run_span,
                status="ok" if run_span_status == "ok" else "error",
                error=run_span_error,
                attributes={
                    "state": state,
                    "status": run_span_status,
                    "duration_ms": run_duration_ms,
                    "steps": step,
                    "llm_calls": llm_calls,
                    "tool_calls": tool_calls,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "total_cost_usd": total_cost_usd,
                },
            )
            self._active_runs -= 1
            if self._active_runs <= 0:
                self._active_runs = 0
                await self._stop_checkpoint_writer()
                if self._owns_memory_store and self._memory_store is not None:
                    try:
                        await self._memory_store.close()
                    finally:
                        self._memory_store = None
