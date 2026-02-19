"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Subagent and interaction/policy orchestration mixin.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from typing import Any

from ...agents.a2a import InternalA2AProtocol
from ...agents.contracts import AgentInvocationRequest, AgentInvocationResponse
from ...agents.core.base import BaseAgent
from ...agents.delegation import (
    DelegationEdge,
    DelegationNode,
    DelegationPlan,
    RetryPolicy,
)
from ...agents.errors import (
    AgentExecutionError,
    SubagentExecutionError,
    SubagentRoutingError,
)
from ...agents.policy.engine import PolicyDecision as RulePolicyDecision
from ...agents.lifecycle.runtime import to_message_payload
from ...agents.types import (
    AgentRunEvent,
    AgentState,
    ApprovalDecision,
    ApprovalRequest,
    PolicyDecision,
    PolicyEvent,
    RouterDecision,
    RouterInput,
    SubagentExecutionRecord,
    ToolExecutionRecord,
    UserInputDecision,
    UserInputRequest,
    json_value_from_tool_result,
)
from ...llms.types import JSONValue, Message
from ...memory import MemoryStore
from ...observability import contracts as obs_contracts
from ..runtime import DelegationEngine
from .types import _RunHandle


class RunnerInteractionMixin:
    """Implements policy evaluation, subagent dispatch, and HITL requests."""

    async def _run_subagents(
        self,
        *,
        agent: BaseAgent,
        targets: list[str],
        parallel: bool,
        router_metadata: dict[str, JSONValue] | None,
        context: dict[str, Any],
        thread_id: str,
        depth: int,
        lineage: tuple[int, ...],
        run_id: str,
        step: int,
        handle: _RunHandle,
        memory: MemoryStore,
        user_id: str | None,
    ) -> tuple[list[SubagentExecutionRecord], str, list[ToolExecutionRecord]]:
        """
        Execute selected subagents through DAG orchestration and A2A protocol.

        Args:
            agent: Parent agent declaring available subagents.
            targets: Subagent names selected by router/policy.
            parallel: Whether selected subagents run in parallel.
            router_metadata: Optional router metadata carrying a full
                `delegation_plan` payload.
            context: Parent run context snapshot.
            thread_id: Thread id used for child run continuity.
            depth: Current nesting depth.
            lineage: Current lineage tuple for nested tracing.
            run_id: Parent run identifier.
            step: Parent step number.
            handle: Parent run handle for event emission.
            memory: Memory backend for emitted events.
            user_id: Optional user id propagated to memory events.

        Returns:
            Tuple of execution records, bridge text inserted back into parent
            transcript.
        """
        index = {sa.name: sa for sa in agent.subagents}
        selected_names: list[str] = []
        for name in targets:
            normalized = name.strip()
            if not normalized:
                continue
            if normalized not in index:
                raise SubagentRoutingError(f"Unknown subagent target '{name}'")
            selected_names.append(normalized)

        if not parallel and selected_names:
            selected_names = selected_names[:1]
        if not selected_names:
            return [], "", []

        engine = DelegationEngine(
            max_parallel_subagents_global=self.config.max_parallel_subagents_global,
            max_parallel_subagents_per_parent=self.config.max_parallel_subagents_per_parent,
            max_parallel_subagents_per_target_agent=self.config.max_parallel_subagents_per_target_agent,
            subagent_queue_backpressure_limit=self.config.subagent_queue_backpressure_limit,
        )

        maybe_plan = self._delegation_plan_from_metadata(router_metadata)
        plan = maybe_plan or engine.planner.create_plan(
            targets=selected_names,
            parallel=parallel,
            max_parallelism=(
                self.config.max_parallel_subagents_per_parent if parallel else 1
            ),
        )

        async def _dispatch(request: AgentInvocationRequest) -> AgentInvocationResponse:
            sub = index.get(request.target_agent)
            if sub is None:
                return AgentInvocationResponse(
                    run_id=request.run_id,
                    thread_id=request.thread_id,
                    conversation_id=request.conversation_id,
                    correlation_id=request.correlation_id,
                    idempotency_key=request.idempotency_key,
                    source_agent=request.target_agent,
                    target_agent=request.source_agent,
                    success=False,
                    error=f"Unknown subagent target '{request.target_agent}'",
                    metadata={"retryable": False},
                )

            started = time.time()
            await self._emit(
                handle,
                memory,
                AgentRunEvent(
                    type="subagent_started",
                    run_id=run_id,
                    thread_id=thread_id,
                    state="running",
                    step=step,
                    data={
                        "subagent_name": sub.name,
                        "correlation_id": request.correlation_id,
                    },
                ),
                user_id=user_id,
            )
            try:
                decision = await self._evaluate_policy(
                    agent=agent,
                    event=PolicyEvent(
                        event_type="subagent_before_execute",
                        run_id=run_id,
                        thread_id=thread_id,
                        step=step,
                        context={
                            str(k): json_value_from_tool_result(v)
                            for k, v in context.items()
                        },
                        subagent_name=sub.name,
                        metadata=request.metadata,
                    ),
                    handle=handle,
                    memory=memory,
                    user_id=user_id,
                    state="running",
                )
                if decision.action == "deny":
                    reason = (
                        decision.reason or f"Subagent '{sub.name}' denied by policy"
                    )
                    await self._emit(
                        handle,
                        memory,
                        AgentRunEvent(
                            type="subagent_completed",
                            run_id=run_id,
                            thread_id=thread_id,
                            state="running",
                            step=step,
                            data={
                                "subagent_name": sub.name,
                                "success": False,
                                "error": reason,
                            },
                        ),
                        user_id=user_id,
                    )
                    return AgentInvocationResponse(
                        run_id=request.run_id,
                        thread_id=request.thread_id,
                        conversation_id=request.conversation_id,
                        correlation_id=request.correlation_id,
                        idempotency_key=request.idempotency_key,
                        source_agent=sub.name,
                        target_agent=request.source_agent,
                        success=False,
                        error=reason,
                        metadata={"retryable": False},
                    )

                inherited = self._build_subagent_context(
                    context=context,
                    inherit_keys=sub.inherit_context_keys,
                )
                for key, value in request.payload.items():
                    inherited[str(key)] = value

                sub_handle = await self.run_handle(
                    sub,
                    context=inherited,
                    thread_id=thread_id,
                    _depth=depth,
                    _lineage=lineage,
                )
                async def _relay_child_events() -> None:
                    async for child_event in sub_handle.events:
                        if child_event.type != "tool_completed":
                            continue
                        payload = {
                            **child_event.data,
                            "subagent_name": sub.name,
                            "agent_name": child_event.data.get("agent_name", sub.name)
                            if isinstance(child_event.data, dict)
                            else sub.name,
                        }
                        await self._emit(
                            handle,
                            memory,
                            AgentRunEvent(
                                type="tool_completed",
                                run_id=run_id,
                                thread_id=thread_id,
                                state="running",
                                step=step,
                                data=payload,
                            ),
                            user_id=user_id,
                        )

                relay_task = asyncio.create_task(_relay_child_events())
                try:
                    out = await sub_handle.await_result()
                finally:
                    await asyncio.gather(relay_task, return_exceptions=True)
                if out is None:
                    raise SubagentExecutionError(f"Subagent '{sub.name}' cancelled")

                latency_ms = (time.time() - started) * 1000.0
                await self._emit(
                    handle,
                    memory,
                    AgentRunEvent(
                        type="subagent_completed",
                        run_id=run_id,
                        thread_id=thread_id,
                        state="running",
                        step=step,
                        data={
                            "subagent_name": sub.name,
                            "success": True,
                            "latency_ms": latency_ms,
                        },
                    ),
                    user_id=user_id,
                )
                return AgentInvocationResponse(
                    run_id=request.run_id,
                    thread_id=request.thread_id,
                    conversation_id=request.conversation_id,
                    correlation_id=request.correlation_id,
                    idempotency_key=request.idempotency_key,
                    source_agent=sub.name,
                    target_agent=request.source_agent,
                    success=True,
                    output={
                        "final_text": out.final_text,
                        "state": out.state,
                        "run_id": out.run_id,
                        "tool_executions": [
                            {
                                "tool_name": row.tool_name,
                                "tool_call_id": row.tool_call_id,
                                "success": row.success,
                                "output": row.output,
                                "error": row.error,
                                "latency_ms": row.latency_ms,
                                "agent_name": row.agent_name or sub.name,
                                "agent_depth": row.agent_depth
                                if isinstance(row.agent_depth, int)
                                else depth,
                                "agent_path": row.agent_path or sub.name,
                            }
                            for row in out.tool_executions
                        ],
                    },
                    metadata={"latency_ms": latency_ms},
                )
            except Exception as exc:
                latency_ms = (time.time() - started) * 1000.0
                error_text = str(exc)
                await self._emit(
                    handle,
                    memory,
                    AgentRunEvent(
                        type="subagent_completed",
                        run_id=run_id,
                        thread_id=thread_id,
                        state="running",
                        step=step,
                        data={
                            "subagent_name": sub.name,
                            "success": False,
                            "error": error_text,
                            "latency_ms": latency_ms,
                        },
                    ),
                    user_id=user_id,
                )
                return AgentInvocationResponse(
                    run_id=request.run_id,
                    thread_id=request.thread_id,
                    conversation_id=request.conversation_id,
                    correlation_id=request.correlation_id,
                    idempotency_key=request.idempotency_key,
                    source_agent=sub.name,
                    target_agent=request.source_agent,
                    success=False,
                    error=error_text,
                    metadata={"retryable": True, "latency_ms": latency_ms},
                )

        protocol = InternalA2AProtocol(dispatch=_dispatch)

        def _request_factory(
            node: DelegationNode,
            payload: dict[str, JSONValue],
            attempt: int,
        ) -> AgentInvocationRequest:
            node_correlation = f"{run_id}:{step}:{node.node_id}"
            return AgentInvocationRequest(
                run_id=run_id,
                thread_id=thread_id,
                conversation_id=f"{run_id}:{thread_id}",
                correlation_id=node_correlation,
                idempotency_key=f"{run_id}:{step}:{node.node_id}",
                causation_id=f"{run_id}:{step}",
                source_agent=agent.name,
                target_agent=node.target_agent,
                payload=payload,
                metadata={
                    "step": step,
                    "node_id": node.node_id,
                    "attempt": attempt,
                    "parallel": parallel,
                },
                timeout_s=node.timeout_s,
            )

        result, audit_rows = await engine.execute(
            plan=plan,
            available_targets=set(index.keys()),
            protocol=protocol,
            request_factory=_request_factory,
            cancel_requested=handle.is_cancel_requested,
            interrupt_requested=handle.is_interrupt_requested,
        )
        self._telemetry_counter(
            obs_contracts.METRIC_AGENT_SUBAGENT_NODES_TOTAL,
            value=max(0, len(result.ordered_outputs)),
            attributes={
                "result": result.final_status,
            },
        )

        for row in audit_rows:
            await self._emit(
                handle,
                memory,
                AgentRunEvent(
                    type="warning",
                    run_id=run_id,
                    thread_id=thread_id,
                    state="running",
                    step=step,
                    data=row,
                    message="Ignored late subagent response after cancellation",
                ),
                user_id=user_id,
            )

        for dead_letter in protocol.dead_letters():
            self._telemetry_counter(
                obs_contracts.METRIC_AGENT_SUBAGENT_DEAD_LETTERS_TOTAL,
                value=1,
                attributes={
                    "target_agent": dead_letter.request.target_agent,
                },
            )
            await self._emit(
                handle,
                memory,
                AgentRunEvent(
                    type="warning",
                    run_id=run_id,
                    thread_id=thread_id,
                    state="running",
                    step=step,
                    message="Subagent delivery exhausted retry budget",
                    data={
                        "node_correlation_id": dead_letter.request.correlation_id,
                        "target_agent": dead_letter.request.target_agent,
                        "attempts": dead_letter.attempts,
                        "error": dead_letter.error,
                    },
                ),
                user_id=user_id,
            )

        records: list[SubagentExecutionRecord] = []
        tool_records: list[ToolExecutionRecord] = []
        bridge_parts: list[str] = []
        for node_output in result.ordered_outputs:
            latency_ms = float(node_output.finished_at_ms - node_output.started_at_ms)
            self._telemetry_histogram(
                obs_contracts.METRIC_AGENT_SUBAGENT_NODE_LATENCY_MS,
                value=latency_ms,
                attributes={
                    "target_agent": node_output.target_agent,
                    "status": node_output.status,
                },
            )
            text_output: str | None = None
            if isinstance(node_output.output, dict):
                final_text = node_output.output.get("final_text")
                if isinstance(final_text, str):
                    text_output = final_text
                raw_tool_execs = node_output.output.get("tool_executions")
                if isinstance(raw_tool_execs, list):
                    for raw_row in raw_tool_execs:
                        if not isinstance(raw_row, dict):
                            continue
                        tool_name = raw_row.get("tool_name")
                        if not isinstance(tool_name, str):
                            continue
                        tool_records.append(
                            ToolExecutionRecord(
                                tool_name=tool_name,
                                tool_call_id=raw_row.get("tool_call_id")
                                if isinstance(raw_row.get("tool_call_id"), str)
                                else None,
                                success=bool(raw_row.get("success", False)),
                                output=raw_row.get("output"),
                                error=raw_row.get("error")
                                if isinstance(raw_row.get("error"), str)
                                else None,
                                latency_ms=raw_row.get("latency_ms")
                                if isinstance(raw_row.get("latency_ms"), (int, float))
                                else None,
                                agent_name=raw_row.get("agent_name")
                                if isinstance(raw_row.get("agent_name"), str)
                                else node_output.target_agent,
                                agent_depth=raw_row.get("agent_depth")
                                if isinstance(raw_row.get("agent_depth"), int)
                                else depth,
                                agent_path=(
                                    f"{agent.name}>"
                                    f"{raw_row.get('agent_path')}"
                                )
                                if isinstance(raw_row.get("agent_path"), str)
                                else f"{agent.name}>{node_output.target_agent}",
                            )
                        )
            elif isinstance(node_output.output, str):
                text_output = node_output.output

            records.append(
                SubagentExecutionRecord(
                    subagent_name=node_output.target_agent,
                    success=node_output.success,
                    output_text=text_output,
                    error=node_output.error,
                    latency_ms=latency_ms,
                )
            )
            if node_output.success and text_output:
                bridge_parts.append(
                    f"Subagent '{node_output.target_agent}' result:\n{text_output}"
                )
            elif node_output.error:
                bridge_parts.append(
                    f"Subagent '{node_output.target_agent}' failed: {node_output.error}"
                )

        return records, "\n\n".join(bridge_parts), tool_records

    def _delegation_plan_from_metadata(
        self,
        metadata: dict[str, JSONValue] | None,
    ) -> DelegationPlan | None:
        """Parse an optional router-provided delegation plan payload."""
        if not isinstance(metadata, dict):
            return None
        raw_plan = metadata.get("delegation_plan")
        if not isinstance(raw_plan, dict):
            return None

        raw_nodes = raw_plan.get("nodes")
        if not isinstance(raw_nodes, list):
            return None

        parsed_nodes: list[DelegationNode] = []
        for row in raw_nodes:
            if not isinstance(row, dict):
                continue
            node_id = row.get("node_id")
            target = row.get("target_agent")
            if not isinstance(node_id, str) or not node_id.strip():
                continue
            if not isinstance(target, str) or not target.strip():
                continue

            input_binding: dict[str, JSONValue] = {}
            raw_binding = row.get("input_binding")
            if isinstance(raw_binding, dict):
                input_binding = {
                    str(key): json_value_from_tool_result(value)
                    for key, value in raw_binding.items()
                }

            retry = row.get("retry_policy")
            retry_policy = RetryPolicy()
            if isinstance(retry, dict):
                retry_policy = RetryPolicy(
                    max_attempts=max(1, int(retry.get("max_attempts", 1))),
                    backoff_base_s=max(0.0, float(retry.get("backoff_base_s", 0.25))),
                    max_backoff_s=max(0.0, float(retry.get("max_backoff_s", 5.0))),
                    jitter_s=max(0.0, float(retry.get("jitter_s", 0.0))),
                )

            timeout_s = row.get("timeout_s")
            parsed_nodes.append(
                DelegationNode(
                    node_id=node_id.strip(),
                    target_agent=target.strip(),
                    input_binding=input_binding,
                    timeout_s=float(timeout_s)
                    if isinstance(timeout_s, (int, float))
                    else 120.0,
                    retry_policy=retry_policy,
                    required=bool(row.get("required", True)),
                )
            )

        if not parsed_nodes:
            return None

        parsed_edges: list[DelegationEdge] = []
        raw_edges = raw_plan.get("edges")
        if isinstance(raw_edges, list):
            for row in raw_edges:
                if not isinstance(row, dict):
                    continue
                source = row.get("from_node")
                target = row.get("to_node")
                if not isinstance(source, str) or not isinstance(target, str):
                    continue
                raw_map = row.get("output_key_map")
                key_map: dict[str, str] = {}
                if isinstance(raw_map, dict):
                    key_map = {str(key): str(value) for key, value in raw_map.items()}
                parsed_edges.append(
                    DelegationEdge(
                        from_node=source,
                        to_node=target,
                        output_key_map=key_map,
                    )
                )

        join_policy = raw_plan.get("join_policy")
        max_parallelism = raw_plan.get("max_parallelism")
        quorum = raw_plan.get("quorum")
        return DelegationPlan(
            nodes=parsed_nodes,
            edges=parsed_edges,
            join_policy=(
                join_policy
                if isinstance(join_policy, str)
                and join_policy
                in {
                    "all_required",
                    "allow_optional_failures",
                    "first_success",
                    "quorum",
                }
                else "all_required"
            ),
            max_parallelism=(
                max(1, int(max_parallelism))
                if isinstance(max_parallelism, (int, float))
                else 1
            ),
            quorum=int(quorum) if isinstance(quorum, (int, float)) else None,
        )

    async def _call_router(
        self,
        agent: BaseAgent,
        *,
        run_id: str,
        thread_id: str,
        step: int,
        context: dict[str, Any],
        messages: list[Message],
    ) -> RouterDecision:
        """
        Invoke configured subagent router and normalize response.

        Args:
            agent: Agent containing optional router callback.
            run_id: Current run identifier.
            thread_id: Current thread identifier.
            step: Current step index.
            context: Current run context.
            messages: Current transcript messages.

        Returns:
            Router decision; empty decision when no router is configured.

        Raises:
            SubagentRoutingError: If router returns invalid payload type.
        """
        router = agent.subagent_router
        if router is None:
            return RouterDecision()
        raw = router(
            RouterInput(
                run_id=run_id,
                thread_id=thread_id,
                step=step,
                context={
                    str(k): json_value_from_tool_result(v) for k, v in context.items()
                },
                messages=[
                    to_message_payload(
                        msg.role,
                        msg.content
                        if isinstance(msg.content, str)
                        else json.dumps(msg.content, ensure_ascii=True),
                    )
                    for msg in messages
                ],
            )
        )
        if inspect.isawaitable(raw):
            raw = await raw
        if not isinstance(raw, RouterDecision):
            raise SubagentRoutingError("subagent_router must return RouterDecision")
        return raw

    async def _evaluate_policy(
        self,
        *,
        agent: BaseAgent,
        event: PolicyEvent,
        handle: _RunHandle | None = None,
        memory: MemoryStore | None = None,
        user_id: str | None = None,
        state: AgentState = "running",
    ) -> PolicyDecision:
        """
        Evaluate policy engine and policy roles for runtime event.

        Args:
            agent: Active agent.
            event: Policy event payload under evaluation.
            handle: Optional run handle for policy audit events.
            memory: Optional memory store for policy audit persistence.
            user_id: Optional user id for emitted audit events.
            state: Current run state for audit events.

        Returns:
            Final policy decision.

        Raises:
            AgentExecutionError: If policy engine evaluation crashes.
        """
        decision = PolicyDecision(action="allow")

        engine = agent.policy_engine or self._policy_engine
        if engine is not None:
            try:
                evaluation = engine.evaluate(event)
                engine_decision = (
                    evaluation.decision
                    if hasattr(evaluation, "decision")
                    else evaluation
                )
            except Exception as e:
                raise AgentExecutionError(
                    f"Policy engine evaluation failed: {e}"
                ) from e
            if isinstance(engine_decision, RulePolicyDecision):
                decision = engine_decision
                if decision.action in {
                    "deny",
                    "defer",
                    "request_approval",
                    "request_user_input",
                }:
                    await self._emit_policy_audit(
                        handle=handle,
                        memory=memory,
                        event=event,
                        decision=decision,
                        user_id=user_id,
                        state=state,
                    )
                    return decision

        for role in agent.policy_roles:
            out = role(event)
            if inspect.isawaitable(out):
                out = await out
            if isinstance(out, PolicyDecision):
                decision = out
                if decision.action in {
                    "deny",
                    "defer",
                    "request_approval",
                    "request_user_input",
                }:
                    await self._emit_policy_audit(
                        handle=handle,
                        memory=memory,
                        event=event,
                        decision=decision,
                        user_id=user_id,
                        state=state,
                    )
                    return decision
        await self._emit_policy_audit(
            handle=handle,
            memory=memory,
            event=event,
            decision=decision,
            user_id=user_id,
            state=state,
        )
        return decision

    async def _emit_policy_audit(
        self,
        *,
        handle: _RunHandle | None,
        memory: MemoryStore | None,
        event: PolicyEvent,
        decision: PolicyDecision,
        user_id: str | None,
        state: AgentState,
    ) -> None:
        """
        Emit normalized policy-decision audit event.

        Args:
            handle: Active run handle.
            memory: Memory store used for persistence.
            event: Policy event that was evaluated.
            decision: Decision produced by policy layer.
            user_id: Optional user id for event persistence.
            state: Current run state.
        """
        if handle is None or memory is None:
            return
        await self._emit(
            handle,
            memory,
            AgentRunEvent(
                type="policy_decision",
                run_id=event.run_id,
                thread_id=event.thread_id,
                state=state,
                step=event.step,
                data={
                    "event_type": event.event_type,
                    "action": decision.action,
                    "reason": decision.reason,
                    "policy_id": decision.policy_id,
                    "matched_rules": [str(item) for item in decision.matched_rules],
                },
            ),
            user_id=user_id,
        )

    async def _request_approval(
        self,
        *,
        handle: _RunHandle,
        memory: MemoryStore,
        run_id: str,
        thread_id: str,
        step: int,
        reason: str,
        payload: dict[str, Any],
        user_id: str | None,
    ) -> bool:
        """
        Request approval and handle deferred pause/resume flow.

        Args:
            handle: Active run handle.
            memory: Memory store for pause/resume checkpoints.
            run_id: Current run identifier.
            thread_id: Current thread identifier.
            step: Current step index.
            reason: Human-readable reason for approval.
            payload: JSON-like request payload.
            user_id: Optional user id for event persistence.

        Returns:
            `True` when approved; otherwise `False`.
        """
        req = ApprovalRequest(
            run_id=run_id,
            thread_id=thread_id,
            step=step,
            reason=reason,
            payload={
                str(k): json_value_from_tool_result(v) for k, v in payload.items()
            },
        )
        decision = await self._interaction.request_approval(req)
        if isinstance(decision, ApprovalDecision):
            return decision.kind == "allow"

        await self._emit(
            handle,
            memory,
            AgentRunEvent(
                type="run_paused",
                run_id=run_id,
                thread_id=thread_id,
                state="paused",
                step=step,
                message="Waiting for deferred approval",
            ),
            user_id=user_id,
        )
        await self._persist_checkpoint(
            memory=memory,
            thread_id=thread_id,
            run_id=run_id,
            step=step,
            phase="paused",
            payload={"kind": "approval", "reason": reason},
        )
        wait_started_s = time.time()
        wait_span = self._telemetry_start_span(
            obs_contracts.SPAN_AGENT_INTERACTION_WAIT,
            attributes={
                "run_id": run_id,
                "thread_id": thread_id,
                "step": step,
                "kind": "approval",
            },
        )
        deferred = await self._interaction.await_deferred(
            decision.token,
            timeout_s=self.config.approval_timeout_s,
        )
        wait_latency_ms = (time.time() - wait_started_s) * 1000.0
        self._telemetry_histogram(
            obs_contracts.METRIC_AGENT_INTERACTION_WAIT_MS,
            value=wait_latency_ms,
            attributes={"kind": "approval"},
        )
        self._telemetry_counter(
            obs_contracts.METRIC_AGENT_INTERACTION_WAIT_TOTAL,
            value=1,
            attributes={
                "kind": "approval",
                "result": "resolved" if deferred is not None else "timeout",
            },
        )
        self._telemetry_end_span(
            wait_span,
            status="ok" if deferred is not None else "error",
            error=None if deferred is not None else "timeout",
            attributes={
                "kind": "approval",
                "wait_ms": wait_latency_ms,
            },
        )
        await self._emit(
            handle,
            memory,
            AgentRunEvent(
                type="run_resumed",
                run_id=run_id,
                thread_id=thread_id,
                state="running",
                step=step,
                message="Deferred approval resolved",
            ),
            user_id=user_id,
        )
        await self._persist_checkpoint(
            memory=memory,
            thread_id=thread_id,
            run_id=run_id,
            step=step,
            phase="resumed",
            payload={"kind": "approval"},
        )
        if isinstance(deferred, ApprovalDecision):
            return deferred.kind == "allow"
        return self.config.approval_fallback == "allow"

    async def _request_user_input(
        self,
        *,
        handle: _RunHandle,
        memory: MemoryStore,
        run_id: str,
        thread_id: str,
        step: int,
        prompt: str,
        payload: dict[str, Any],
        user_id: str | None,
    ) -> UserInputDecision:
        """
        Request user input and handle deferred pause/resume flow.

        Args:
            handle: Active run handle.
            memory: Memory store for pause/resume checkpoints.
            run_id: Current run identifier.
            thread_id: Current thread identifier.
            step: Current step index.
            prompt: Prompt shown to human operator.
            payload: JSON-like request payload.
            user_id: Optional user id for event persistence.

        Returns:
            Resolved user-input decision (or fallback on timeout).
        """
        req = UserInputRequest(
            run_id=run_id,
            thread_id=thread_id,
            step=step,
            prompt=prompt,
            payload={
                str(k): json_value_from_tool_result(v) for k, v in payload.items()
            },
        )
        decision = await self._interaction.request_user_input(req)
        if isinstance(decision, UserInputDecision):
            return decision

        await self._emit(
            handle,
            memory,
            AgentRunEvent(
                type="run_paused",
                run_id=run_id,
                thread_id=thread_id,
                state="paused",
                step=step,
                message="Waiting for deferred user input",
            ),
            user_id=user_id,
        )
        await self._persist_checkpoint(
            memory=memory,
            thread_id=thread_id,
            run_id=run_id,
            step=step,
            phase="paused",
            payload={"kind": "user_input", "prompt": prompt},
        )
        wait_started_s = time.time()
        wait_span = self._telemetry_start_span(
            obs_contracts.SPAN_AGENT_INTERACTION_WAIT,
            attributes={
                "run_id": run_id,
                "thread_id": thread_id,
                "step": step,
                "kind": "user_input",
            },
        )
        deferred = await self._interaction.await_deferred(
            decision.token,
            timeout_s=self.config.input_timeout_s,
        )
        wait_latency_ms = (time.time() - wait_started_s) * 1000.0
        self._telemetry_histogram(
            obs_contracts.METRIC_AGENT_INTERACTION_WAIT_MS,
            value=wait_latency_ms,
            attributes={"kind": "user_input"},
        )
        self._telemetry_counter(
            obs_contracts.METRIC_AGENT_INTERACTION_WAIT_TOTAL,
            value=1,
            attributes={
                "kind": "user_input",
                "result": "resolved" if deferred is not None else "timeout",
            },
        )
        self._telemetry_end_span(
            wait_span,
            status="ok" if deferred is not None else "error",
            error=None if deferred is not None else "timeout",
            attributes={
                "kind": "user_input",
                "wait_ms": wait_latency_ms,
            },
        )
        await self._emit(
            handle,
            memory,
            AgentRunEvent(
                type="run_resumed",
                run_id=run_id,
                thread_id=thread_id,
                state="running",
                step=step,
                message="Deferred user input resolved",
            ),
            user_id=user_id,
        )
        await self._persist_checkpoint(
            memory=memory,
            thread_id=thread_id,
            run_id=run_id,
            step=step,
            phase="resumed",
            payload={"kind": "user_input"},
        )
        if isinstance(deferred, UserInputDecision):
            return deferred
        return UserInputDecision(
            kind=self.config.input_fallback, reason="input_timeout"
        )

    def _is_defer_user_input(self, decision: PolicyDecision) -> bool:
        """
        Check whether a defer decision represents user-input interaction.

        Args:
            decision: Policy decision payload.

        Returns:
            `True` when defer marker maps to user input.
        """
        marker = decision.request_payload.get("interaction")
        if not isinstance(marker, str):
            return False
        return marker.strip().lower() in {"user_input", "input"}

    def _resolve_user_input_prompt(
        self,
        *,
        tool_name: str,
        decision: PolicyDecision,
    ) -> str:
        """
        Resolve prompt text used for user-input interaction requests.

        Args:
            tool_name: Tool name requiring user input.
            decision: Policy decision containing optional prompt metadata.

        Returns:
            Prompt string to show to interaction provider.
        """
        payload_prompt = decision.request_payload.get("prompt")
        if isinstance(payload_prompt, str) and payload_prompt.strip():
            return payload_prompt.strip()
        if isinstance(decision.reason, str) and decision.reason.strip():
            return decision.reason.strip()
        return f"Provide input for tool '{tool_name}'"
