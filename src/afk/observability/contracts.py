"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Telemetry contract constants for AFK runtime observability.
"""

from __future__ import annotations

AGENT_RUN_EVENT = "agent.run.event"
AGENT_RUN_EVENTS_TOTAL = "agent.run.events.total"

SPAN_AGENT_RUN = "agent.run"
SPAN_AGENT_LLM_CALL = "agent.llm.call"
SPAN_AGENT_TOOL_BATCH = "agent.tool.batch"
SPAN_AGENT_SUBAGENT_BATCH = "agent.subagent.batch"
SPAN_AGENT_INTERACTION_WAIT = "agent.interaction.wait"

METRIC_AGENT_RUNS_TOTAL = "agent.runs.total"
METRIC_AGENT_RUN_DURATION_MS = "agent.run.duration_ms"

METRIC_AGENT_LLM_CALLS_TOTAL = "agent.llm.calls.total"
METRIC_AGENT_LLM_LATENCY_MS = "agent.llm.latency_ms"

METRIC_AGENT_TOOL_BATCHES_TOTAL = "agent.tool.batches.total"
METRIC_AGENT_TOOL_BATCH_LATENCY_MS = "agent.tool.batch.latency_ms"
METRIC_AGENT_TOOL_CALLS_TOTAL = "agent.tool.calls.total"
METRIC_AGENT_TOOL_CALL_LATENCY_MS = "agent.tool.call.latency_ms"

METRIC_AGENT_SUBAGENT_BATCHES_TOTAL = "agent.subagent.batches.total"
METRIC_AGENT_SUBAGENT_BATCH_LATENCY_MS = "agent.subagent.batch.latency_ms"
METRIC_AGENT_SUBAGENT_NODES_TOTAL = "agent.subagent.nodes.total"
METRIC_AGENT_SUBAGENT_NODE_LATENCY_MS = "agent.subagent.node.latency_ms"
METRIC_AGENT_SUBAGENT_DEAD_LETTERS_TOTAL = "agent.subagent.dead_letters.total"

METRIC_AGENT_INTERACTION_WAIT_TOTAL = "agent.interaction.wait.total"
METRIC_AGENT_INTERACTION_WAIT_MS = "agent.interaction.wait_ms"

METRIC_AGENT_BGTOOLS_DEFERRED_TOTAL = "agent.bgtools.deferred.total"
METRIC_AGENT_BGTOOLS_RESOLVED_TOTAL = "agent.bgtools.resolved.total"
METRIC_AGENT_BGTOOLS_FAILED_TOTAL = "agent.bgtools.failed.total"
METRIC_AGENT_BGTOOLS_EXPIRED_TOTAL = "agent.bgtools.expired.total"
METRIC_AGENT_BGTOOLS_RESOLVE_LATENCY_MS = "agent.bgtools.resolve_latency_ms"
