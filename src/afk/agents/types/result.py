"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Result and execution-record types for agent runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...llms.types import JSONValue, LLMResponse, Usage
from ...tools import ToolResult


@dataclass(frozen=True, slots=True)
class UsageAggregate:
    """
    Aggregated token usage across LLM calls in a run.

    Attributes:
        input_tokens: Sum of prompt/input tokens.
        output_tokens: Sum of completion/output tokens.
        total_tokens: Sum of total token counts.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add_usage(self, usage: Usage) -> "UsageAggregate":
        """
        Return a new aggregate with additional usage applied.

        Args:
            usage: Usage payload from a single LLM response.

        Returns:
            New `UsageAggregate` with usage added.
        """
        return UsageAggregate(
            input_tokens=self.input_tokens + (usage.input_tokens or 0),
            output_tokens=self.output_tokens + (usage.output_tokens or 0),
            total_tokens=self.total_tokens + (usage.total_tokens or 0),
        )


@dataclass(frozen=True, slots=True)
class ToolExecutionRecord:
    """
    Normalized record for one tool execution.

    Attributes:
        tool_name: Executed tool name.
        tool_call_id: Provider/LLM tool-call identifier when available.
        success: Whether tool execution succeeded.
        output: JSON-safe tool output payload.
        error: Error message when execution failed.
        latency_ms: Execution latency in milliseconds.
        agent_name: Agent that executed the tool.
        agent_depth: Agent nesting depth where tool executed.
        agent_path: Agent lineage path for nested/subagent calls.
    """

    tool_name: str
    tool_call_id: str | None
    success: bool
    output: JSONValue | None = None
    error: str | None = None
    latency_ms: float | None = None
    agent_name: str | None = None
    agent_depth: int | None = None
    agent_path: str | None = None


@dataclass(frozen=True, slots=True)
class SubagentExecutionRecord:
    """
    Normalized record for one subagent execution.

    Attributes:
        subagent_name: Executed subagent name.
        success: Whether subagent run succeeded.
        output_text: Final text returned by subagent.
        error: Error message when subagent failed.
        latency_ms: Subagent execution latency in milliseconds.
    """

    subagent_name: str
    success: bool
    output_text: str | None = None
    error: str | None = None
    latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class SkillReadRecord:
    """
    Record of reading skill metadata or files.

    Attributes:
        skill_name: Skill name being read.
        path: Absolute path accessed by the runtime/tool.
        checksum: Optional content checksum for auditability.
    """

    skill_name: str
    path: str
    checksum: str | None = None


@dataclass(frozen=True, slots=True)
class CommandExecutionRecord:
    """
    Record for skill/runtime command execution.

    Attributes:
        command: Executed command as argv list.
        exit_code: Process exit code.
        stdout: Captured stdout content (possibly truncated).
        stderr: Captured stderr content (possibly truncated).
        denied: Whether execution was denied before process launch.
    """

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    denied: bool = False


@dataclass(frozen=True, slots=True)
class AgentResult:
    """
    Terminal result payload returned by runner/agent calls.

    Attributes:
        run_id: Run identifier.
        thread_id: Thread identifier.
        state: Terminal agent state.
        final_text: Final assistant text.
        requested_model: User-requested model identifier.
        normalized_model: Effective model identifier used in runtime.
        provider_adapter: Adapter/provider id used for execution.
        final_structured: Final structured output payload when available.
        llm_response: Final raw normalized LLM response.
        tool_executions: Ordered tool execution records.
        subagent_executions: Ordered subagent execution records.
        skills_used: Skill names enabled for this run.
        skill_reads: Skill file read audit records.
        skill_command_executions: Skill command execution records.
        usage_aggregate: Total token usage across LLM calls.
        total_cost_usd: Optional aggregated cost in USD.
        session_token: Optional provider session token for resume.
        checkpoint_token: Optional provider checkpoint token for resume.
        state_snapshot: Terminal runtime snapshot payload.
    """

    run_id: str
    thread_id: str
    state: "AgentState"
    final_text: str
    requested_model: str | None = None
    normalized_model: str | None = None
    provider_adapter: str | None = None
    final_structured: dict[str, JSONValue] | None = None
    llm_response: LLMResponse | None = None
    tool_executions: list[ToolExecutionRecord] = field(default_factory=list)
    subagent_executions: list[SubagentExecutionRecord] = field(default_factory=list)
    skills_used: list[str] = field(default_factory=list)
    skill_reads: list[SkillReadRecord] = field(default_factory=list)
    skill_command_executions: list[CommandExecutionRecord] = field(default_factory=list)
    usage_aggregate: UsageAggregate = field(default_factory=UsageAggregate)
    total_cost_usd: float | None = None
    session_token: str | None = None
    checkpoint_token: str | None = None
    state_snapshot: dict[str, JSONValue] = field(default_factory=dict)


# Avoid circular import — use string reference for AgentState
from .common import AgentState  # noqa: E402


def json_value_from_tool_result(value: Any) -> JSONValue:
    """
    Best-effort conversion of tool outputs to JSON-safe payloads.

    Args:
        value: Arbitrary tool output object.

    Returns:
        JSON-safe equivalent value. Unsupported objects are stringified with
        `repr(...)`.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [json_value_from_tool_result(v) for v in value]
    if isinstance(value, dict):
        out: dict[str, JSONValue] = {}
        for key, item in value.items():
            out[str(key)] = json_value_from_tool_result(item)
        return out
    return repr(value)


def tool_record_from_result(
    tool_name: str,
    tool_call_id: str | None,
    result: ToolResult[Any],
    *,
    latency_ms: float | None = None,
    agent_name: str | None = None,
    agent_depth: int | None = None,
    agent_path: str | None = None,
) -> ToolExecutionRecord:
    """
    Convert a `ToolResult` into a normalized execution record.

    Args:
        tool_name: Name of executed tool.
        tool_call_id: Tool-call identifier from LLM, when present.
        result: Raw tool execution result object.
        latency_ms: Optional measured latency in milliseconds.

    Returns:
        Normalized `ToolExecutionRecord` for event/result reporting.
    """
    return ToolExecutionRecord(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        success=result.success,
        output=json_value_from_tool_result(result.output),
        error=result.error_message,
        latency_ms=latency_ms,
        agent_name=agent_name,
        agent_depth=agent_depth,
        agent_path=agent_path,
    )
