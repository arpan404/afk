"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Common literals and aliases shared across agent type sub-modules.
"""

from __future__ import annotations

from typing import Literal

AgentState = Literal[
    "pending",
    "running",
    "paused",
    "cancelling",
    "cancelled",
    "degraded",
    "failed",
    "completed",
]

SubagentParallelismMode = Literal["single", "parallel", "configurable"]
FailurePolicy = Literal[
    "retry_then_fail",
    "retry_then_degrade",
    "fail_fast",
    "continue_with_error",
    "retry_then_continue",
    "continue",
    "fail_run",
    "skip_action",
]

InteractionMode = Literal["headless", "interactive", "external"]
DecisionKind = Literal["allow", "deny", "defer"]
PolicyAction = Literal[
    "allow",
    "deny",
    "defer",
    "request_approval",
    "request_user_input",
]
AgentEventType = Literal[
    "run_started",
    "step_started",
    "policy_decision",
    "llm_called",
    "text_delta",
    "llm_completed",
    "tool_batch_started",
    "tool_completed",
    "subagent_started",
    "subagent_completed",
    "run_paused",
    "run_resumed",
    "run_cancelled",
    "run_interrupted",
    "run_failed",
    "run_completed",
    "warning",
]
