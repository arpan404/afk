"""
Tests for agent type dataclasses, error hierarchy, and debugger config.
"""

from __future__ import annotations

import dataclasses

import pytest

from afk.agents.errors import (
    AgentBudgetExceededError,
    AgentCancelledError,
    AgentCheckpointCorruptionError,
    AgentCircuitOpenError,
    AgentConfigurationError,
    AgentError,
    AgentExecutionError,
    AgentInterruptedError,
    AgentLoopLimitError,
    AgentPausedError,
    AgentRetryableError,
    PromptAccessError,
    PromptResolutionError,
    PromptTemplateError,
    SkillAccessError,
    SkillCommandDeniedError,
    SkillResolutionError,
    SubagentExecutionError,
    SubagentRoutingError,
)
from afk.agents.types.config import (
    RouterDecision,
    RouterInput,
    SkillRef,
    SkillResolutionResult,
    SkillToolPolicy,
)
from afk.agents.types.policy import (
    AgentRunEvent,
    FailSafeConfig,
    PolicyDecision,
    PolicyEvent,
)
from afk.debugger.types import DebuggerConfig

# ---------------------------------------------------------------------------
# SkillRef
# ---------------------------------------------------------------------------

class TestSkillRef:
    """Tests for SkillRef frozen dataclass."""

    def test_fields_stored(self):
        ref = SkillRef(
            name="deploy",
            description="Deploy service",
            root_dir="/skills",
            skill_md_path="/skills/deploy/SKILL.md",
            checksum="abc123",
        )
        assert ref.name == "deploy"
        assert ref.description == "Deploy service"
        assert ref.root_dir == "/skills"
        assert ref.skill_md_path == "/skills/deploy/SKILL.md"
        assert ref.checksum == "abc123"

    def test_checksum_default_none(self):
        ref = SkillRef(
            name="test",
            description="desc",
            root_dir="/r",
            skill_md_path="/r/test/SKILL.md",
        )
        assert ref.checksum is None

    def test_has_slots(self):
        assert hasattr(SkillRef, "__slots__")

    def test_frozen(self):
        ref = SkillRef(
            name="x",
            description="d",
            root_dir="/r",
            skill_md_path="/r/x/SKILL.md",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ref.name = "changed"


# ---------------------------------------------------------------------------
# SkillResolutionResult
# ---------------------------------------------------------------------------

class TestSkillResolutionResult:
    """Tests for SkillResolutionResult."""

    def test_missing_skills_default_empty(self):
        skill = SkillRef(
            name="a", description="d", root_dir="/r", skill_md_path="/r/a/SKILL.md"
        )
        result = SkillResolutionResult(resolved_skills=[skill])
        assert result.missing_skills == []

    def test_holds_resolved_and_missing(self):
        skill = SkillRef(
            name="a", description="d", root_dir="/r", skill_md_path="/r/a/SKILL.md"
        )
        result = SkillResolutionResult(
            resolved_skills=[skill],
            missing_skills=["b", "c"],
        )
        assert len(result.resolved_skills) == 1
        assert result.resolved_skills[0].name == "a"
        assert result.missing_skills == ["b", "c"]


# ---------------------------------------------------------------------------
# SkillToolPolicy
# ---------------------------------------------------------------------------

class TestSkillToolPolicy:
    """Tests for SkillToolPolicy defaults, custom values, and frozen constraint."""

    def test_all_defaults(self):
        policy = SkillToolPolicy()
        assert policy.command_allowlist == []
        assert policy.deny_shell_operators is True
        assert policy.max_stdout_chars == 20_000
        assert policy.max_stderr_chars == 20_000
        assert policy.command_timeout_s == 30.0

    def test_custom_values(self):
        policy = SkillToolPolicy(
            command_allowlist=["npm", "python"],
            deny_shell_operators=False,
            max_stdout_chars=5000,
            max_stderr_chars=3000,
            command_timeout_s=60.0,
        )
        assert policy.command_allowlist == ["npm", "python"]
        assert policy.deny_shell_operators is False
        assert policy.max_stdout_chars == 5000
        assert policy.max_stderr_chars == 3000
        assert policy.command_timeout_s == 60.0

    def test_frozen(self):
        policy = SkillToolPolicy()
        with pytest.raises(dataclasses.FrozenInstanceError):
            policy.deny_shell_operators = False


# ---------------------------------------------------------------------------
# RouterInput
# ---------------------------------------------------------------------------

class TestRouterInput:
    """Tests for RouterInput frozen dataclass."""

    def test_fields_stored(self):
        ri = RouterInput(
            run_id="run-1",
            thread_id="thread-1",
            step=5,
            context={"key": "value"},
            messages=[{"role": "user", "text": "hello"}],
        )
        assert ri.run_id == "run-1"
        assert ri.thread_id == "thread-1"
        assert ri.step == 5
        assert ri.context == {"key": "value"}
        assert ri.messages == [{"role": "user", "text": "hello"}]

    def test_frozen_and_slots(self):
        ri = RouterInput(
            run_id="r",
            thread_id="t",
            step=0,
            context={},
            messages=[],
        )
        assert hasattr(RouterInput, "__slots__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ri.run_id = "changed"


# ---------------------------------------------------------------------------
# RouterDecision
# ---------------------------------------------------------------------------

class TestRouterDecision:
    """Tests for RouterDecision defaults and custom values."""

    def test_defaults(self):
        rd = RouterDecision()
        assert rd.targets == []
        assert rd.parallel is False
        assert rd.metadata == {}

    def test_custom_values(self):
        rd = RouterDecision(
            targets=["agent-a", "agent-b"],
            parallel=True,
            metadata={"reason": "load-balancing"},
        )
        assert rd.targets == ["agent-a", "agent-b"]
        assert rd.parallel is True
        assert rd.metadata == {"reason": "load-balancing"}


# ---------------------------------------------------------------------------
# AgentRunEvent
# ---------------------------------------------------------------------------

class TestAgentRunEvent:
    """Tests for AgentRunEvent required fields and defaults."""

    def test_required_fields(self):
        evt = AgentRunEvent(
            type="run_started",
            run_id="r1",
            thread_id="t1",
            state="running",
        )
        assert evt.type == "run_started"
        assert evt.run_id == "r1"
        assert evt.thread_id == "t1"
        assert evt.state == "running"

    def test_defaults(self):
        evt = AgentRunEvent(
            type="run_started",
            run_id="r1",
            thread_id="t1",
            state="running",
        )
        assert evt.step is None
        assert evt.message is None
        assert evt.data == {}
        assert evt.schema_version == "v1"

    def test_frozen(self):
        evt = AgentRunEvent(
            type="run_started",
            run_id="r1",
            thread_id="t1",
            state="running",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            evt.run_id = "changed"


# ---------------------------------------------------------------------------
# PolicyEvent
# ---------------------------------------------------------------------------

class TestPolicyEvent:
    """Tests for PolicyEvent required fields and defaults."""

    def test_required_fields(self):
        pe = PolicyEvent(
            event_type="tool_before_execute",
            run_id="r1",
            thread_id="t1",
            step=3,
            context={"mode": "headless"},
        )
        assert pe.event_type == "tool_before_execute"
        assert pe.run_id == "r1"
        assert pe.thread_id == "t1"
        assert pe.step == 3
        assert pe.context == {"mode": "headless"}

    def test_defaults(self):
        pe = PolicyEvent(
            event_type="tool_before_execute",
            run_id="r1",
            thread_id="t1",
            step=0,
            context={},
        )
        assert pe.tool_name is None
        assert pe.tool_args is None
        assert pe.subagent_name is None
        assert pe.metadata == {}


# ---------------------------------------------------------------------------
# PolicyDecision
# ---------------------------------------------------------------------------

class TestPolicyDecision:
    """Tests for PolicyDecision defaults."""

    def test_defaults(self):
        pd = PolicyDecision()
        assert pd.action == "allow"
        assert pd.reason is None
        assert pd.updated_tool_args is None
        assert pd.request_payload == {}
        assert pd.policy_id is None
        assert pd.matched_rules == []

    def test_custom_values(self):
        pd = PolicyDecision(
            action="deny",
            reason="blocked by policy",
            policy_id="pol-1",
            matched_rules=["rule-a", "rule-b"],
        )
        assert pd.action == "deny"
        assert pd.reason == "blocked by policy"
        assert pd.policy_id == "pol-1"
        assert pd.matched_rules == ["rule-a", "rule-b"]


# ---------------------------------------------------------------------------
# FailSafeConfig
# ---------------------------------------------------------------------------

class TestFailSafeConfig:
    """Tests for FailSafeConfig defaults."""

    def test_all_defaults(self):
        cfg = FailSafeConfig()
        assert cfg.llm_failure_policy == "retry_then_fail"
        assert cfg.tool_failure_policy == "continue_with_error"
        assert cfg.subagent_failure_policy == "continue"
        assert cfg.approval_denial_policy == "skip_action"
        assert cfg.max_steps == 20
        assert cfg.max_wall_time_s == 300.0
        assert cfg.max_llm_calls == 50
        assert cfg.max_tool_calls == 200
        assert cfg.max_parallel_tools == 16
        assert cfg.max_subagent_depth == 3
        assert cfg.max_subagent_fanout_per_step == 4
        assert cfg.max_total_cost_usd is None
        assert cfg.fallback_model_chain == []
        assert cfg.breaker_failure_threshold == 5
        assert cfg.breaker_cooldown_s == 30.0

    def test_custom_values(self):
        cfg = FailSafeConfig(
            max_steps=100,
            max_total_cost_usd=10.0,
            fallback_model_chain=["gpt-4", "gpt-3.5-turbo"],
            breaker_failure_threshold=10,
        )
        assert cfg.max_steps == 100
        assert cfg.max_total_cost_usd == 10.0
        assert cfg.fallback_model_chain == ["gpt-4", "gpt-3.5-turbo"]
        assert cfg.breaker_failure_threshold == 10


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class TestAgentErrorHierarchy:
    """Tests for the agent error class hierarchy."""

    def test_agent_error_is_exception(self):
        assert issubclass(AgentError, Exception)

    def test_agent_configuration_error(self):
        assert issubclass(AgentConfigurationError, AgentError)

    def test_agent_execution_error(self):
        assert issubclass(AgentExecutionError, AgentError)

    def test_agent_retryable_error(self):
        assert issubclass(AgentRetryableError, AgentExecutionError)

    def test_agent_loop_limit_error(self):
        assert issubclass(AgentLoopLimitError, AgentExecutionError)

    def test_agent_budget_exceeded_error(self):
        assert issubclass(AgentBudgetExceededError, AgentExecutionError)

    def test_agent_cancelled_error(self):
        assert issubclass(AgentCancelledError, AgentExecutionError)

    def test_agent_interrupted_error(self):
        assert issubclass(AgentInterruptedError, AgentExecutionError)

    def test_agent_paused_error(self):
        assert issubclass(AgentPausedError, AgentExecutionError)

    def test_subagent_routing_error(self):
        assert issubclass(SubagentRoutingError, AgentExecutionError)

    def test_subagent_execution_error(self):
        assert issubclass(SubagentExecutionError, AgentExecutionError)

    def test_skill_resolution_error(self):
        assert issubclass(SkillResolutionError, AgentConfigurationError)

    def test_skill_access_error(self):
        assert issubclass(SkillAccessError, AgentExecutionError)

    def test_skill_command_denied_error(self):
        assert issubclass(SkillCommandDeniedError, AgentExecutionError)

    def test_prompt_resolution_error(self):
        assert issubclass(PromptResolutionError, AgentConfigurationError)

    def test_prompt_access_error(self):
        assert issubclass(PromptAccessError, AgentExecutionError)

    def test_prompt_template_error(self):
        assert issubclass(PromptTemplateError, AgentExecutionError)

    def test_agent_checkpoint_corruption_error(self):
        assert issubclass(AgentCheckpointCorruptionError, AgentExecutionError)

    def test_agent_circuit_open_error(self):
        assert issubclass(AgentCircuitOpenError, AgentExecutionError)

    def test_errors_are_catchable_as_base(self):
        """All leaf errors should be catchable via AgentError."""
        for cls in (
            AgentConfigurationError,
            AgentExecutionError,
            AgentRetryableError,
            AgentLoopLimitError,
            AgentBudgetExceededError,
            AgentCancelledError,
            AgentInterruptedError,
            AgentPausedError,
            SubagentRoutingError,
            SubagentExecutionError,
            SkillResolutionError,
            SkillAccessError,
            SkillCommandDeniedError,
            PromptResolutionError,
            PromptAccessError,
            PromptTemplateError,
            AgentCheckpointCorruptionError,
            AgentCircuitOpenError,
        ):
            with pytest.raises(AgentError):
                raise cls("test")


# ---------------------------------------------------------------------------
# DebuggerConfig
# ---------------------------------------------------------------------------

class TestDebuggerConfig:
    """Tests for DebuggerConfig defaults, custom values, and frozen constraint."""

    def test_all_defaults(self):
        cfg = DebuggerConfig()
        assert cfg.enabled is True
        assert cfg.verbosity == "detailed"
        assert cfg.include_content is True
        assert cfg.redact_secrets is True
        assert cfg.max_payload_chars == 4000
        assert cfg.emit_timestamps is True
        assert cfg.emit_step_snapshots is True

    def test_custom_values(self):
        cfg = DebuggerConfig(
            enabled=False,
            verbosity="minimal",
            include_content=False,
            redact_secrets=False,
            max_payload_chars=1000,
            emit_timestamps=False,
            emit_step_snapshots=False,
        )
        assert cfg.enabled is False
        assert cfg.verbosity == "minimal"
        assert cfg.include_content is False
        assert cfg.redact_secrets is False
        assert cfg.max_payload_chars == 1000
        assert cfg.emit_timestamps is False
        assert cfg.emit_step_snapshots is False

    def test_frozen(self):
        cfg = DebuggerConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.enabled = False
