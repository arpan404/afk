"""
Tests for agent lifecycle runtime helpers.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from afk.agents.errors import AgentCircuitOpenError, AgentExecutionError
from afk.agents.lifecycle.runtime import (
    CircuitBreaker,
    EffectJournal,
    _is_under,
    build_skill_manifest_prompt,
    checkpoint_latest_key,
    checkpoint_state_key,
    effect_state_key,
    json_hash,
    state_snapshot,
    to_message_payload,
    validate_state_transition,
)
from afk.agents.types.config import SkillRef
from afk.agents.types.policy import FailSafeConfig


def run_async(coro):
    """Helper to run async coroutines in synchronous tests."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# json_hash
# ---------------------------------------------------------------------------

class TestJsonHash:
    """Tests for json_hash determinism and output format."""

    def test_deterministic_same_payload(self):
        payload = {"a": 1, "b": "two"}
        assert json_hash(payload) == json_hash(payload)

    def test_different_payloads_differ(self):
        h1 = json_hash({"x": 1})
        h2 = json_hash({"x": 2})
        assert h1 != h2

    def test_returns_hex_string_length_64(self):
        h = json_hash({"key": "value"})
        assert isinstance(h, str)
        assert len(h) == 64
        # All hex characters
        int(h, 16)

    def test_key_order_irrelevant(self):
        h1 = json_hash({"b": 2, "a": 1})
        h2 = json_hash({"a": 1, "b": 2})
        assert h1 == h2


# ---------------------------------------------------------------------------
# checkpoint_state_key
# ---------------------------------------------------------------------------

class TestCheckpointStateKey:
    """Tests for checkpoint_state_key format."""

    def test_format(self):
        result = checkpoint_state_key("run-42", 7, "pre_tool")
        assert result == "checkpoint:run-42:7:pre_tool"

    def test_different_inputs(self):
        k1 = checkpoint_state_key("r1", 0, "init")
        k2 = checkpoint_state_key("r2", 1, "post")
        assert k1 != k2


# ---------------------------------------------------------------------------
# checkpoint_latest_key
# ---------------------------------------------------------------------------

class TestCheckpointLatestKey:
    """Tests for checkpoint_latest_key format."""

    def test_format(self):
        result = checkpoint_latest_key("run-99")
        assert result == "checkpoint:run-99:latest"


# ---------------------------------------------------------------------------
# effect_state_key
# ---------------------------------------------------------------------------

class TestEffectStateKey:
    """Tests for effect_state_key format."""

    def test_format(self):
        result = effect_state_key("run-1", 3, "tc-abc")
        assert result == "effect:run-1:3:tc-abc"


# ---------------------------------------------------------------------------
# to_message_payload
# ---------------------------------------------------------------------------

class TestToMessagePayload:
    """Tests for to_message_payload helper."""

    def test_returns_expected_dict(self):
        result = to_message_payload("user", "hello world")
        assert result == {"role": "user", "text": "hello world"}

    def test_assistant_role(self):
        result = to_message_payload("assistant", "reply")
        assert result == {"role": "assistant", "text": "reply"}


# ---------------------------------------------------------------------------
# build_skill_manifest_prompt
# ---------------------------------------------------------------------------

class TestBuildSkillManifestPrompt:
    """Tests for build_skill_manifest_prompt."""

    def test_empty_skills_returns_empty_string(self):
        assert build_skill_manifest_prompt([]) == ""

    def test_non_empty_returns_formatted_string(self):
        skill = SkillRef(
            name="deploy",
            description="Deploy the service",
            root_dir="/skills",
            skill_md_path="/skills/deploy/SKILL.md",
            checksum="abc123",
        )
        result = build_skill_manifest_prompt([skill])
        assert "Skills are enabled for this run." in result
        assert "deploy" in result
        assert "Deploy the service" in result
        assert "/skills/deploy/SKILL.md" in result
        assert "abc123" in result

    def test_multiple_skills(self):
        skills = [
            SkillRef(
                name="build",
                description="Build project",
                root_dir="/s",
                skill_md_path="/s/build/SKILL.md",
            ),
            SkillRef(
                name="test",
                description="Run tests",
                root_dir="/s",
                skill_md_path="/s/test/SKILL.md",
            ),
        ]
        result = build_skill_manifest_prompt(skills)
        assert "build" in result
        assert "test" in result
        # Skills with no checksum should show n/a
        assert "n/a" in result

    def test_no_checksum_shows_na(self):
        skill = SkillRef(
            name="lint",
            description="Lint code",
            root_dir="/s",
            skill_md_path="/s/lint/SKILL.md",
        )
        result = build_skill_manifest_prompt([skill])
        assert "n/a" in result


# ---------------------------------------------------------------------------
# validate_state_transition
# ---------------------------------------------------------------------------

class TestValidateStateTransition:
    """Tests for validate_state_transition."""

    def test_identity_returns_same(self):
        assert validate_state_transition("running", "running") == "running"
        assert validate_state_transition("pending", "pending") == "pending"
        assert validate_state_transition("completed", "completed") == "completed"

    @pytest.mark.parametrize(
        "current,target",
        [
            ("pending", "running"),
            ("running", "completed"),
            ("running", "failed"),
            ("running", "paused"),
            ("paused", "running"),
            ("running", "cancelled"),
        ],
    )
    def test_valid_transitions(self, current, target):
        result = validate_state_transition(current, target)
        assert result == target

    @pytest.mark.parametrize(
        "current,target",
        [
            ("completed", "running"),
            ("failed", "running"),
            ("cancelled", "running"),
        ],
    )
    def test_invalid_transitions_raise(self, current, target):
        with pytest.raises(AgentExecutionError, match="Invalid state transition"):
            validate_state_transition(current, target)


# ---------------------------------------------------------------------------
# EffectJournal
# ---------------------------------------------------------------------------

class TestEffectJournal:
    """Tests for EffectJournal get/put operations."""

    def test_get_unknown_key_returns_none(self):
        journal = EffectJournal()
        assert journal.get("run-1", 0, "tc-unknown") is None

    def test_put_then_get(self):
        journal = EffectJournal()
        journal.put(
            "run-1",
            0,
            "tc-1",
            input_hash="in-hash",
            output_hash="out-hash",
            output={"result": "ok"},
            success=True,
        )
        record = journal.get("run-1", 0, "tc-1")
        assert record is not None
        assert record["input_hash"] == "in-hash"
        assert record["output_hash"] == "out-hash"
        assert record["output"] == {"result": "ok"}
        assert record["success"] is True

    def test_multiple_puts_independent(self):
        journal = EffectJournal()
        journal.put(
            "run-1", 0, "tc-a",
            input_hash="h1", output_hash="h2",
            output="a", success=True,
        )
        journal.put(
            "run-1", 0, "tc-b",
            input_hash="h3", output_hash="h4",
            output="b", success=False,
        )
        rec_a = journal.get("run-1", 0, "tc-a")
        rec_b = journal.get("run-1", 0, "tc-b")
        assert rec_a["output"] == "a"
        assert rec_b["output"] == "b"
        assert rec_a["success"] is True
        assert rec_b["success"] is False


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    """Tests for CircuitBreaker async operations."""

    def _make_breaker(self, threshold=3, cooldown=1.0):
        config = FailSafeConfig(
            breaker_failure_threshold=threshold,
            breaker_cooldown_s=cooldown,
        )
        return CircuitBreaker(config=config)

    def test_record_success_clears_failures(self):
        breaker = self._make_breaker()

        async def _test():
            await breaker.record_failure("dep-a")
            await breaker.record_failure("dep-a")
            breaker.record_success("dep-a")
            # Should not raise because failures were cleared
            await breaker.ensure_closed("dep-a")

        run_async(_test())

    def test_record_failure_adds(self):
        breaker = self._make_breaker(threshold=5)

        async def _test():
            await breaker.record_failure("dep-x")
            await breaker.record_failure("dep-x")
            # Still below threshold, should not raise
            await breaker.ensure_closed("dep-x")

        run_async(_test())

    def test_ensure_closed_passes_below_threshold(self):
        breaker = self._make_breaker(threshold=3)

        async def _test():
            await breaker.record_failure("dep-1")
            await breaker.record_failure("dep-1")
            # 2 failures, threshold is 3 -- should pass
            await breaker.ensure_closed("dep-1")

        run_async(_test())

    def test_ensure_closed_raises_at_threshold(self):
        breaker = self._make_breaker(threshold=3)

        async def _test():
            await breaker.record_failure("dep-1")
            await breaker.record_failure("dep-1")
            await breaker.record_failure("dep-1")
            with pytest.raises(AgentCircuitOpenError, match="Circuit open"):
                await breaker.ensure_closed("dep-1")

        run_async(_test())

    def test_different_keys_independent(self):
        breaker = self._make_breaker(threshold=2)

        async def _test():
            await breaker.record_failure("dep-a")
            await breaker.record_failure("dep-a")
            # dep-a is at threshold, dep-b should be fine
            await breaker.ensure_closed("dep-b")
            with pytest.raises(AgentCircuitOpenError):
                await breaker.ensure_closed("dep-a")

        run_async(_test())

    def test_ensure_closed_auto_prunes_stale_entries(self):
        breaker = self._make_breaker(threshold=3, cooldown=0.1)

        async def _test():
            await breaker.record_failure("dep-s")
            await breaker.record_failure("dep-s")
            await breaker.record_failure("dep-s")
            # Wait for cooldown to expire
            await asyncio.sleep(0.15)
            # Stale entries should be pruned, so ensure_closed should pass
            await breaker.ensure_closed("dep-s")

        run_async(_test())


# ---------------------------------------------------------------------------
# state_snapshot
# ---------------------------------------------------------------------------

class TestStateSnapshot:
    """Tests for state_snapshot helper."""

    def test_returns_dict_with_all_keys(self):
        now = time.time()
        snap = state_snapshot(
            state="running",
            step=3,
            llm_calls=5,
            tool_calls=10,
            started_at_s=now,
        )
        expected_keys = {
            "state",
            "step",
            "llm_calls",
            "tool_calls",
            "started_at_s",
            "elapsed_s",
            "requested_model",
            "normalized_model",
            "provider_adapter",
            "total_cost_usd",
            "replayed_effect_count",
        }
        assert set(snap.keys()) == expected_keys

    def test_state_and_step_values(self):
        now = time.time()
        snap = state_snapshot(
            state="paused",
            step=7,
            llm_calls=2,
            tool_calls=3,
            started_at_s=now,
        )
        assert snap["state"] == "paused"
        assert snap["step"] == 7
        assert snap["llm_calls"] == 2
        assert snap["tool_calls"] == 3

    def test_elapsed_is_non_negative(self):
        started = time.time() - 1.0  # 1 second ago
        snap = state_snapshot(
            state="running",
            step=0,
            llm_calls=0,
            tool_calls=0,
            started_at_s=started,
        )
        assert snap["elapsed_s"] >= 0

    def test_optional_fields_default_none(self):
        snap = state_snapshot(
            state="running",
            step=0,
            llm_calls=0,
            tool_calls=0,
            started_at_s=time.time(),
        )
        assert snap["requested_model"] is None
        assert snap["normalized_model"] is None
        assert snap["provider_adapter"] is None
        assert snap["total_cost_usd"] is None
        assert snap["replayed_effect_count"] == 0

    def test_custom_optional_fields(self):
        snap = state_snapshot(
            state="completed",
            step=10,
            llm_calls=20,
            tool_calls=50,
            started_at_s=time.time(),
            requested_model="gpt-4",
            normalized_model="openai/gpt-4",
            provider_adapter="openai",
            total_cost_usd=1.50,
            replayed_effect_count=3,
        )
        assert snap["requested_model"] == "gpt-4"
        assert snap["normalized_model"] == "openai/gpt-4"
        assert snap["provider_adapter"] == "openai"
        assert snap["total_cost_usd"] == 1.50
        assert snap["replayed_effect_count"] == 3


# ---------------------------------------------------------------------------
# _is_under
# ---------------------------------------------------------------------------

class TestIsUnder:
    """Tests for the _is_under path helper."""

    def test_inside_path_returns_true(self):
        root = Path("/home/user/skills")
        child = Path("/home/user/skills/deploy/SKILL.md")
        assert _is_under(child, root) is True

    def test_outside_path_returns_false(self):
        root = Path("/home/user/skills")
        outside = Path("/etc/passwd")
        assert _is_under(outside, root) is False

    def test_same_path_returns_true(self):
        p = Path("/home/user/skills")
        assert _is_under(p, p) is True

    def test_parent_is_not_under_child(self):
        parent = Path("/home/user")
        child = Path("/home/user/skills")
        assert _is_under(parent, child) is False
