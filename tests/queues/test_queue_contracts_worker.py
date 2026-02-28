"""Tests for queue contracts, TaskItem, RetryPolicy, and worker edge cases."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from afk.queues import (
    ExecutionContractContext,
    ExecutionContractValidationError,
    InMemoryTaskQueue,
    JobDispatchExecutionContract,
    RunnerChatExecutionContract,
)
from afk.queues.types import (
    NEXT_ATTEMPT_AT_KEY,
    RETRY_BACKOFF_BASE_KEY,
    RETRY_BACKOFF_JITTER_KEY,
    RETRY_BACKOFF_MAX_KEY,
    RetryPolicy,
    TaskItem,
)
from afk.queues.contracts import EXECUTION_CONTRACT_KEY


def run_async(coro):
    return asyncio.run(coro)


# -----------------------------------------------------------------------
# TaskItem tests
# -----------------------------------------------------------------------


class TestTaskItemIsTerminal:
    def test_completed_is_terminal(self):
        task = TaskItem(agent_name="a", payload={}, status="completed")
        assert task.is_terminal is True

    def test_failed_is_terminal(self):
        task = TaskItem(agent_name="a", payload={}, status="failed")
        assert task.is_terminal is True

    def test_cancelled_is_terminal(self):
        task = TaskItem(agent_name="a", payload={}, status="cancelled")
        assert task.is_terminal is True

    def test_pending_is_not_terminal(self):
        task = TaskItem(agent_name="a", payload={}, status="pending")
        assert task.is_terminal is False

    def test_running_is_not_terminal(self):
        task = TaskItem(agent_name="a", payload={}, status="running")
        assert task.is_terminal is False

    def test_retrying_is_not_terminal(self):
        task = TaskItem(agent_name="a", payload={}, status="retrying")
        assert task.is_terminal is False


class TestTaskItemDuration:
    def test_duration_s_when_both_set(self):
        task = TaskItem(
            agent_name="a",
            payload={},
            started_at=100.0,
            completed_at=105.5,
        )
        assert task.duration_s == 5.5

    def test_duration_s_none_when_started_at_missing(self):
        task = TaskItem(
            agent_name="a",
            payload={},
            started_at=None,
            completed_at=105.5,
        )
        assert task.duration_s is None

    def test_duration_s_none_when_completed_at_missing(self):
        task = TaskItem(
            agent_name="a",
            payload={},
            started_at=100.0,
            completed_at=None,
        )
        assert task.duration_s is None

    def test_duration_s_none_when_both_missing(self):
        task = TaskItem(agent_name="a", payload={})
        assert task.duration_s is None


class TestTaskItemExecutionContract:
    def test_reads_from_metadata(self):
        task = TaskItem(
            agent_name="a",
            payload={},
            metadata={EXECUTION_CONTRACT_KEY: "runner.chat.v1"},
        )
        assert task.execution_contract == "runner.chat.v1"

    def test_returns_none_when_missing(self):
        task = TaskItem(agent_name="a", payload={}, metadata={})
        assert task.execution_contract is None

    def test_returns_none_for_empty_string(self):
        task = TaskItem(
            agent_name="a",
            payload={},
            metadata={EXECUTION_CONTRACT_KEY: ""},
        )
        assert task.execution_contract is None

    def test_returns_none_for_whitespace_only(self):
        task = TaskItem(
            agent_name="a",
            payload={},
            metadata={EXECUTION_CONTRACT_KEY: "   "},
        )
        assert task.execution_contract is None

    def test_returns_none_for_non_string(self):
        task = TaskItem(
            agent_name="a",
            payload={},
            metadata={EXECUTION_CONTRACT_KEY: 42},
        )
        assert task.execution_contract is None


class TestTaskItemSetExecutionContract:
    def test_sets_metadata_key(self):
        task = TaskItem(agent_name="a", payload={})
        task.set_execution_contract("job.dispatch.v1")
        assert task.metadata[EXECUTION_CONTRACT_KEY] == "job.dispatch.v1"

    def test_overwrites_existing(self):
        task = TaskItem(
            agent_name="a",
            payload={},
            metadata={EXECUTION_CONTRACT_KEY: "old.v1"},
        )
        task.set_execution_contract("new.v2")
        assert task.metadata[EXECUTION_CONTRACT_KEY] == "new.v2"


class TestTaskItemNextAttemptAt:
    def test_reads_float_from_metadata(self):
        task = TaskItem(
            agent_name="a",
            payload={},
            metadata={NEXT_ATTEMPT_AT_KEY: 1234567890.5},
        )
        assert task.next_attempt_at == 1234567890.5

    def test_reads_int_from_metadata(self):
        task = TaskItem(
            agent_name="a",
            payload={},
            metadata={NEXT_ATTEMPT_AT_KEY: 1000},
        )
        assert task.next_attempt_at == 1000.0

    def test_returns_none_when_missing(self):
        task = TaskItem(agent_name="a", payload={}, metadata={})
        assert task.next_attempt_at is None

    def test_returns_none_for_non_numeric(self):
        task = TaskItem(
            agent_name="a",
            payload={},
            metadata={NEXT_ATTEMPT_AT_KEY: "not-a-number"},
        )
        assert task.next_attempt_at is None

    def test_set_next_attempt_at_sets_value(self):
        task = TaskItem(agent_name="a", payload={})
        task.set_next_attempt_at(9999.0)
        assert task.metadata[NEXT_ATTEMPT_AT_KEY] == 9999.0

    def test_set_next_attempt_at_none_clears_key(self):
        task = TaskItem(
            agent_name="a",
            payload={},
            metadata={NEXT_ATTEMPT_AT_KEY: 5000.0},
        )
        task.set_next_attempt_at(None)
        assert NEXT_ATTEMPT_AT_KEY not in task.metadata

    def test_set_next_attempt_at_none_noop_when_absent(self):
        task = TaskItem(agent_name="a", payload={}, metadata={})
        task.set_next_attempt_at(None)
        assert NEXT_ATTEMPT_AT_KEY not in task.metadata


# -----------------------------------------------------------------------
# RetryPolicy tests
# -----------------------------------------------------------------------


class TestRetryPolicyAsMetadata:
    def test_serializes_correctly(self):
        policy = RetryPolicy(backoff_base_s=1.0, backoff_max_s=60.0, backoff_jitter_s=0.5)
        meta = policy.as_metadata()
        assert meta[RETRY_BACKOFF_BASE_KEY] == 1.0
        assert meta[RETRY_BACKOFF_MAX_KEY] == 60.0
        assert meta[RETRY_BACKOFF_JITTER_KEY] == 0.5

    def test_serializes_defaults(self):
        policy = RetryPolicy()
        meta = policy.as_metadata()
        assert meta[RETRY_BACKOFF_BASE_KEY] == 0.0
        assert meta[RETRY_BACKOFF_MAX_KEY] == 30.0
        assert meta[RETRY_BACKOFF_JITTER_KEY] == 0.0


class TestRetryPolicyFromMetadata:
    def test_parses_valid_metadata(self):
        meta = {
            RETRY_BACKOFF_BASE_KEY: 2.0,
            RETRY_BACKOFF_MAX_KEY: 120.0,
            RETRY_BACKOFF_JITTER_KEY: 1.5,
        }
        policy = RetryPolicy.from_metadata(meta)
        assert policy is not None
        assert policy.backoff_base_s == 2.0
        assert policy.backoff_max_s == 120.0
        assert policy.backoff_jitter_s == 1.5

    def test_parses_integer_values(self):
        meta = {
            RETRY_BACKOFF_BASE_KEY: 5,
            RETRY_BACKOFF_MAX_KEY: 30,
            RETRY_BACKOFF_JITTER_KEY: 0,
        }
        policy = RetryPolicy.from_metadata(meta)
        assert policy is not None
        assert policy.backoff_base_s == 5.0
        assert policy.backoff_max_s == 30.0
        assert policy.backoff_jitter_s == 0.0

    def test_returns_none_for_incomplete_metadata(self):
        # Only two of three keys present
        meta = {
            RETRY_BACKOFF_BASE_KEY: 1.0,
            RETRY_BACKOFF_MAX_KEY: 30.0,
        }
        assert RetryPolicy.from_metadata(meta) is None

    def test_returns_none_for_empty_metadata(self):
        assert RetryPolicy.from_metadata({}) is None

    def test_returns_none_for_non_numeric_values(self):
        meta = {
            RETRY_BACKOFF_BASE_KEY: "fast",
            RETRY_BACKOFF_MAX_KEY: "slow",
            RETRY_BACKOFF_JITTER_KEY: "random",
        }
        assert RetryPolicy.from_metadata(meta) is None

    def test_returns_none_for_mixed_types(self):
        meta = {
            RETRY_BACKOFF_BASE_KEY: 1.0,
            RETRY_BACKOFF_MAX_KEY: "not-a-number",
            RETRY_BACKOFF_JITTER_KEY: 0.5,
        }
        assert RetryPolicy.from_metadata(meta) is None

    def test_roundtrip_via_as_metadata(self):
        original = RetryPolicy(backoff_base_s=3.0, backoff_max_s=90.0, backoff_jitter_s=2.0)
        restored = RetryPolicy.from_metadata(original.as_metadata())
        assert restored is not None
        assert restored.backoff_base_s == original.backoff_base_s
        assert restored.backoff_max_s == original.backoff_max_s
        assert restored.backoff_jitter_s == original.backoff_jitter_s


# -----------------------------------------------------------------------
# JobDispatchExecutionContract tests
# -----------------------------------------------------------------------


class TestJobDispatchExecutionContract:
    def test_contract_id(self):
        contract = JobDispatchExecutionContract()
        assert contract.contract_id == "job.dispatch.v1"

    def test_requires_agent_is_false(self):
        contract = JobDispatchExecutionContract()
        assert contract.requires_agent is False

    def test_validates_job_type_is_non_empty_string(self):
        contract = JobDispatchExecutionContract()
        task = TaskItem(agent_name=None, payload={"job_type": ""})
        ctx = ExecutionContractContext(job_handlers={})

        async def scenario():
            try:
                await contract.execute(task, agent=None, worker_context=ctx)
                assert False, "Expected ExecutionContractValidationError"
            except ExecutionContractValidationError as exc:
                assert "payload.job_type" in str(exc)

        run_async(scenario())

    def test_raises_for_missing_job_type(self):
        contract = JobDispatchExecutionContract()
        task = TaskItem(agent_name=None, payload={})
        ctx = ExecutionContractContext(job_handlers={})

        async def scenario():
            try:
                await contract.execute(task, agent=None, worker_context=ctx)
                assert False, "Expected ExecutionContractValidationError"
            except ExecutionContractValidationError as exc:
                assert "payload.job_type" in str(exc)

        run_async(scenario())

    def test_raises_for_non_string_job_type(self):
        contract = JobDispatchExecutionContract()
        task = TaskItem(agent_name=None, payload={"job_type": 123})
        ctx = ExecutionContractContext(job_handlers={})

        async def scenario():
            try:
                await contract.execute(task, agent=None, worker_context=ctx)
                assert False, "Expected ExecutionContractValidationError"
            except ExecutionContractValidationError as exc:
                assert "payload.job_type" in str(exc)

        run_async(scenario())

    def test_raises_for_unknown_handler(self):
        contract = JobDispatchExecutionContract()
        task = TaskItem(
            agent_name=None,
            payload={"job_type": "unknown_handler", "arguments": {}},
        )
        ctx = ExecutionContractContext(job_handlers={})

        async def scenario():
            try:
                await contract.execute(task, agent=None, worker_context=ctx)
                assert False, "Expected ExecutionContractValidationError"
            except ExecutionContractValidationError as exc:
                assert "Unknown job handler" in str(exc)

        run_async(scenario())

    def test_handles_sync_handler(self):
        def sync_handler(arguments, *, task_item):
            _ = task_item
            return {"doubled": arguments["x"] * 2}

        contract = JobDispatchExecutionContract()
        task = TaskItem(
            agent_name=None,
            payload={"job_type": "double", "arguments": {"x": 5}},
        )
        ctx = ExecutionContractContext(job_handlers={"double": sync_handler})

        async def scenario():
            result = await contract.execute(task, agent=None, worker_context=ctx)
            assert result == {"doubled": 10}

        run_async(scenario())

    def test_handles_async_handler(self):
        async def async_handler(arguments, *, task_item):
            _ = task_item
            return {"tripled": arguments["x"] * 3}

        contract = JobDispatchExecutionContract()
        task = TaskItem(
            agent_name=None,
            payload={"job_type": "triple", "arguments": {"x": 4}},
        )
        ctx = ExecutionContractContext(job_handlers={"triple": async_handler})

        async def scenario():
            result = await contract.execute(task, agent=None, worker_context=ctx)
            assert result == {"tripled": 12}

        run_async(scenario())

    def test_raises_for_non_dict_arguments(self):
        contract = JobDispatchExecutionContract()
        task = TaskItem(
            agent_name=None,
            payload={"job_type": "my_job", "arguments": "not-a-dict"},
        )
        ctx = ExecutionContractContext(job_handlers={"my_job": lambda a, **kw: a})

        async def scenario():
            try:
                await contract.execute(task, agent=None, worker_context=ctx)
                assert False, "Expected ExecutionContractValidationError"
            except ExecutionContractValidationError as exc:
                assert "payload.arguments" in str(exc)

        run_async(scenario())


# -----------------------------------------------------------------------
# RunnerChatExecutionContract tests
# -----------------------------------------------------------------------


class TestRunnerChatExecutionContract:
    def test_contract_id(self):
        contract = RunnerChatExecutionContract()
        assert contract.contract_id == "runner.chat.v1"

    def test_requires_agent_is_true(self):
        contract = RunnerChatExecutionContract()
        assert contract.requires_agent is True

    def test_raises_when_agent_is_none(self):
        contract = RunnerChatExecutionContract()
        task = TaskItem(
            agent_name="demo",
            payload={"user_message": "hello", "context": {}},
        )
        ctx = ExecutionContractContext()

        async def scenario():
            try:
                await contract.execute(task, agent=None, worker_context=ctx)
                assert False, "Expected ExecutionContractValidationError"
            except ExecutionContractValidationError as exc:
                assert "requires an agent" in str(exc)

        run_async(scenario())

    def test_validates_user_message_type(self):
        """user_message must be str or None; an int should raise."""
        contract = RunnerChatExecutionContract()
        task = TaskItem(
            agent_name="demo",
            payload={"user_message": 12345, "context": {}},
        )
        ctx = ExecutionContractContext()

        async def scenario():
            # Provide a non-None agent sentinel so the agent-None check passes
            try:
                await contract.execute(
                    task, agent=object(), worker_context=ctx  # type: ignore[arg-type]
                )
                assert False, "Expected ExecutionContractValidationError"
            except ExecutionContractValidationError as exc:
                assert "payload.user_message" in str(exc)

        run_async(scenario())

    def test_validates_context_type(self):
        """context must be dict or None; a list should raise."""
        contract = RunnerChatExecutionContract()
        task = TaskItem(
            agent_name="demo",
            payload={"user_message": "hello", "context": [1, 2, 3]},
        )
        ctx = ExecutionContractContext()

        async def scenario():
            try:
                await contract.execute(
                    task, agent=object(), worker_context=ctx  # type: ignore[arg-type]
                )
                assert False, "Expected ExecutionContractValidationError"
            except ExecutionContractValidationError as exc:
                assert "payload.context" in str(exc)

        run_async(scenario())


# -----------------------------------------------------------------------
# ExecutionContractContext tests
# -----------------------------------------------------------------------


class TestExecutionContractContext:
    def test_default_job_handlers_is_empty(self):
        ctx = ExecutionContractContext()
        assert dict(ctx.job_handlers) == {}

    def test_accepts_custom_handlers(self):
        handler = lambda a, **kw: a  # noqa: E731
        ctx = ExecutionContractContext(job_handlers={"echo": handler})
        assert "echo" in ctx.job_handlers


# -----------------------------------------------------------------------
# _compute_retry_delay_s via InMemoryTaskQueue
# -----------------------------------------------------------------------


class TestComputeRetryDelay:
    def test_zero_base_returns_jitter_only(self):
        queue = InMemoryTaskQueue()
        policy = RetryPolicy(backoff_base_s=0.0, backoff_max_s=100.0, backoff_jitter_s=2.0)
        with patch("afk.queues.base.random", return_value=0.5):
            delay = queue._compute_retry_delay_s(1, policy=policy)  # noqa: SLF001
        # base=0 -> capped=0 -> delay = 0 + 0.5*2.0 = 1.0
        assert delay == 1.0

    def test_zero_base_zero_jitter(self):
        queue = InMemoryTaskQueue()
        policy = RetryPolicy(backoff_base_s=0.0, backoff_max_s=100.0, backoff_jitter_s=0.0)
        delay = queue._compute_retry_delay_s(5, policy=policy)  # noqa: SLF001
        assert delay == 0.0

    def test_non_zero_base_retry_count_1(self):
        queue = InMemoryTaskQueue()
        policy = RetryPolicy(backoff_base_s=2.0, backoff_max_s=100.0, backoff_jitter_s=0.0)
        with patch("afk.queues.base.random", return_value=0.0):
            delay = queue._compute_retry_delay_s(1, policy=policy)  # noqa: SLF001
        # base = 2.0 * 2^max(0, 1-1) = 2.0 * 2^0 = 2.0, capped at 100 -> 2.0
        assert delay == 2.0

    def test_non_zero_base_retry_count_3(self):
        queue = InMemoryTaskQueue()
        policy = RetryPolicy(backoff_base_s=1.0, backoff_max_s=100.0, backoff_jitter_s=0.0)
        with patch("afk.queues.base.random", return_value=0.0):
            delay = queue._compute_retry_delay_s(3, policy=policy)  # noqa: SLF001
        # base = 1.0 * 2^max(0, 3-1) = 1.0 * 4 = 4.0, capped at 100 -> 4.0
        assert delay == 4.0

    def test_caps_at_backoff_max_s(self):
        queue = InMemoryTaskQueue()
        policy = RetryPolicy(backoff_base_s=10.0, backoff_max_s=5.0, backoff_jitter_s=0.0)
        with patch("afk.queues.base.random", return_value=0.0):
            delay = queue._compute_retry_delay_s(1, policy=policy)  # noqa: SLF001
        # base = 10.0 * 2^0 = 10.0, capped at 5.0 -> 5.0
        assert delay == 5.0

    def test_caps_with_large_retry_count(self):
        queue = InMemoryTaskQueue()
        policy = RetryPolicy(backoff_base_s=1.0, backoff_max_s=30.0, backoff_jitter_s=0.0)
        with patch("afk.queues.base.random", return_value=0.0):
            delay = queue._compute_retry_delay_s(20, policy=policy)  # noqa: SLF001
        # base = 1.0 * 2^19 = huge, capped at 30.0
        assert delay == 30.0

    def test_jitter_adds_to_base(self):
        queue = InMemoryTaskQueue()
        policy = RetryPolicy(backoff_base_s=3.0, backoff_max_s=100.0, backoff_jitter_s=4.0)
        with patch("afk.queues.base.random", return_value=0.75):
            delay = queue._compute_retry_delay_s(1, policy=policy)  # noqa: SLF001
        # base = 3.0 * 2^0 = 3.0, jitter = 0.75*4.0 = 3.0, total = 6.0
        assert delay == 6.0

    def test_delay_is_always_non_negative(self):
        queue = InMemoryTaskQueue()
        policy = RetryPolicy(backoff_base_s=0.0, backoff_max_s=0.0, backoff_jitter_s=0.0)
        delay = queue._compute_retry_delay_s(0, policy=policy)  # noqa: SLF001
        assert delay >= 0.0
