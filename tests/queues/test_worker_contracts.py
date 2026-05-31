from __future__ import annotations

import asyncio

from afk.queues import (
    JOB_DISPATCH_CONTRACT,
    RUNNER_CHAT_CONTRACT,
    ExecutionContractContext,
    InMemoryTaskQueue,
    RetryPolicy,
    TaskItem,
    TaskWorker,
    WorkerMetrics,
)


def run_async(coro):
    return asyncio.run(coro)


def test_missing_contract_is_terminal_without_retry():
    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        task = TaskItem(
            agent_name="demo",
            payload={"user_message": "hello", "context": {}},
            metadata={},
            max_retries=5,
        )
        await queue.enqueue(task)

        worker = TaskWorker(queue, agents={"demo": object()})
        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        await worker._execute_task(running)  # noqa: SLF001

        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "failed"
        assert current.retry_count == 1
        assert "Missing execution contract metadata" in (current.error or "")
        assert queue.pending_count == 0

    run_async(scenario())


def test_cancelled_task_execution_marks_task_as_failed_non_retryable():
    class _SlowContract:
        contract_id = "slow.v1"
        requires_agent = False

        async def execute(
            self,
            task_item: TaskItem,
            *,
            agent,
            worker_context: ExecutionContractContext,
        ):
            _ = task_item
            _ = agent
            _ = worker_context
            await asyncio.sleep(1.0)
            return {"done": True}

    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        task = await queue.enqueue_contract(
            "slow.v1",
            payload={},
            agent_name=None,
            max_retries=3,
        )
        worker = TaskWorker(
            queue,
            agents={},
            execution_contracts={"slow.v1": _SlowContract()},
        )
        running = await queue.dequeue(timeout=0.1)
        assert running is not None

        handle = asyncio.create_task(worker._execute_task(running))
        await asyncio.sleep(0.1)
        handle.cancel()
        await handle

        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "failed"
        assert current.retry_count == 1
        assert current.error == "Task execution cancelled"

    run_async(scenario())


def test_unknown_contract_is_terminal_without_retry():
    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        task = await queue.enqueue_contract(
            "missing.contract.v1",
            payload={},
            agent_name=None,
            max_retries=5,
        )
        worker = TaskWorker(queue, agents={})

        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        await worker._execute_task(running)  # noqa: SLF001

        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "failed"
        assert current.retry_count == 1
        assert "Unknown execution contract" in (current.error or "")
        assert queue.pending_count == 0

    run_async(scenario())


def test_invalid_dispatch_payload_is_terminal_without_retry():
    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        task = await queue.enqueue_contract(
            JOB_DISPATCH_CONTRACT,
            payload={"arguments": {"a": 1}},
            agent_name=None,
            max_retries=5,
        )
        worker = TaskWorker(queue, agents={})

        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        await worker._execute_task(running)  # noqa: SLF001

        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "failed"
        assert current.retry_count == 1
        assert "payload.job_type" in (current.error or "")
        assert queue.pending_count == 0

    run_async(scenario())


def test_job_dispatch_contract_executes_non_agent_job_handler():
    async def scenario() -> None:
        queue = InMemoryTaskQueue()

        async def _sum_handler(arguments, *, task_item):
            _ = task_item
            return {"sum": int(arguments["a"]) + int(arguments["b"])}

        task = await queue.enqueue_contract(
            JOB_DISPATCH_CONTRACT,
            payload={"job_type": "sum", "arguments": {"a": 2, "b": 3}},
            agent_name=None,
        )
        worker = TaskWorker(queue, agents={}, job_handlers={"sum": _sum_handler})

        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        await worker._execute_task(running)  # noqa: SLF001

        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "completed"
        assert current.result == {
            "contract": JOB_DISPATCH_CONTRACT,
            "output": {"sum": 5},
        }

    run_async(scenario())


def test_job_dispatch_unknown_handler_is_terminal_without_retry():
    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        task = await queue.enqueue_contract(
            JOB_DISPATCH_CONTRACT,
            payload={"job_type": "missing", "arguments": {}},
            agent_name=None,
            max_retries=4,
        )
        worker = TaskWorker(queue, agents={}, job_handlers={})

        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        await worker._execute_task(running)  # noqa: SLF001

        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "failed"
        assert current.retry_count == 1
        assert "Unknown job handler" in (current.error or "")
        assert queue.pending_count == 0

    run_async(scenario())


def test_agent_required_contract_fails_when_agent_name_missing():
    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        task = await queue.enqueue_contract(
            RUNNER_CHAT_CONTRACT,
            payload={"user_message": "hello"},
            agent_name=None,
            max_retries=3,
        )
        worker = TaskWorker(queue, agents={})

        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        await worker._execute_task(running)  # noqa: SLF001

        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "failed"
        assert current.retry_count == 1
        assert "requires a non-empty task.agent_name" in (current.error or "")
        assert queue.pending_count == 0

    run_async(scenario())


def test_retryable_contract_runtime_error_uses_retry_budget():
    class _BoomContract:
        contract_id = "boom.v1"
        requires_agent = False

        async def execute(
            self,
            task_item: TaskItem,
            *,
            agent,
            worker_context: ExecutionContractContext,
        ):
            _ = task_item
            _ = agent
            _ = worker_context
            raise RuntimeError("boom")

    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        task = await queue.enqueue_contract(
            "boom.v1",
            payload={},
            agent_name=None,
            max_retries=2,
        )
        worker = TaskWorker(
            queue,
            agents={},
            execution_contracts={"boom.v1": _BoomContract()},
        )

        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        await worker._execute_task(running)  # noqa: SLF001

        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "retrying"
        assert current.retry_count == 1
        assert queue.pending_count == 1

    run_async(scenario())


def test_result_envelope_applies_to_agent_required_custom_contract():
    class _NeedsAgentContract:
        contract_id = "needs-agent.v1"
        requires_agent = True

        async def execute(
            self,
            task_item: TaskItem,
            *,
            agent,
            worker_context: ExecutionContractContext,
        ):
            _ = task_item
            _ = agent
            _ = worker_context
            return {"channel": "runner"}

    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        task = await queue.enqueue_contract(
            "needs-agent.v1",
            payload={},
            agent_name="demo",
        )
        worker = TaskWorker(
            queue,
            agents={"demo": object()},
            execution_contracts={"needs-agent.v1": _NeedsAgentContract()},
        )

        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        await worker._execute_task(running)  # noqa: SLF001

        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "completed"
        assert current.result == {
            "contract": "needs-agent.v1",
            "output": {"channel": "runner"},
        }

    run_async(scenario())


def test_worker_metrics_counters_are_emitted():
    class _Metrics(WorkerMetrics):
        def __init__(self) -> None:
            self.counts: dict[str, int] = {}

        def incr(self, name: str, value: int = 1, *, tags=None) -> None:
            _ = tags
            self.counts[name] = self.counts.get(name, 0) + value

    class _OkContract:
        contract_id = "ok.v1"
        requires_agent = False

        async def execute(
            self,
            task_item: TaskItem,
            *,
            agent,
            worker_context: ExecutionContractContext,
        ):
            _ = task_item
            _ = agent
            _ = worker_context
            return {"ok": True}

    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        metrics = _Metrics()
        await queue.enqueue_contract("ok.v1", payload={}, agent_name=None)
        worker = TaskWorker(
            queue,
            agents={},
            execution_contracts={"ok.v1": _OkContract()},
            metrics=metrics,
        )
        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        await worker._execute_task(running)  # noqa: SLF001

        assert metrics.counts.get("queue_worker_completed_total", 0) == 1

        failing = await queue.enqueue_contract(
            "missing.v1", payload={}, agent_name=None
        )
        running2 = await queue.dequeue(timeout=0.1)
        assert running2 is not None
        await worker._execute_task(running2)  # noqa: SLF001
        _ = failing
        assert metrics.counts.get("queue_worker_failed_non_retryable_total", 0) == 1

    run_async(scenario())


def test_per_contract_retry_policy_override_is_applied():
    class _BoomContract:
        contract_id = "boom-override.v1"
        requires_agent = False

        async def execute(
            self,
            task_item: TaskItem,
            *,
            agent,
            worker_context: ExecutionContractContext,
        ):
            _ = task_item
            _ = agent
            _ = worker_context
            raise RuntimeError("boom")

    async def scenario() -> None:
        queue = InMemoryTaskQueue(
            retry_backoff_base_s=0.0,
            retry_backoff_max_s=0.0,
            retry_backoff_jitter_s=0.0,
        )
        task = await queue.enqueue_contract(
            "boom-override.v1",
            payload={},
            agent_name=None,
            max_retries=1,
        )
        worker = TaskWorker(
            queue,
            agents={},
            execution_contracts={"boom-override.v1": _BoomContract()},
            retry_policies={
                "boom-override.v1": RetryPolicy(
                    backoff_base_s=0.2,
                    backoff_max_s=0.2,
                    backoff_jitter_s=0.0,
                )
            },
        )

        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        await worker._execute_task(running)  # noqa: SLF001

        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "retrying"
        assert current.next_attempt_at is not None
        assert current.next_attempt_at - current.created_at >= 0.15

    run_async(scenario())


def test_on_complete_callback_error_does_not_requeue_completed_task():
    class _OkContract:
        contract_id = "ok.v1"
        requires_agent = False

        async def execute(
            self,
            task_item: TaskItem,
            *,
            agent,
            worker_context: ExecutionContractContext,
        ):
            _ = task_item
            _ = agent
            _ = worker_context
            return {"ok": True}

    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        task = await queue.enqueue_contract(
            "ok.v1",
            payload={},
            agent_name=None,
            max_retries=3,
        )

        async def _bad_complete(_task: TaskItem) -> None:
            raise RuntimeError("on_complete boom")

        worker = TaskWorker(
            queue,
            agents={},
            execution_contracts={"ok.v1": _OkContract()},
            on_complete=_bad_complete,
        )

        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        await worker._execute_task(running)  # noqa: SLF001

        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "completed"
        assert current.retry_count == 0
        assert current.error is None
        assert current.result == {"contract": "ok.v1", "output": {"ok": True}}
        assert queue.pending_count == 0

    run_async(scenario())


def test_on_failure_callback_error_does_not_change_failure_transition():
    class _BoomContract:
        contract_id = "boom-callback.v1"
        requires_agent = False

        async def execute(
            self,
            task_item: TaskItem,
            *,
            agent,
            worker_context: ExecutionContractContext,
        ):
            _ = task_item
            _ = agent
            _ = worker_context
            raise RuntimeError("boom")

    async def scenario() -> None:
        queue = InMemoryTaskQueue()
        task = await queue.enqueue_contract(
            "boom-callback.v1",
            payload={},
            agent_name=None,
            max_retries=2,
        )

        async def _bad_failure(_task: TaskItem) -> None:
            raise RuntimeError("on_failure boom")

        worker = TaskWorker(
            queue,
            agents={},
            execution_contracts={"boom-callback.v1": _BoomContract()},
            on_failure=_bad_failure,
        )

        running = await queue.dequeue(timeout=0.1)
        assert running is not None
        await worker._execute_task(running)  # noqa: SLF001

        current = await queue.get(task.id)
        assert current is not None
        assert current.status == "retrying"
        assert current.retry_count == 1
        assert current.error == "boom"
        assert queue.pending_count == 1

    run_async(scenario())
