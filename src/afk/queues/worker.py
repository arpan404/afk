"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Task worker — consumer loop that dequeues and executes contract-aware tasks.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from ..agents import BaseAgent
from ..llms.types import JSONValue
from .contracts import (
    EXECUTION_CONTRACT_KEY,
    JOB_DISPATCH_CONTRACT,
    RUNNER_CHAT_CONTRACT,
    ExecutionContract,
    ExecutionContractContext,
    ExecutionContractResolutionError,
    ExecutionContractValidationError,
    JobDispatchExecutionContract,
    JobHandler,
    RunnerChatExecutionContract,
)
from .types import (
    RetryPolicy,
    StartupRecoveryCapable,
    TaskItem,
    TaskQueue,
    WorkerPresenceCapable,
)

logger = logging.getLogger("afk.queues.worker")

# Callback signature: called after each task completes or fails.
TaskCallback = Callable[[TaskItem], Awaitable[None] | None]


class WorkerMetrics(Protocol):
    """Minimal metrics interface for queue worker instrumentation."""

    def incr(
        self, name: str, value: int = 1, *, tags: Mapping[str, str] | None = None
    ) -> None:
        """Increment a counter metric."""


class NoOpWorkerMetrics:
    """Default metrics sink when no metrics backend is provided."""

    def incr(
        self, name: str, value: int = 1, *, tags: Mapping[str, str] | None = None
    ) -> None:
        _ = name
        _ = value
        _ = tags


@dataclass
class TaskWorkerConfig:
    """
    Configuration for the task worker.

    Attributes:
        poll_interval_s: Seconds between dequeue attempts when idle.
        max_concurrent_tasks: Maximum tasks executed concurrently.
        shutdown_timeout_s: Grace period for in-flight tasks on shutdown.
        recover_inflight_on_startup: Whether to run startup in-flight recovery
            on queues that support it.
        worker_presence_ttl_s: Presence TTL (seconds) for queue backends that
            track active workers.
        worker_presence_refresh_s: Presence heartbeat interval (seconds).
    """

    poll_interval_s: float = 1.0
    max_concurrent_tasks: int = 4
    shutdown_timeout_s: float = 30.0
    recover_inflight_on_startup: bool = True
    worker_presence_ttl_s: float = 30.0
    worker_presence_refresh_s: float = 10.0


class TaskWorker:
    """
    Consumer loop that dequeues tasks and executes them via execution contracts.

    Contracts are resolved from `task.metadata["execution_contract"]`.
    Missing/unknown/invalid contracts fail immediately without retry.
    """

    def __init__(
        self,
        queue: TaskQueue,
        *,
        agents: Mapping[str, BaseAgent],
        execution_contracts: Mapping[str, ExecutionContract] | None = None,
        job_handlers: Mapping[str, JobHandler] | None = None,
        retry_policies: Mapping[str, RetryPolicy] | None = None,
        metrics: WorkerMetrics | None = None,
        config: TaskWorkerConfig | None = None,
        on_complete: TaskCallback | None = None,
        on_failure: TaskCallback | None = None,
    ) -> None:
        self._queue = queue
        self._agents = dict(agents)
        self._contract_context = ExecutionContractContext(
            job_handlers=dict(job_handlers or {})
        )
        self._execution_contracts = self._build_contract_map(
            execution_contracts=execution_contracts
        )
        self._retry_policies = dict(retry_policies or {})
        self._config = config or TaskWorkerConfig()
        self._metrics: WorkerMetrics = metrics or NoOpWorkerMetrics()
        self._on_complete = on_complete
        self._on_failure = on_failure
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._worker_id = uuid.uuid4().hex
        self._presence_queue = (
            queue if isinstance(queue, WorkerPresenceCapable) else None
        )
        self._recovery_queue = (
            queue if isinstance(queue, StartupRecoveryCapable) else None
        )
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent_tasks)
        self._active_tasks: set[asyncio.Task[None]] = set()

    async def _register_presence(self) -> None:
        if self._presence_queue is None:
            return
        await self._presence_queue.register_worker(
            self._worker_id,
            ttl_s=self._config.worker_presence_ttl_s,
        )

    async def _refresh_presence(self) -> None:
        if self._presence_queue is None:
            return
        await self._presence_queue.refresh_worker(
            self._worker_id,
            ttl_s=self._config.worker_presence_ttl_s,
        )

    async def _unregister_presence(self) -> None:
        if self._presence_queue is None:
            return
        await self._presence_queue.unregister_worker(self._worker_id)

    async def _presence_heartbeat_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._config.worker_presence_refresh_s)
                if not self._running:
                    break
                await self._refresh_presence()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception(
                    "TaskWorker presence heartbeat failed (worker_id=%s)",
                    self._worker_id[:8],
                )

    async def _maybe_recover_inflight(self) -> None:
        if not self._config.recover_inflight_on_startup:
            return
        if self._recovery_queue is None:
            return
        moved = await self._recovery_queue.recover_inflight_if_idle(
            active_worker_id=self._worker_id
        )
        if moved > 0:
            self._metrics.incr("queue_worker_recovered_inflight_total", moved)
            logger.info(
                "TaskWorker recovered %d in-flight task(s) on startup (worker_id=%s)",
                moved,
                self._worker_id[:8],
            )

    def _build_contract_map(
        self,
        *,
        execution_contracts: Mapping[str, ExecutionContract] | None,
    ) -> dict[str, ExecutionContract]:
        contract_map: dict[str, ExecutionContract] = {
            RUNNER_CHAT_CONTRACT: RunnerChatExecutionContract(),
            JOB_DISPATCH_CONTRACT: JobDispatchExecutionContract(),
        }

        for key, value in (execution_contracts or {}).items():
            contract_id = key.strip()
            if not contract_id:
                raise ValueError("execution contract ids must be non-empty")
            declared_id = getattr(value, "contract_id", None)
            if (
                isinstance(declared_id, str)
                and declared_id
                and declared_id != contract_id
            ):
                raise ValueError(
                    f"Contract id mismatch: key '{contract_id}' != handler.contract_id '{declared_id}'"
                )
            contract_map[contract_id] = value
        return contract_map

    async def start(self) -> None:
        """
        Start the worker loop in the background.

        Dequeues tasks and executes them concurrently up to
        ``max_concurrent_tasks`` until ``shutdown()`` is called.
        """
        if self._running:
            raise RuntimeError("TaskWorker is already running")
        if self._presence_queue is not None:
            if self._config.worker_presence_ttl_s <= 0:
                raise ValueError("worker_presence_ttl_s must be > 0")
            if self._config.worker_presence_refresh_s <= 0:
                raise ValueError("worker_presence_refresh_s must be > 0")
            if (
                self._config.worker_presence_refresh_s
                >= self._config.worker_presence_ttl_s
            ):
                raise ValueError(
                    "worker_presence_refresh_s must be less than worker_presence_ttl_s"
                )

        self._running = True
        try:
            await self._register_presence()
            await self._maybe_recover_inflight()
            self._task = asyncio.create_task(self._loop())
            if self._presence_queue is not None:
                self._heartbeat_task = asyncio.create_task(
                    self._presence_heartbeat_loop()
                )
            logger.info(
                "TaskWorker started (max_concurrent=%d, poll=%.1fs, worker_id=%s)",
                self._config.max_concurrent_tasks,
                self._config.poll_interval_s,
                self._worker_id[:8],
            )
        except Exception:
            self._running = False
            if self._heartbeat_task and not self._heartbeat_task.done():
                self._heartbeat_task.cancel()
                await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            self._heartbeat_task = None
            try:
                await self._unregister_presence()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "TaskWorker failed to unregister presence after start error (worker_id=%s)",
                    self._worker_id[:8],
                )
            raise

    async def shutdown(self) -> None:
        """
        Gracefully shut down the worker.

        Waits for in-flight tasks up to ``shutdown_timeout_s``.
        """
        self._running = False

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
        self._heartbeat_task = None

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(
                    self._task, timeout=self._config.shutdown_timeout_s
                )
            except (TimeoutError, asyncio.CancelledError):
                pass

        if self._active_tasks:
            logger.info("Waiting for %d active tasks...", len(self._active_tasks))
            _, pending = await asyncio.wait(
                self._active_tasks,
                timeout=self._config.shutdown_timeout_s,
            )
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        self._task = None
        try:
            await self._unregister_presence()
        except Exception:  # noqa: BLE001
            logger.exception(
                "TaskWorker failed to unregister presence on shutdown (worker_id=%s)",
                self._worker_id[:8],
            )
        logger.info("TaskWorker shut down (worker_id=%s)", self._worker_id[:8])

    @property
    def is_running(self) -> bool:
        """Whether the worker loop is active."""
        return self._running

    @property
    def active_task_count(self) -> int:
        """Number of currently executing tasks."""
        return len(self._active_tasks)

    async def _loop(self) -> None:
        """Main consumer loop."""
        while self._running:
            permit_acquired = False
            try:
                await self._semaphore.acquire()
                permit_acquired = True
                task_item = await self._queue.dequeue(
                    timeout=self._config.poll_interval_s
                )
                if task_item is None:
                    continue
                self._metrics.incr("queue_worker_dequeued_total")

                exec_task = asyncio.create_task(self._execute_task(task_item))
                self._active_tasks.add(exec_task)
                exec_task.add_done_callback(self._task_done)
                permit_acquired = False  # released by _task_done
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Worker loop error")
                await asyncio.sleep(self._config.poll_interval_s)
            finally:
                if permit_acquired:
                    self._semaphore.release()

    def _task_done(self, task: asyncio.Task[None]) -> None:
        """Cleanup callback when a task execution completes."""
        self._active_tasks.discard(task)
        self._semaphore.release()

    def _resolve_contract(self, task_item: TaskItem) -> tuple[str, ExecutionContract]:
        contract_id = task_item.execution_contract
        if contract_id is None:
            raise ExecutionContractResolutionError(
                f"Missing execution contract metadata '{EXECUTION_CONTRACT_KEY}'"
            )

        contract = self._execution_contracts.get(contract_id)
        if contract is None:
            raise ExecutionContractResolutionError(
                f"Unknown execution contract '{contract_id}'"
            )
        return contract_id, contract

    def _resolve_agent_for_contract(
        self,
        task_item: TaskItem,
        *,
        contract: ExecutionContract,
    ) -> BaseAgent | None:
        if not contract.requires_agent:
            return None

        if not isinstance(task_item.agent_name, str) or not task_item.agent_name:
            raise ExecutionContractValidationError(
                f"Contract '{contract.contract_id}' requires a non-empty task.agent_name"
            )

        agent = self._agents.get(task_item.agent_name)
        if agent is None:
            raise ExecutionContractValidationError(
                f"Agent '{task_item.agent_name}' not found for contract '{contract.contract_id}'"
            )
        return agent

    async def _execute_task(self, task_item: TaskItem) -> None:
        """Execute a single task item."""
        try:
            contract_id, contract = self._resolve_contract(task_item)
            agent = self._resolve_agent_for_contract(task_item, contract=contract)
            output = await contract.execute(
                task_item,
                agent=agent,
                worker_context=self._contract_context,
            )
            result_envelope: JSONValue = {"contract": contract_id, "output": output}
            await self._queue.complete(task_item.id, result=result_envelope)
            self._metrics.incr(
                "queue_worker_completed_total",
                tags={"contract": contract_id},
            )
            logger.info(
                "Task %s completed (contract=%s, agent=%s)",
                task_item.id[:8],
                contract_id,
                task_item.agent_name,
            )
            await self._handle_completion(task_item, result=result_envelope)

        except (
            ExecutionContractResolutionError,
            ExecutionContractValidationError,
        ) as exc:
            error = str(exc)
            self._metrics.incr("queue_worker_failed_non_retryable_total")
            logger.error("Task %s failed (non-retryable): %s", task_item.id[:8], error)
            await self._handle_failure(task_item, error=error, retryable=False)
        except asyncio.CancelledError:
            error = "Task execution cancelled"
            self._metrics.incr("queue_worker_failed_non_retryable_total")
            logger.info("Task %s cancelled", task_item.id[:8])
            await self._handle_failure(task_item, error=error, retryable=False)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            self._metrics.incr("queue_worker_failed_retryable_total")
            logger.exception("Task %s failed (retryable): %s", task_item.id[:8], error)
            await self._handle_failure(
                task_item,
                error=error,
                retryable=True,
                retry_policy=self._retry_policy_for(task_item),
            )

    async def _handle_completion(
        self,
        task_item: TaskItem,
        *,
        result: JSONValue,
    ) -> None:
        if not self._on_complete:
            return

        callback_item = await self._queue.get(task_item.id)
        if callback_item is None:
            task_item.status = "completed"
            task_item.result = result
            callback_item = task_item
        try:
            await self._invoke_callback(self._on_complete, callback_item)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Task %s on_complete callback failed",
                task_item.id[:8],
            )

    async def _handle_failure(
        self,
        task_item: TaskItem,
        *,
        error: str,
        retryable: bool,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        await self._queue.fail(
            task_item.id,
            error=error,
            retryable=retryable,
            retry_policy=retry_policy,
        )
        if not self._on_failure:
            return

        callback_item = await self._queue.get(task_item.id)
        if callback_item is None:
            task_item.error = error
            callback_item = task_item
        try:
            await self._invoke_callback(self._on_failure, callback_item)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Task %s on_failure callback failed",
                task_item.id[:8],
            )

    async def _invoke_callback(self, cb: TaskCallback, item: TaskItem) -> None:
        """Invoke a callback, handling both sync and async signatures."""
        result = cb(item)
        if inspect.isawaitable(result):
            await result

    def _retry_policy_for(self, task_item: TaskItem) -> RetryPolicy | None:
        """Resolve retry policy override from task metadata or contract mapping."""
        policy = RetryPolicy.from_metadata(task_item.metadata)
        if policy is not None:
            return policy
        contract_id = task_item.execution_contract
        if contract_id is None:
            return None
        return self._retry_policies.get(contract_id)
