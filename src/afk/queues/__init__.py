"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Persistent task queue package for distributed agent execution.

Provides a ``TaskQueue`` abstraction for enqueuing, dequeuing, and tracking
agent tasks with automatic retry, plus a ``TaskWorker`` consumer loop.

Quick start::

    from afk.queues import InMemoryTaskQueue, RUNNER_CHAT_CONTRACT, TaskWorker

    queue = InMemoryTaskQueue()
    await queue.enqueue_contract(
        RUNNER_CHAT_CONTRACT,
        payload={"user_message": "Hello!", "context": {}},
        agent_name="greeter",
    )

    worker = TaskWorker(queue, agents={"greeter": my_agent})
    await worker.start()
"""

from .base import BaseTaskQueue
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
from .factory import create_task_queue_from_env
from .memory import InMemoryTaskQueue
from .metrics import PrometheusWorkerMetrics
from .types import (
    DEAD_LETTER_REASON_KEY,
    NEXT_ATTEMPT_AT_KEY,
    RETRY_BACKOFF_BASE_KEY,
    RETRY_BACKOFF_JITTER_KEY,
    RETRY_BACKOFF_MAX_KEY,
    RetryPolicy,
    StartupRecoveryCapable,
    TaskItem,
    TaskQueue,
    TaskStatus,
    WorkerPresenceCapable,
)
from .worker import NoOpWorkerMetrics, TaskWorker, TaskWorkerConfig, WorkerMetrics

__all__ = [
    "TaskItem",
    "TaskQueue",
    "BaseTaskQueue",
    "TaskStatus",
    "NEXT_ATTEMPT_AT_KEY",
    "DEAD_LETTER_REASON_KEY",
    "RETRY_BACKOFF_BASE_KEY",
    "RETRY_BACKOFF_MAX_KEY",
    "RETRY_BACKOFF_JITTER_KEY",
    "RetryPolicy",
    "WorkerPresenceCapable",
    "StartupRecoveryCapable",
    "InMemoryTaskQueue",
    "RedisTaskQueue",
    "EXECUTION_CONTRACT_KEY",
    "RUNNER_CHAT_CONTRACT",
    "JOB_DISPATCH_CONTRACT",
    "ExecutionContract",
    "ExecutionContractContext",
    "ExecutionContractResolutionError",
    "ExecutionContractValidationError",
    "RunnerChatExecutionContract",
    "JobDispatchExecutionContract",
    "JobHandler",
    "create_task_queue_from_env",
    "TaskWorker",
    "TaskWorkerConfig",
    "WorkerMetrics",
    "NoOpWorkerMetrics",
    "PrometheusWorkerMetrics",
]


# Lazy import for Redis queue
def __getattr__(name: str):
    """Lazily expose optional queue backends that require extra dependencies."""
    if name == "RedisTaskQueue":
        from .redis_queue import RedisTaskQueue

        return RedisTaskQueue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
