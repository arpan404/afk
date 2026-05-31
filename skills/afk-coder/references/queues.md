# Queues

AFK task queue subsystem: `TaskQueue` for distributed agent execution with retry policies,
`TaskWorker` consumer loop, and pluggable backends.

- Doc page: https://afk.arpan.sh/library/queues
- Source: `src/afk/queues/`
- Cross-refs: `agents-and-runner.md`, `evals-and-testing.md`

---

## Overview

AFK provides a persistent task queue for enqueuing, dequeuing, and tracking agent tasks with:
- Automatic retry with configurable backoff
- Pluggable backends (InMemory, Redis)
- Worker consumer loop with metrics
- Execution contracts for agent invocation

Key public imports:

```python
from afk.queues import (
    TaskQueue,
    TaskItem,
    TaskStatus,
    InMemoryTaskQueue,
    TaskWorker,
    TaskWorkerConfig,
    RetryPolicy,
    RunnerChatExecutionContract,
    JobDispatchExecutionContract,
    create_task_queue_from_env,
)
```

---

## TaskItem

One queued task item.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task_id` | `str` | -- | Unique task identifier |
| `contract_key` | `str` | -- | Execution contract key |
| `payload` | `dict[str, JSONValue]` | -- | Task payload |
| `status` | `TaskStatus` | `"pending"` | Current task status |
| `attempt` | `int` | `0` | Current attempt number |
| `created_at` | `int` | -- | Epoch milliseconds |
| `next_attempt_at` | `int \| None` | `None` | Next attempt scheduled time |
| `dead_letter_reason` | `str \| None` | `None` | Reason if moved to dead letter |

---

## TaskStatus

| Value | Description |
|-------|-------------|
| `"pending"` | Task waiting to be processed |
| `"running"` | Task currently being executed |
| `"completed"` | Task completed successfully |
| `"failed"` | Task failed after retries exhausted |
| `"dead_letter"` | Task moved to dead letter queue |

---

## TaskQueue

Abstract queue interface.

| Method | Signature | Description |
|--------|-----------|-------------|
| `enqueue_contract` | `(contract, payload, *, agent_name, thread_id=None) -> str` | Enqueue a task using contract |
| `dequeue` | `(timeout=None) -> TaskItem \| None` | Dequeue the next pending task |
| `ack` | `(task_id, success) -> None` | Acknowledge task completion |
| `get` | `(task_id) -> TaskItem \| None` | Get task by ID |
| `list` | `(status, limit=100) -> list[TaskItem]` | List tasks by status |
| `update_status` | `(task_id, status) -> None` | Update task status |

---

## RetryPolicy

Retry configuration for failed tasks.

```python
RetryPolicy(
    max_attempts=3,
    backoff_base_s=1.0,
    backoff_max_s=60.0,
    backoff_jitter_s=0.5,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_attempts` | `int` | `3` | Maximum retry attempts |
| `backoff_base_s` | `float` | `1.0` | Base backoff duration in seconds |
| `backoff_max_s` | `float` | `60.0` | Maximum backoff cap |
| `backoff_jitter_s` | `float` | `0.5` | Random jitter added to backoff |

---

## Execution Contracts

### RunnerChatExecutionContract

Execute an agent via Runner chat contract.

```python
from afk.queues import RunnerChatExecutionContract

contract = RunnerChatExecutionContract(
    agent=my_agent,
    max_steps=20,
    fail_safe=FailSafeConfig(max_steps=20),
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `agent` | `BaseAgent` | -- | Agent to execute |
| `payload_key` | `"user_message"` | User message key in payload |
| `context_key` | `"context"` | Context key |
| `thread_id_key` | `"thread_id"` | Thread ID key |
| `max_steps` | `int` | `None` | Override agent max_steps |
| `fail_safe` | `FailSafeConfig \| None` | `None` | Fail-safe config |

### JobDispatchExecutionContract

Execute a job handler function.

```python
from afk.queues import JobDispatchExecutionContract

contract = JobDispatchExecutionContract(
    handler=my_handler_function,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `handler` | `Callable` | -- | Job handler function |

---

## TaskWorker

Consumer loop that processes tasks from a queue.

```python
from afk.queues import TaskWorker, TaskWorkerConfig

worker = TaskWorker(
    queue,
    agents={"greeter": my_agent},
    config=TaskWorkerConfig(
        poll_interval_s=1.0,
        max_concurrency=4,
    ),
)

await worker.start()
```

### TaskWorkerConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `poll_interval_s` | `float` | `1.0` | Queue poll interval |
| `max_concurrency` | `int` | `4` | Max concurrent task processing |
| `shutdown_timeout_s` | `float` | `30.0` | Graceful shutdown timeout |

### TaskWorker Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `start()` | `None` | Start the worker loop |
| `stop()` | `None` | Stop the worker loop |
| `running()` | `bool` | Check if worker is running |

---

## In-Memory Queue

Simple in-memory task queue for development and testing.

```python
from afk.queues import InMemoryTaskQueue

queue = InMemoryTaskQueue()
await queue.enqueue_contract(
    RunnerChatExecutionContract(agent=my_agent),
    payload={"user_message": "Hello!"},
    agent_name="greeter",
)
```

---

## Redis Queue

Production queue with Redis backend (optional extra).

```python
from afk.queues import RedisTaskQueue

queue = RedisTaskQueue(
    url="redis://localhost:6379/0",
    key_prefix="afk:tasks",
)
```

Requires the Redis client package alongside AFK:

```bash
python -m pip install afk-py redis
```

---

## Factory

Create queue from environment variables.

```python
from afk.queues import create_task_queue_from_env

# Respects AFK_QUEUE_BACKEND env var
queue = create_task_queue_from_env()
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `AFK_QUEUE_BACKEND` | `memory` | `memory` or `redis` |
| `AFK_REDIS_URL` | -- | Redis connection URL |

---

## Basic Example

```python
from afk.agents import Agent
from afk.queues import (
    InMemoryTaskQueue,
    RunnerChatExecutionContract,
    TaskWorker,
)

agent = Agent(model="gpt-4.1-mini", instructions="You are a greeter.")
queue = InMemoryTaskQueue()

# Enqueue a task
task_id = await queue.enqueue_contract(
    RunnerChatExecutionContract(agent=agent),
    payload={"user_message": "Hello!"},
    agent_name="greeter",
)

# Start worker to process
worker = TaskWorker(queue, agents={"greeter": agent})
await worker.start()

# Or manually process
item = await queue.dequeue()
# ... process item ...
await queue.ack(item.task_id, success=True)
```

---

## Source Files

| File | Purpose |
|------|---------|
| `src/afk/queues/__init__.py` | Public API exports |
| `src/afk/queues/base.py` | `BaseTaskQueue` abstract class |
| `src/afk/queues/memory.py` | `InMemoryTaskQueue` implementation |
| `src/afk/queues/redis_queue.py` | `RedisTaskQueue` implementation |
| `src/afk/queues/factory.py` | `create_task_queue_from_env` |
| `src/afk/queues/worker.py` | `TaskWorker` implementation |
| `src/afk/queues/types.py` | `TaskItem`, `TaskStatus`, `RetryPolicy` |
| `src/afk/queues/contracts.py` | Execution contracts |

---

## Cross-References

- **Agents**: See [agents-and-runner.md](./agents-and-runner.md) for `Agent` configuration
- **Evals**: See [evals-and-testing.md](./evals-and-testing.md) for testing queued agents
