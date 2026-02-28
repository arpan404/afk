"""
---
name: Production Agent — Tools
description: Tool definitions with ToolRegistry, policy enforcement, middleware logging, and ToolContext for the production task manager.
tags: [tools, registry, policy, middleware, context]
---
---
This module defines the tools for the production task management agent. Tools are registered in a ToolRegistry with a policy hook (for access control) and a middleware (for logging). Each tool uses ToolContext to access runtime metadata like request_id and user_id. Tasks are persisted in SQLiteMemoryStore via put_state/get_state, and every operation is logged as a MemoryEvent for audit trails.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas with validation.
from afk.tools import tool, ToolRegistry, ToolCallRecord  # <- tool decorator, ToolRegistry for centralized management, ToolCallRecord for call history.
from afk.tools.core.base import ToolContext  # <- ToolContext carries runtime info (request_id, user_id, metadata) into tool functions. Useful for logging, access control, and audit trails.
from afk.tools.core.errors import ToolPolicyError  # <- Raised by policy hooks to block tool calls. The runner catches this and sends an error message to the model.
from afk.memory import MemoryEvent, now_ms, new_id  # <- MemoryEvent for logging, now_ms/new_id for timestamps and unique IDs.

from config import memory, THREAD_ID  # <- Import the shared memory store and thread ID from config.py.


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class AddTaskArgs(BaseModel):  # <- Schema for creating a new task.
    title: str = Field(description="The title of the task")
    priority: str = Field(default="medium", description="Priority level: low, medium, high, critical")
    category: str = Field(default="general", description="Task category: general, bug, feature, docs, ops")


class ListTasksArgs(BaseModel):  # <- Schema for listing tasks with optional filters.
    status: str = Field(default="all", description="Filter by status: all, open, done")
    priority: str = Field(default="all", description="Filter by priority: all, low, medium, high, critical")


class CompleteTaskArgs(BaseModel):  # <- Schema for marking a task as completed.
    task_id: int = Field(description="The numeric ID of the task to mark as complete")


class UpdateTaskArgs(BaseModel):  # <- Schema for updating a task's properties.
    task_id: int = Field(description="The numeric ID of the task to update")
    title: str | None = Field(default=None, description="New title for the task")
    priority: str | None = Field(default=None, description="New priority: low, medium, high, critical")
    category: str | None = Field(default=None, description="New category: general, bug, feature, docs, ops")


class DeleteTaskArgs(BaseModel):  # <- Schema for deleting a task.
    task_id: int = Field(description="The numeric ID of the task to delete")


class GetStatsArgs(BaseModel):  # <- No-argument tool for retrieving task statistics.
    pass


# ===========================================================================
# Tool definitions
# ===========================================================================

@tool(args_model=AddTaskArgs, name="add_task", description="Create a new task with title, priority, and category")
async def add_task(args: AddTaskArgs, ctx: ToolContext) -> str:  # <- The (args, ctx) signature gives this tool access to ToolContext. The runner automatically injects ctx with request_id, user_id, and metadata.
    # Get current task counter from state
    counter = await memory.get_state(thread_id=THREAD_ID, key="task_counter")  # <- Atomic counter stored in memory. Each new task gets the next available ID.
    task_id = (counter or 0) + 1

    # Build the task object
    task = {
        "id": task_id,
        "title": args.title,
        "priority": args.priority.lower(),
        "category": args.category.lower(),
        "status": "open",
        "created_at": now_ms(),
        "created_by": ctx.user_id or "unknown",  # <- ToolContext.user_id comes from the runner's runtime. In production, this would be the authenticated user making the request.
        "request_id": ctx.request_id,  # <- ToolContext.request_id traces this specific operation. Useful for correlating tool calls with API requests in production logs.
    }

    # Persist the task in memory state
    await memory.put_state(thread_id=THREAD_ID, key=f"task:{task_id}", value=task)  # <- Each task is stored as a separate state key like "task:1", "task:2", etc. This allows individual retrieval and updates.
    await memory.put_state(thread_id=THREAD_ID, key="task_counter", value=task_id)  # <- Update the counter so the next task gets the right ID.

    # Log the creation as an event for audit trail
    event = MemoryEvent(
        id=new_id("task"),
        thread_id=THREAD_ID,
        user_id=ctx.user_id or "system",
        type="trace",
        timestamp=now_ms(),
        payload={"action": "create", "task_id": task_id, "title": args.title, "priority": args.priority},
    )
    await memory.append_event(event)  # <- Every task operation is logged as a MemoryEvent. This creates an audit trail you can query with get_recent_events.

    return f"Created task #{task_id}: \"{args.title}\" (priority: {args.priority}, category: {args.category})"


@tool(args_model=ListTasksArgs, name="list_tasks", description="List all tasks with optional status and priority filters")
async def list_tasks(args: ListTasksArgs) -> str:
    # Get all task state keys
    all_state = await memory.list_state(thread_id=THREAD_ID, prefix="task:")  # <- list_state with prefix="task:" returns only task-related keys. This is more efficient than fetching all state.

    if not all_state:
        return "No tasks found. Create one with add_task."

    tasks = []
    for key, value in all_state.items():
        if key == "task_counter":  # <- Skip the counter key -- it's not a task.
            continue
        if not isinstance(value, dict):
            continue
        tasks.append(value)

    # Apply filters
    if args.status != "all":
        tasks = [t for t in tasks if t.get("status") == args.status]

    if args.priority != "all":
        tasks = [t for t in tasks if t.get("priority") == args.priority.lower()]

    if not tasks:
        return f"No tasks matching filters (status={args.status}, priority={args.priority})."

    # Sort by priority (critical > high > medium > low), then by ID
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    tasks.sort(key=lambda t: (priority_order.get(t.get("priority", "medium"), 2), t.get("id", 0)))

    lines = [f"Tasks ({len(tasks)} total):"]
    for t in tasks:
        status_icon = "[x]" if t.get("status") == "done" else "[ ]"
        lines.append(
            f"  {status_icon} #{t['id']}: {t['title']} "
            f"(priority: {t.get('priority', 'medium')}, category: {t.get('category', 'general')}, status: {t.get('status', 'open')})"
        )

    return "\n".join(lines)


@tool(args_model=CompleteTaskArgs, name="complete_task", description="Mark a task as completed by its ID")
async def complete_task(args: CompleteTaskArgs, ctx: ToolContext) -> str:
    task = await memory.get_state(thread_id=THREAD_ID, key=f"task:{args.task_id}")  # <- Fetch the specific task by ID.
    if task is None or not isinstance(task, dict):
        return f"Task #{args.task_id} not found."

    if task.get("status") == "done":
        return f"Task #{args.task_id} is already completed."

    task["status"] = "done"
    task["completed_at"] = now_ms()
    task["completed_by"] = ctx.user_id or "unknown"

    await memory.put_state(thread_id=THREAD_ID, key=f"task:{args.task_id}", value=task)  # <- Overwrite the task state with the updated version.

    # Log completion event
    event = MemoryEvent(
        id=new_id("task"),
        thread_id=THREAD_ID,
        user_id=ctx.user_id or "system",
        type="trace",
        timestamp=now_ms(),
        payload={"action": "complete", "task_id": args.task_id, "title": task.get("title", "")},
    )
    await memory.append_event(event)

    return f"Completed task #{args.task_id}: \"{task['title']}\""


@tool(args_model=UpdateTaskArgs, name="update_task", description="Update a task's title, priority, or category")
async def update_task(args: UpdateTaskArgs, ctx: ToolContext) -> str:
    task = await memory.get_state(thread_id=THREAD_ID, key=f"task:{args.task_id}")
    if task is None or not isinstance(task, dict):
        return f"Task #{args.task_id} not found."

    changes = []
    if args.title is not None:
        task["title"] = args.title
        changes.append(f"title='{args.title}'")
    if args.priority is not None:
        task["priority"] = args.priority.lower()
        changes.append(f"priority={args.priority.lower()}")
    if args.category is not None:
        task["category"] = args.category.lower()
        changes.append(f"category={args.category.lower()}")

    if not changes:
        return f"No changes specified for task #{args.task_id}."

    task["updated_at"] = now_ms()
    await memory.put_state(thread_id=THREAD_ID, key=f"task:{args.task_id}", value=task)

    # Log update event
    event = MemoryEvent(
        id=new_id("task"),
        thread_id=THREAD_ID,
        user_id=ctx.user_id or "system",
        type="trace",
        timestamp=now_ms(),
        payload={"action": "update", "task_id": args.task_id, "changes": changes},
    )
    await memory.append_event(event)

    return f"Updated task #{args.task_id}: {', '.join(changes)}"


@tool(args_model=DeleteTaskArgs, name="delete_task", description="Delete a task by its ID")
async def delete_task(args: DeleteTaskArgs, ctx: ToolContext) -> str:
    task = await memory.get_state(thread_id=THREAD_ID, key=f"task:{args.task_id}")
    if task is None or not isinstance(task, dict):
        return f"Task #{args.task_id} not found."

    title = task.get("title", "unknown")
    await memory.delete_state(thread_id=THREAD_ID, key=f"task:{args.task_id}")  # <- delete_state removes a key entirely. The task is gone from memory.

    # Log deletion event
    event = MemoryEvent(
        id=new_id("task"),
        thread_id=THREAD_ID,
        user_id=ctx.user_id or "system",
        type="trace",
        timestamp=now_ms(),
        payload={"action": "delete", "task_id": args.task_id, "title": title},
    )
    await memory.append_event(event)

    return f"Deleted task #{args.task_id}: \"{title}\""


@tool(args_model=GetStatsArgs, name="get_stats", description="Show task statistics: total, open, completed, by priority")
async def get_stats(args: GetStatsArgs) -> str:
    all_state = await memory.list_state(thread_id=THREAD_ID, prefix="task:")

    tasks = []
    for key, value in all_state.items():
        if key == "task_counter":
            continue
        if isinstance(value, dict):
            tasks.append(value)

    if not tasks:
        return "No tasks found. Create some with add_task first."

    total = len(tasks)
    open_count = sum(1 for t in tasks if t.get("status") == "open")
    done_count = sum(1 for t in tasks if t.get("status") == "done")

    # Count by priority
    by_priority: dict[str, int] = {}
    for t in tasks:
        p = t.get("priority", "medium")
        by_priority[p] = by_priority.get(p, 0) + 1

    # Count by category
    by_category: dict[str, int] = {}
    for t in tasks:
        c = t.get("category", "general")
        by_category[c] = by_category.get(c, 0) + 1

    # Get recent activity from event log
    recent_events = await memory.get_recent_events(thread_id=THREAD_ID, limit=5)
    activity_lines = []
    for evt in recent_events:
        if evt.type == "trace" and "action" in evt.payload:
            action = evt.payload.get("action", "")
            title = evt.payload.get("title", "")
            activity_lines.append(f"  - {action}: {title}")

    priority_str = ", ".join(f"{k}: {v}" for k, v in sorted(by_priority.items()))
    category_str = ", ".join(f"{k}: {v}" for k, v in sorted(by_category.items()))
    activity_str = "\n".join(activity_lines) if activity_lines else "  (no recent activity)"

    return (
        f"Task Statistics:\n"
        f"  Total: {total} | Open: {open_count} | Completed: {done_count}\n"
        f"  Completion rate: {done_count / total * 100:.0f}%\n"
        f"  By priority: {priority_str}\n"
        f"  By category: {category_str}\n"
        f"\nRecent activity:\n{activity_str}"
    )


# ===========================================================================
# ToolRegistry with policy and middleware
# ===========================================================================

def task_policy(tool_name: str, raw_args: dict, ctx: ToolContext) -> None:  # <- A policy hook that runs BEFORE every tool call. Use it for access control, budget checks, rate limiting, or audit logging. Raise ToolPolicyError to block the call.
    """Policy hook: validate tool calls before execution.

    This policy enforces:
    - Priority values must be valid
    - Category values must be valid
    - Delete operations require user identification
    """
    valid_priorities = {"low", "medium", "high", "critical"}
    valid_categories = {"general", "bug", "feature", "docs", "ops"}

    if tool_name in ("add_task", "update_task"):
        priority = raw_args.get("priority", "").lower()
        if priority and priority not in valid_priorities:
            raise ToolPolicyError(f"Invalid priority '{priority}'. Must be one of: {valid_priorities}")  # <- ToolPolicyError blocks the tool call and sends the error message to the model, which can then ask the user for a valid priority.

        category = raw_args.get("category", "").lower()
        if category and category not in valid_categories:
            raise ToolPolicyError(f"Invalid category '{category}'. Must be one of: {valid_categories}")

    if tool_name == "delete_task" and not ctx.user_id:
        raise ToolPolicyError("Delete operations require user identification. Set user_id in ToolContext.")  # <- Require user identification for destructive operations. In production, this would check authentication and authorization.


async def logging_middleware(call_next, tool, raw_args, ctx):  # <- A registry-level middleware that wraps ALL tool calls. Runs before AND after every tool execution. Great for logging, metrics, and tracing.
    """Registry middleware: log every tool call with timing."""
    import time
    start = time.time()
    tool_name = tool.spec.name

    # Pre-execution logging
    print(f"  [middleware] -> {tool_name}(args={raw_args})")  # <- Log which tool is being called and with what arguments. In production, you'd send this to a structured logging system.

    # Execute the actual tool call
    result = await call_next(tool, raw_args, ctx, None, None)  # <- call_next passes through to the next middleware or the actual tool. This is the middleware chain pattern.

    # Post-execution logging
    elapsed = time.time() - start
    status = "ok" if result.success else f"error: {result.error_message}"
    print(f"  [middleware] <- {tool_name}: {status} ({elapsed:.3f}s)")  # <- Log the result and timing. This creates an observable execution trace for debugging and monitoring.

    return result


# --- Build the registry ---

registry = ToolRegistry(  # <- ToolRegistry with policy and middleware. The policy runs first (can block calls), then middleware wraps execution (can log, transform, or retry).
    policy=task_policy,  # <- Attach the policy hook. It's called before every tool execution and can raise ToolPolicyError to block the call.
    max_concurrency=8,  # <- Allow up to 8 concurrent tool executions.
)

registry.register(add_task)  # <- Register each tool with the registry.
registry.register(list_tasks)
registry.register(complete_task)
registry.register(update_task)
registry.register(delete_task)
registry.register(get_stats)

registry.add_middleware(logging_middleware)  # <- Add the logging middleware. It wraps every tool call made through this registry.
