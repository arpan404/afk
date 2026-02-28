"""
---
name: Chat History Manager — Tools
description: Thread management and history tools for the chat history agent.
tags: [tools, memory]
---
---
All tool definitions for the chat history manager. Tools use the shared memory
store from config.py and operate on thread-scoped events and state.
---
"""

from pydantic import BaseModel, Field  # <- Pydantic for typed tool argument schemas.
from afk.tools import tool  # <- @tool decorator.
from afk.memory import MemoryEvent, now_ms, new_id  # <- Memory helpers.

from config import memory  # <- Shared memory store from config.py.


# ===========================================================================
# Shared state
# ===========================================================================

current_thread_id = "default"
known_threads: list[str] = ["default"]


# ===========================================================================
# Tool argument schemas
# ===========================================================================

class ListThreadsArgs(BaseModel):
    pass

class SwitchThreadArgs(BaseModel):
    thread_name: str = Field(description="The name of the conversation thread to switch to")

class GetHistoryArgs(BaseModel):
    limit: int = Field(default=10, description="Maximum number of recent messages to retrieve")

class ClearHistoryArgs(BaseModel):
    pass

class GetThreadInfoArgs(BaseModel):
    pass


# ===========================================================================
# Tool definitions
# ===========================================================================

@tool(args_model=ListThreadsArgs, name="list_threads", description="List all available conversation threads")
async def list_threads(args: ListThreadsArgs) -> str:
    thread_details = []
    for thread_name in known_threads:
        msg_count = await memory.get_state(thread_id=thread_name, key="message_count")
        count_str = f" ({msg_count} messages)" if msg_count is not None else " (empty)"
        active = " [ACTIVE]" if thread_name == current_thread_id else ""
        thread_details.append(f"  - {thread_name}{count_str}{active}")
    return "Conversation threads:\n" + "\n".join(thread_details)


@tool(args_model=SwitchThreadArgs, name="switch_thread", description="Switch to a different conversation thread, creating it if it doesn't exist")
async def switch_thread(args: SwitchThreadArgs) -> str:
    global current_thread_id
    thread_name = args.thread_name.strip().lower().replace(" ", "-")
    if thread_name not in known_threads:
        known_threads.append(thread_name)
        await memory.put_state(thread_id=thread_name, key="created_at", value=now_ms())
        await memory.put_state(thread_id=thread_name, key="message_count", value=0)
        current_thread_id = thread_name
        return f"Created new thread '{thread_name}' and switched to it."
    current_thread_id = thread_name
    events = await memory.get_recent_events(thread_id=thread_name, limit=3)
    if not events:
        return f"Switched to thread '{thread_name}'. No conversation history yet."
    preview_lines = []
    for evt in events:
        if evt.type == "trace" and "role" in evt.payload:
            role = evt.payload.get("role", "unknown")
            text = evt.payload.get("text", "")
            preview_lines.append(f"  [{role}]: {text[:80]}...")
    preview = "\n".join(preview_lines) if preview_lines else "  (no readable messages)"
    return f"Switched to thread '{thread_name}'. Recent:\n{preview}"


@tool(args_model=GetHistoryArgs, name="get_history", description="Retrieve conversation history for the current thread")
async def get_history(args: GetHistoryArgs) -> str:
    events = await memory.get_recent_events(thread_id=current_thread_id, limit=args.limit)
    if not events:
        return f"Thread '{current_thread_id}' has no conversation history yet."
    lines = [f"History for '{current_thread_id}' (last {args.limit}):"]
    for evt in events:
        if evt.type == "trace" and "role" in evt.payload:
            lines.append(f"  [{evt.payload['role']}]: {evt.payload.get('text', '')}")
    all_state = await memory.list_state(thread_id=current_thread_id)
    if all_state:
        lines.append(f"\n  Metadata keys: {', '.join(all_state.keys())}")
    return "\n".join(lines)


@tool(args_model=ClearHistoryArgs, name="clear_history", description="Clear the conversation history for the current thread")
async def clear_history(args: ClearHistoryArgs) -> str:
    await memory.replace_thread_events(thread_id=current_thread_id, events=[])
    await memory.put_state(thread_id=current_thread_id, key="message_count", value=0)
    return f"Cleared all history for thread '{current_thread_id}'."


@tool(args_model=GetThreadInfoArgs, name="get_thread_info", description="Show detailed metadata about the current thread")
async def get_thread_info(args: GetThreadInfoArgs) -> str:
    all_state = await memory.list_state(thread_id=current_thread_id)
    if not all_state:
        return f"Thread '{current_thread_id}' has no stored metadata."
    lines = [f"Thread info for '{current_thread_id}':"]
    for key, value in all_state.items():
        lines.append(f"  {key}: {value}")
    event_count = len(await memory.get_recent_events(thread_id=current_thread_id, limit=1000))
    lines.append(f"  total_events: {event_count}")
    return "\n".join(lines)


# ===========================================================================
# Helper: log a message as a MemoryEvent
# ===========================================================================

async def log_message(thread_id: str, role: str, text: str) -> None:
    """Record a chat message as a MemoryEvent."""
    event = MemoryEvent(
        id=new_id("msg"),
        thread_id=thread_id,
        user_id=role,
        type="trace",
        timestamp=now_ms(),
        payload={"role": role, "text": text},
    )
    await memory.append_event(event)
    count = await memory.get_state(thread_id=thread_id, key="message_count")
    await memory.put_state(thread_id=thread_id, key="message_count", value=(count or 0) + 1)
