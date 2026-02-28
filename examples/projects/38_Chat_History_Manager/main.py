"""
---
name: Chat History Manager
description: A multi-thread chat manager demonstrating create_memory_store_from_env() factory, Runner.resume() checkpoint restoration, and full memory lifecycle.
tags: [agent, runner, tools, memory, state, events, factory, resume, checkpoint]
---
---
This example demonstrates three key AFK features:

1. **create_memory_store_from_env()**: Environment-variable-based memory store factory. Set
   AFK_MEMORY_BACKEND to "inmemory", "sqlite", "redis", or "postgres" and the factory creates
   the right store automatically. No code changes needed to switch backends.

2. **Runner.resume()**: Checkpoint-based run resumption. If a run is interrupted (crash,
   timeout, cancel), Runner.resume(agent, run_id=..., thread_id=...) picks up where it left
   off using checkpointed state. This requires a persistent memory store (SQLite, Redis, Postgres).

3. **Full memory lifecycle**: append_event, get_recent_events, put_state/get_state, list_state,
   replace_thread_events — the complete MemoryStore API for thread-scoped events and state.
---
"""

import asyncio  # <- Async for memory and resume operations.
from afk.core import Runner  # <- Runner executes agents and manages checkpoints.
from afk.agents import Agent  # <- Agent defines the chat manager.
from afk.memory import new_id  # <- ID generator for run_id.

from config import memory  # <- Shared memory store created via create_memory_store_from_env().
from tools import (  # <- Import tools and helpers from tools module.
    list_threads, switch_thread, get_history, clear_history, get_thread_info,
    log_message, current_thread_id, known_threads,
)
import tools as tools_module  # <- Import module for updating current_thread_id.


# ===========================================================================
# Agent definition
# ===========================================================================

chat_manager = Agent(
    name="chat-history-manager",
    model="ollama_chat/gpt-oss:20b",
    instructions="""
    You are a helpful chat assistant that manages multiple conversation threads.

    You can:
    - Chat normally (respond to messages)
    - List all threads with list_threads
    - Switch threads with switch_thread
    - View history with get_history
    - Clear history with clear_history
    - View thread metadata with get_thread_info

    Always confirm which thread is active after switching. Keep responses concise.
    """,
    tools=[list_threads, switch_thread, get_history, clear_history, get_thread_info],
)


runner = Runner(
    memory_store=memory,  # <- Pass the memory store to the Runner. This enables checkpointing for Runner.resume() and gives the runner access to thread state.
)


# ===========================================================================
# Runner.resume() demonstration
# ===========================================================================

async def demonstrate_resume():
    """Demonstrate Runner.resume() for checkpoint-based run resumption.

    Runner.resume() picks up an interrupted run from its last checkpoint.
    This requires:
    1. A persistent memory store (SQLite, Redis, or Postgres — not InMemory)
    2. The original run_id and thread_id
    3. The same agent definition

    For this demo, we simulate a run, capture its run_id, and show how
    resume() would be called to continue it.
    """
    print("\n--- Runner.resume() Demo ---")
    print("NOTE: resume() requires a persistent backend (SQLite/Redis/Postgres).")
    print("With InMemoryMemoryStore, checkpoints are lost on restart.\n")

    thread_id = "resume-demo"
    run_id = new_id("run")  # <- Generate a unique run ID. In production, this comes from the original run.

    print(f"  run_id:    {run_id}")
    print(f"  thread_id: {thread_id}")
    print()

    # --- First run: execute normally ---
    print("Step 1: Execute initial run...")
    result = await runner.run(
        chat_manager,
        user_message="Create a thread called 'project-alpha' and list all threads.",
        thread_id=thread_id,
    )
    print(f"  Result: {result.final_text[:100]}...")
    print()

    # --- Simulate resumption ---
    # In a real scenario, the first run would have been interrupted (crash, timeout).
    # Runner.resume() uses checkpointed state to continue from where it stopped.
    print("Step 2: Runner.resume() would be called like this:")
    print(f"  result = await runner.resume(")
    print(f"      chat_manager,")
    print(f"      run_id='{run_id}',")
    print(f"      thread_id='{thread_id}',")
    print(f"      context={{'resumed': True}},  # optional context overlay")
    print(f"  )")
    print()
    print("  resume() restores the checkpoint and continues execution from the")
    print("  last safe boundary. Tool side-effects that already completed are")
    print("  replayed from the idempotency journal (not re-executed).")
    print()
    print("  Related method: runner.resume_handle() returns an AgentRunHandle")
    print("  for lifecycle control (pause/resume/cancel) on the resumed run.")
    print()
    print("Resume demo complete.")


# ===========================================================================
# Interactive session
# ===========================================================================

async def interactive_session():
    """Standard multi-thread chat loop."""
    await memory.setup()  # <- Initialize the memory store. MUST be called before any operations.

    # Initialize default thread
    from afk.memory import now_ms
    await memory.put_state(thread_id="default", key="created_at", value=now_ms())
    await memory.put_state(thread_id="default", key="message_count", value=0)

    print()
    print("Commands: 'list threads', 'switch to <name>', 'show history', 'thread info'")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input(f"[{tools_module.current_thread_id}] > ")
        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        await log_message(tools_module.current_thread_id, "user", user_input)

        await memory.put_state(
            thread_id=tools_module.current_thread_id,
            key="last_topic",
            value=user_input[:100],
        )

        response = await runner.run(chat_manager, user_message=user_input)

        await log_message(tools_module.current_thread_id, "assistant", response.final_text)
        print(f"[chat-history-manager] > {response.final_text}\n")

    # Session summary
    print("\n--- Session Summary ---")
    for thread_name in tools_module.known_threads:
        count = await memory.get_state(thread_id=thread_name, key="message_count")
        print(f"  {thread_name}: {count or 0} messages")
    await memory.close()


async def main():
    print("Chat History Manager")
    print("=" * 55)

    print()
    print("Modes:")
    print("  1. Interactive chat session")
    print("  2. Runner.resume() demonstration")
    print()

    choice = input("Choose mode (1-2): ").strip()

    if choice == "2":
        await memory.setup()
        await demonstrate_resume()
        await memory.close()
    else:
        await interactive_session()


if __name__ == "__main__":
    asyncio.run(main())



"""
---
Tl;dr: This example demonstrates three features: create_memory_store_from_env() creates a memory
store from the AFK_MEMORY_BACKEND environment variable (inmemory, sqlite, redis, postgres) with
zero code changes between backends. Runner.resume(agent, run_id=..., thread_id=...) restores a
checkpointed run after interruption — tool side-effects are replayed from the idempotency journal.
The full memory lifecycle is shown: append_event for messages, get_recent_events for history,
put_state/get_state for metadata, list_state for inspection, replace_thread_events for clearing.
---
---
What's next?
- Set AFK_MEMORY_BACKEND=sqlite and AFK_SQLITE_PATH=chat.db to persist across restarts.
- Use Runner.resume() with a persistent backend to actually resume interrupted runs.
- Try resume_handle() for lifecycle control (pause/resume/cancel) on resumed runs.
- Add vector search with search_long_term_memory_vector for semantic thread search.
- Build a "merge threads" tool combining events from multiple threads.
- Check out the Production Agent example for the ultimate combination of all features!
---
"""
