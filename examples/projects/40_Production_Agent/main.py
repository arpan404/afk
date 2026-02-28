"""
---
name: Production Agent
description: A capstone production-ready task management system combining SQLiteMemoryStore, streaming, ToolRegistry, FailSafeConfig, RunnerConfig, dynamic instructions, subagents, and ToolContext.
tags: [agent, runner, streaming, memory, sqlite, tools, registry, policy, middleware, subagents, config, failsafe, context, production]
---
---
This is the capstone example for AFK -- a production-ready task management agent that combines nearly every major framework feature into one cohesive system. It demonstrates: SQLiteMemoryStore for persistent task storage that survives restarts, run_stream for real-time streamed responses, ToolRegistry with policy hooks (access control) and middleware (logging), FailSafeConfig for execution safety limits, RunnerConfig for runtime security (sanitization, output limits), dynamic InstructionProvider for context-aware behavior, subagents for delegation (analyst and planner specialists), ToolContext for injecting runtime metadata into tools, and thread-based conversation history. The project is split across four files (main.py, agents.py, tools.py, config.py) to demonstrate production code organization patterns.
---
"""

import asyncio  # <- We use asyncio because streaming (run_stream) and memory operations are both async APIs.

from afk.core import Runner  # <- Runner executes agents. We configure it with RunnerConfig for security settings.
from afk.tools.core.base import ToolContext  # <- ToolContext carries runtime info into tool functions. We set user_id and request_id here and they propagate to every tool call.
from afk.memory import new_id  # <- new_id generates unique IDs. We use it for request tracing.

from config import memory, runner_config, THREAD_ID  # <- Import shared configuration from config.py: SQLiteMemoryStore, RunnerConfig, and the thread ID constant.
from agents import coordinator  # <- Import the coordinator agent from agents.py. It has subagents, dynamic instructions, tools, and FailSafeConfig already configured.
from tools import registry  # <- Import the ToolRegistry for showing registered tools on startup.


# ===========================================================================
# Runner with production config
# ===========================================================================

runner = Runner(
    memory_store=memory,  # <- Pass the SQLiteMemoryStore to the Runner. The runner uses this for checkpointing, thread state, and conversation continuity. All tools also access this same store via the shared config import.
    config=runner_config,  # <- Apply RunnerConfig security settings: output sanitization, character limits, command allowlists, debug mode.
)


# ===========================================================================
# Streaming conversation loop
# ===========================================================================

async def main():
    """Main entry point: setup memory, run streaming conversation loop, cleanup."""

    # --- Initialize memory store ---
    await memory.setup()  # <- MUST be called before any memory operations. For SQLiteMemoryStore, this creates the database file and tables.

    # --- Show registered tools ---
    print("=" * 60)
    print("  Production Task Manager — AFK Capstone Example")
    print("=" * 60)
    print()
    print("Registered tools:")
    for t in registry.list():  # <- Show all tools managed by the ToolRegistry. This confirms the registry is loaded correctly.
        print(f"  - {t.spec.name}: {t.spec.description}")
    print()
    print("Subagents: task-analyst, task-planner")
    print(f"Memory: SQLiteMemoryStore ({memory.path})")  # <- Show which storage backend is active. For SQLite, show the file path.
    print(f"Thread: {THREAD_ID}")
    print()
    print("Features active:")
    print(f"  - Streaming: run_stream for real-time output")
    print(f"  - Persistence: SQLiteMemoryStore (survives restarts)")
    print(f"  - Security: sanitize_tool_output={runner_config.sanitize_tool_output}")
    print(f"  - Safety: max_steps=15, max_wall_time=120s")
    print(f"  - Policy: priority/category validation, delete auth")
    print(f"  - Middleware: logging for all tool calls")
    print(f"  - Delegation: analyst + planner subagents")
    print(f"  - Dynamic instructions: context-aware InstructionProvider")
    print()
    print("Commands: manage tasks, ask for stats, request analysis or planning advice.")
    print("Type 'quit' to exit.\n")

    # --- Run context (passed to InstructionProvider and available in ToolContext) ---
    run_context = {  # <- This context dict is passed to runner.run_stream(..., context=...). The InstructionProvider reads it to customize behavior, and it's available to tools via ToolContext.metadata.
        "user_name": "Developer",  # <- Personalize greetings and responses.
        "mode": "verbose",  # <- "verbose" or "brief" -- the InstructionProvider adapts the agent's behavior accordingly.
    }

    while True:
        user_input = input("[] > ")
        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        # Generate a unique request ID for tracing
        request_id = new_id("req")  # <- Each user turn gets a unique request_id. This propagates through ToolContext to every tool call, making it easy to trace a full request chain in logs.

        # -----------------------------------------------------------------
        # Streaming: run_stream returns a handle that yields events in
        # real-time as the agent processes the request.
        # -----------------------------------------------------------------
        handle = await runner.run_stream(  # <- run_stream is the streaming counterpart to run(). It returns immediately with an AgentStreamHandle you iterate over asynchronously.
            coordinator,
            user_message=user_input,
            context=run_context,  # <- Pass the run context. The InstructionProvider reads this, and it's available to tools via the runner's injection.
            thread_id=THREAD_ID,  # <- Thread ID for memory continuity. All state and events for this conversation are scoped to this thread.
        )

        print("[task-manager] > ", end="", flush=True)  # <- Print the agent name prefix, then stream text right after it.

        async for event in handle:  # <- Each iteration yields an AgentStreamEvent. The loop runs until the agent finishes.

            if event.type == "text_delta":
                # ---------------------------------------------------------
                # "text_delta" events carry incremental text chunks. Print
                # each chunk immediately for real-time feedback.
                # ---------------------------------------------------------
                print(event.text_delta, end="", flush=True)  # <- Core of streaming: each text_delta is a small piece of the response. Printing them immediately gives the user real-time feedback.

            elif event.type == "tool_started":
                # ---------------------------------------------------------
                # "tool_started" fires when the agent begins a tool call.
                # Shows which tool and enables progress indicators.
                # ---------------------------------------------------------
                print(f"\n  [calling: {event.tool_name}...]", flush=True)  # <- Show the user which tool is being called. The middleware logs more details to the console.

            elif event.type == "tool_completed":
                # ---------------------------------------------------------
                # "tool_completed" fires when a tool finishes. Check
                # tool_success for status.
                # ---------------------------------------------------------
                if event.tool_success:
                    print(f"  [{event.tool_name}: done]", flush=True)
                else:
                    print(f"  [{event.tool_name}: failed — {event.tool_error}]", flush=True)  # <- Tool errors are visible to the user. The agent also sees them and can adapt (thanks to tool_failure_policy="continue_with_error").

            elif event.type == "step_started":
                # ---------------------------------------------------------
                # "step_started" fires at the beginning of each reasoning
                # step. Shows the agent's progress through complex tasks.
                # ---------------------------------------------------------
                if event.step and event.step > 1:
                    print(f"\n  [step {event.step}]", flush=True)  # <- Show step numbers for multi-step operations. Helps users understand that the agent is still working.

            elif event.type == "completed":
                # ---------------------------------------------------------
                # "completed" fires once when the entire run finishes. The
                # event.result holds the full AgentResult with usage stats.
                # ---------------------------------------------------------
                print()  # <- Newline after the streamed text.
                if event.result:
                    usage = event.result.usage
                    print(
                        f"  [tokens: {usage.input_tokens} in / {usage.output_tokens} out | "
                        f"request: {request_id}]"
                    )  # <- Show token usage and request ID. The request_id lets you trace this entire interaction in logs.

            elif event.type == "error":
                # ---------------------------------------------------------
                # "error" fires if something went wrong during execution.
                # FailSafeConfig limits appear here as errors.
                # ---------------------------------------------------------
                print(f"\n  [error: {event.error}]", flush=True)  # <- Errors from FailSafeConfig limits (max_steps, max_wall_time, etc.) surface here.

        print()  # <- Blank line between turns for readability.

    # --- Show session summary ---
    print("\n" + "=" * 60)
    print("  Session Summary")
    print("=" * 60)

    # Show tool call history from the registry
    records = registry.recent_calls(limit=20)  # <- ToolCallRecord tracks every tool call: name, timing, success/failure. Great for post-session analysis.
    if records:
        print(f"\nTool calls ({len(records)}):")
        for rec in records:
            status = "ok" if rec.ok else f"error: {rec.error}"
            duration = rec.ended_at_s - rec.started_at_s
            print(f"  {rec.tool_name}: {status} ({duration:.3f}s)")

    # Show final task state
    all_state = await memory.list_state(thread_id=THREAD_ID, prefix="task:")
    task_count = sum(1 for k in all_state if k != "task_counter" and isinstance(all_state[k], dict))
    print(f"\nPersisted tasks: {task_count}")
    print(f"Database: {memory.path}")

    # --- Cleanup ---
    await memory.close()  # <- Clean up the memory store. For SQLiteMemoryStore, this closes the database connection and flushes pending writes. Always pair setup() with close().
    print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())  # <- asyncio.run() starts the event loop. Required because both streaming and memory are async APIs.



"""
---
Tl;dr: This capstone example combines nearly every major AFK feature into a production-ready task management system. SQLiteMemoryStore persists tasks and events to disk (surviving restarts). run_stream delivers real-time streamed responses. ToolRegistry manages tools with a policy hook (validates priorities, categories, and delete authorization) and a logging middleware (traces every tool call). FailSafeConfig enforces execution limits (15 steps, 120s wall time, 50 tool calls, circuit breaker). RunnerConfig handles runtime security (output sanitization, 8000-char limit). A dynamic InstructionProvider adapts behavior based on runtime context (user name, mode). Subagents (analyst, planner) handle delegated analytical and planning queries. ToolContext injects request_id and user_id into every tool for tracing and access control. The project is split across four files (main.py, agents.py, tools.py, config.py) to demonstrate production code organization.
---
---
What's next?
- Swap SQLiteMemoryStore for RedisMemoryStore or PostgresMemoryStore to scale to multi-process or distributed deployments. The API is identical.
- Add more subagents (e.g., a "reminder" agent that checks for overdue tasks or a "reporter" that generates weekly summaries).
- Implement custom EvalAssertion classes and run evals against the task manager to verify it handles edge cases correctly.
- Add a web frontend using run_stream over WebSockets or Server-Sent Events for a real-time task management UI.
- Experiment with different models for different subagents — use a fast, cheap model for simple lookups and a larger model for analytical queries.
- Implement InteractionProvider for human-in-the-loop approval on destructive operations (delete, complete) instead of just the policy hook.
- Add LongTermMemory to store user preferences, work patterns, and recurring tasks across sessions.
- Check out the AFK documentation to explore all the features that weren't covered here: skills, MCP servers, prompt stores, and more!
---
"""
