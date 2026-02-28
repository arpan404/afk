"""
---
name: Production Agent
description: A capstone production-ready task management system combining create_memory_store_from_env(), streaming, ToolRegistry, FailSafeConfig, RunnerConfig, InstructionProvider, InstructionRoles, Skills, MCP config, subagents, and ToolContext.
tags: [agent, runner, streaming, memory, env-factory, tools, registry, policy, middleware, subagents, config, failsafe, context, skills, mcp, instruction-role, production]
---
---
This is the capstone example for AFK — a production-ready task management agent that combines
nearly every major framework feature into one cohesive system. It demonstrates:

- **create_memory_store_from_env()** for backend-agnostic persistence (sqlite/redis/postgres/inmemory)
- **run_stream** for real-time streamed responses
- **ToolRegistry** with policy hooks (access control) and middleware (logging)
- **FailSafeConfig** for execution safety limits (steps, time, circuit breaker)
- **RunnerConfig** for runtime security (sanitization, output limits)
- **InstructionProvider** (callable) for context-aware base instructions
- **InstructionRoles** for stacked dynamic instruction augmentation (workload, time context)
- **Skills** system for loading domain knowledge from SKILL.md files
- **MCP servers** configuration for external tool server integration
- **Subagents** for delegation (analyst and planner specialists)
- **ToolContext** for injecting runtime metadata into tools

The project is split across seven files:
- main.py: Entry point with streaming conversation loop
- agents.py: Agent hierarchy with InstructionProvider, InstructionRoles, Skills, and MCP
- tools.py: ToolRegistry with policy and middleware
- config.py: create_memory_store_from_env(), RunnerConfig, FailSafeConfig
- roles.py: InstructionRole callbacks (workload awareness, time context)
- mcp_config.py: MCPServerRef definitions for external tool servers
- skills/task-ops/SKILL.md: Domain knowledge skill for task management
---
"""

import asyncio  # <- We use asyncio because streaming (run_stream) and memory operations are both async APIs.

from afk.core import Runner  # <- Runner executes agents. We configure it with RunnerConfig for security settings.
from afk.tools.core.base import ToolContext  # <- ToolContext carries runtime info into tool functions. We set user_id and request_id here and they propagate to every tool call.
from afk.memory import new_id  # <- new_id generates unique IDs. We use it for request tracing.

from config import memory, runner_config, THREAD_ID  # <- Import shared configuration from config.py: env-based MemoryStore, RunnerConfig, and the thread ID constant.
from agents import coordinator  # <- Import the coordinator agent from agents.py. It has subagents, InstructionProvider, InstructionRoles, Skills, FailSafeConfig, and tools already configured.
from tools import registry  # <- Import the ToolRegistry for showing registered tools on startup.


# ===========================================================================
# Runner with production config
# ===========================================================================

runner = Runner(
    memory_store=memory,  # <- Pass the env-based MemoryStore to the Runner. Created by create_memory_store_from_env() in config.py — change AFK_MEMORY_BACKEND env var to switch between sqlite, redis, postgres, or inmemory with zero code changes.
    config=runner_config,  # <- Apply RunnerConfig security settings: output sanitization, character limits, command allowlists, debug mode.
)


# ===========================================================================
# Streaming conversation loop
# ===========================================================================

async def main():
    """Main entry point: setup memory, run streaming conversation loop, cleanup."""

    # --- Initialize memory store ---
    await memory.setup()  # <- MUST be called before any memory operations. For SQLiteMemoryStore, this creates the database file and tables. For Redis/Postgres, this establishes the connection pool.

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
    print(f"Memory: {type(memory).__name__} (via create_memory_store_from_env)")  # <- Show which storage backend is active, resolved from AFK_MEMORY_BACKEND.
    print(f"Thread: {THREAD_ID}")
    print()
    print("Features active:")
    print(f"  - Streaming: run_stream for real-time output")
    print(f"  - Persistence: create_memory_store_from_env() (env-configurable backend)")
    print(f"  - Security: sanitize_tool_output={runner_config.sanitize_tool_output}")
    print(f"  - Safety: max_steps=15, max_wall_time=120s")
    print(f"  - Policy: priority/category validation, delete auth")
    print(f"  - Middleware: logging for all tool calls")
    print(f"  - Delegation: analyst + planner subagents")
    print(f"  - InstructionProvider: context-aware base instructions")
    print(f"  - InstructionRoles: workload awareness + time context (stacked)")
    print(f"  - Skills: task-ops domain knowledge (from SKILL.md)")
    print(f"  - MCP: configured (see mcp_config.py — uncomment to activate)")
    print()
    print("Commands: manage tasks, ask for stats, request analysis or planning advice.")
    print("Type 'quit' to exit.\n")

    # --- Run context (passed to InstructionProvider and available in ToolContext) ---
    run_context = {  # <- This context dict is passed to runner.run_stream(..., context=...). The InstructionProvider reads it to customize behavior, InstructionRoles receive it for dynamic decisions, and it's available to tools via ToolContext.metadata.
        "user_name": "Developer",  # <- Personalize greetings and responses.
        "mode": "verbose",  # <- "verbose" or "brief" — the InstructionProvider adapts the agent's behavior accordingly.
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
            context=run_context,  # <- Pass the run context. The InstructionProvider reads this, InstructionRoles receive it to generate dynamic additions, and it's available to tools via the runner's injection.
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
    print(f"Memory backend: {type(memory).__name__}")

    # --- Cleanup ---
    await memory.close()  # <- Clean up the memory store. For SQLiteMemoryStore, this closes the database connection. For Redis/Postgres, this closes the connection pool. Always pair setup() with close().
    print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())  # <- asyncio.run() starts the event loop. Required because both streaming and memory are async APIs.



"""
---
Tl;dr: This capstone example combines nearly every major AFK feature into a production-ready task
management system. create_memory_store_from_env() provides backend-agnostic persistence switchable
via AFK_MEMORY_BACKEND env var. InstructionProvider generates context-aware base instructions.
InstructionRoles (workload_awareness_role, time_context_role) append dynamic cross-cutting concerns.
Skills load domain knowledge from SKILL.md files. MCP servers connect to external tool servers.
ToolRegistry manages tools with policy hooks and logging middleware. FailSafeConfig enforces execution
safety limits. RunnerConfig handles runtime security. Subagents handle delegated queries. ToolContext
injects tracing metadata. run_stream delivers real-time output. The project is split across seven
files demonstrating production code organization.
---
---
What's next?
- Set AFK_MEMORY_BACKEND=redis (with AFK_REDIS_URL) to switch to Redis for multi-process deployments.
- Uncomment mcp_servers in agents.py and point to real MCP servers for external tool integration.
- Add more skills (e.g., "sprint-planning", "code-review") to give the agent broader domain knowledge.
- Add more InstructionRoles (e.g., "project_deadline_role" that warns about approaching deadlines).
- Implement custom EvalAssertion classes and run evals against the task manager.
- Add a web frontend using run_stream over WebSockets for a real-time task management UI.
- Experiment with different models for different subagents — fast models for lookups, larger for analysis.
---
"""
