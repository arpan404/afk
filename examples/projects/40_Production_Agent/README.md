
# Production Agent

A capstone example combining every major AFK feature into a production-ready task management system. Demonstrates `create_memory_store_from_env()`, streaming, `ToolRegistry`, `FailSafeConfig`, `RunnerConfig`, `InstructionProvider`, `InstructionRoles`, Skills, MCP servers, subagents, and `ToolContext`.

## Project Structure

```
40_Production_Agent/
  main.py                    # Entry point — streaming conversation loop
  agents.py                  # Agent hierarchy: coordinator, analyst, planner + InstructionProvider, InstructionRoles, Skills
  tools.py                   # ToolRegistry with policy hooks and logging middleware
  config.py                  # create_memory_store_from_env(), RunnerConfig, FailSafeConfig
  roles.py                   # InstructionRole callbacks: workload awareness, time context
  mcp_config.py              # MCPServerRef definitions for external tool servers
  skills/
    task-ops/
      SKILL.md               # Domain knowledge skill for task management best practices
```

## Key Concepts

- **create_memory_store_from_env()**: Set `AFK_MEMORY_BACKEND` env var to switch between inmemory, sqlite, redis, postgres — zero code changes
- **InstructionProvider**: Callable `(context) -> str` generates context-aware base instructions
- **InstructionRoles**: Stack dynamic instruction text on top of base instructions (workload alerts, time context)
- **Skills**: Load domain knowledge from `SKILL.md` files via `skills=["task-ops"]` and `skills_dir=`
- **MCPServerRef**: Connect to external MCP tool servers for cross-agent tool sharing
- **ToolRegistry**: Centralized tool management with policy hooks and middleware
- **FailSafeConfig**: Execution safety limits (steps, time, circuit breaker)
- **RunnerConfig**: Runtime security (sanitization, output limits)
- **Subagents**: Delegation to analyst and planner specialists
- **ToolContext**: Runtime metadata injection (request_id, user_id)
- **run_stream**: Real-time streamed responses

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/40_Production_Agent

Environment Variables
- AFK_MEMORY_BACKEND: "sqlite" (default), "inmemory", "redis", "postgres"
- AFK_SQLITE_PATH: SQLite database path (when using sqlite backend)
- AFK_REDIS_URL: Redis connection URL (when using redis backend)
- AFK_PG_DSN: PostgreSQL DSN (when using postgres backend)

Expected interaction
User: Add a task "Deploy v2.0 to staging" with high priority
Agent: [streaming response] Created task #1: "Deploy v2.0 to staging" (priority: high)
User: Show all tasks
Agent: [streaming response] Tasks: ...
User: Summarize my productivity
Agent: [delegates to analyst subagent, streams summary]
