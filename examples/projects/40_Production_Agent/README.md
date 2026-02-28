
# Production Agent

A capstone example combining multiple advanced AFK features into a production-ready task management system. Demonstrates SQLiteMemoryStore for persistence, streaming (run_stream) for real-time output, ToolRegistry with policy and middleware, FailSafeConfig for safety limits, RunnerConfig with debug settings, dynamic InstructionProvider, subagents for delegation, and ToolContext for runtime info.

This is a multi-file project:
- main.py — Entry point with streaming conversation loop
- agents.py — Agent definitions (coordinator + specialist subagents)
- tools.py — Tool definitions with ToolRegistry, policy, and middleware
- config.py — Configuration (RunnerConfig, FailSafeConfig, memory setup)

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run (relative):
  ./scripts/setup_example.sh --project-dir=examples/projects/40_Production_Agent

- Run (absolute):
  ./scripts/setup_example.sh --project-dir=/Users/username/pathtoafk/examples/projects/40_Production_Agent

Tip: build the absolute path dynamically from the repo root:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/40_Production_Agent

Expected interaction
User: Add a task "Deploy v2.0 to staging" with high priority
Agent: [streaming response] Created task #1: "Deploy v2.0 to staging" (priority: high)
User: Show all tasks
Agent: [streaming response] Tasks: ...
User: Summarize my productivity
Agent: [delegates to analyst subagent, streams summary]

The agent persists tasks in SQLite, streams responses token-by-token, and delegates analytical queries to specialist subagents.
