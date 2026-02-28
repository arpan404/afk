"""
---
name: Production Agent — Configuration
description: Configuration module with RunnerConfig, FailSafeConfig, and environment-based memory store factory.
tags: [config, runner, failsafe, memory, env-factory]
---
---
This module centralizes all configuration for the production agent: RunnerConfig for runtime
security, FailSafeConfig for execution safety, and environment-based memory store factory via
create_memory_store_from_env(). Using the factory function lets you switch backends (inmemory,
sqlite, redis, postgres) by setting AFK_MEMORY_BACKEND — zero code changes needed for
different deployment environments.
---
"""

import os
from pathlib import Path

from afk.core.runner.types import RunnerConfig  # <- RunnerConfig controls runtime security: output sanitization, character limits, command allowlists, debug settings.
from afk.agents.types import FailSafeConfig  # <- FailSafeConfig controls execution safety: step limits, tool call limits, wall-clock timeouts, failure policies.
from afk.memory import create_memory_store_from_env  # <- Factory function that reads AFK_MEMORY_BACKEND env var and returns the appropriate MemoryStore. Supports: "inmemory", "sqlite", "redis", "postgres".


# ===========================================================================
# Constants
# ===========================================================================

THREAD_ID = "production-tasks-v1"  # <- Thread ID scopes all memory to this session. All tasks, events, and state for this application live under this thread. Different applications or users would use different thread IDs.

SKILLS_DIR = Path(__file__).parent / "skills"  # <- Directory containing agent skills. Each skill is a subdirectory with a SKILL.md file.


# ===========================================================================
# Memory store (environment-based factory)
# ===========================================================================
# Set AFK_MEMORY_BACKEND to switch backends without code changes:
#   "inmemory"  -> InMemoryMemoryStore (default for development)
#   "sqlite"    -> SQLiteMemoryStore (set AFK_SQLITE_PATH for database file)
#   "redis"     -> RedisMemoryStore (set AFK_REDIS_URL for connection)
#   "postgres"  -> PostgresMemoryStore (set AFK_PG_DSN for connection)

if not os.environ.get("AFK_MEMORY_BACKEND"):
    os.environ["AFK_MEMORY_BACKEND"] = "sqlite"  # <- Default to sqlite for this production example. Change to "inmemory" for quick testing.

if not os.environ.get("AFK_SQLITE_PATH"):
    os.environ["AFK_SQLITE_PATH"] = "production_agent.sqlite3"  # <- Default SQLite file path. For production, use an absolute path or configure via deployment config.

memory = create_memory_store_from_env()  # <- Reads AFK_MEMORY_BACKEND and returns the matching MemoryStore instance. The API is identical across all backends — you can swap between them with zero code changes. For distributed systems, set AFK_MEMORY_BACKEND=redis or postgres.


# ===========================================================================
# RunnerConfig — runtime security and behavior
# ===========================================================================

runner_config = RunnerConfig(
    sanitize_tool_output=True,  # <- Clean tool output before the model sees it. Prevents prompt injection from tool results.
    tool_output_max_chars=8000,  # <- Truncate tool output to 8000 chars. Prevents data exfiltration and keeps token costs predictable.
    default_allowlisted_commands=("ls", "cat", "echo"),  # <- Only these shell commands are allowed for runtime/skill command tools.
    untrusted_tool_preamble=True,  # <- Prepend a warning to tool output so the model treats it as potentially untrusted.
    debug=True,  # <- Enable debug instrumentation for development. Shows detailed execution traces.
)


# ===========================================================================
# FailSafeConfig — execution limits and failure policies
# ===========================================================================

fail_safe_config = FailSafeConfig(
    max_steps=15,  # <- Maximum run loop iterations. Generous enough for multi-tool task management, but prevents runaway loops.
    max_wall_time_s=120.0,  # <- 2-minute wall-clock limit. Enough for complex multi-step tasks with streaming.
    max_llm_calls=30,  # <- Allows multi-step reasoning with tool calls. Each tool call typically needs 2 LLM calls (decide + respond).
    max_tool_calls=50,  # <- Generous tool limit for task management workflows that may create, list, update, and summarize in one turn.
    max_parallel_tools=8,  # <- Allow parallel tool execution for batch operations.
    max_subagent_depth=2,  # <- Allow coordinator -> specialist delegation but no deeper.
    llm_failure_policy="retry_then_fail",  # <- Retry transient LLM failures before giving up.
    tool_failure_policy="continue_with_error",  # <- Send tool errors to the model so it can adapt.
    subagent_failure_policy="continue",  # <- If a subagent fails, the parent can still produce a result.
    breaker_failure_threshold=5,  # <- Open circuit breaker after 5 consecutive failures.
    breaker_cooldown_s=30.0,  # <- Wait 30s before retrying after circuit breaker opens.
)
