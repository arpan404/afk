"""
---
name: Production Agent — Configuration
description: Configuration module for the production task management agent with RunnerConfig, FailSafeConfig, and SQLiteMemoryStore.
tags: [config, runner, failsafe, memory, sqlite]
---
---
This module centralizes all configuration for the production agent: RunnerConfig for runtime security, FailSafeConfig for execution safety, SQLiteMemoryStore for persistent storage, and constants used across the project. Keeping configuration in a dedicated module makes it easy to swap backends, adjust limits, and manage environment-specific settings.
---
"""

from afk.core.runner.types import RunnerConfig  # <- RunnerConfig controls runtime security: output sanitization, character limits, command allowlists, debug settings.
from afk.agents.types import FailSafeConfig  # <- FailSafeConfig controls execution safety: step limits, tool call limits, wall-clock timeouts, failure policies.
from afk.memory import SQLiteMemoryStore  # <- SQLiteMemoryStore provides persistent local storage. Data survives program restarts, unlike InMemoryMemoryStore.


# ===========================================================================
# Constants
# ===========================================================================

THREAD_ID = "production-tasks-v1"  # <- Thread ID scopes all memory to this session. All tasks, events, and state for this application live under this thread. Different applications or users would use different thread IDs.

DB_PATH = "production_agent.sqlite3"  # <- SQLite database file path. This file is created automatically on first run. For production, use an absolute path or configure via environment variable.


# ===========================================================================
# Memory store (SQLite for persistence)
# ===========================================================================

memory = SQLiteMemoryStore(path=DB_PATH)  # <- SQLiteMemoryStore persists data to a local SQLite file. Unlike InMemoryMemoryStore, data survives program restarts. The API is identical -- you can swap between them with zero code changes. For distributed systems, use RedisMemoryStore or PostgresMemoryStore instead.


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
