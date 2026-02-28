
# Chat History Manager

A multi-thread chat manager demonstrating `create_memory_store_from_env()` factory, `Runner.resume()` checkpoint restoration, and the full memory lifecycle API.

## Project Structure

```
38_Chat_History_Manager/
  main.py       # Entry point — interactive chat + Runner.resume() demo
  tools.py      # Thread management tools (list, switch, history, clear, info)
  config.py     # Memory store via create_memory_store_from_env()
```

## Key Concepts

- **create_memory_store_from_env()**: Set `AFK_MEMORY_BACKEND` env var to switch backends (inmemory, sqlite, redis, postgres) with zero code changes
- **Runner.resume()**: `resume(agent, run_id=..., thread_id=...)` restores a checkpointed run after interruption
- **Full memory API**: append_event, get_recent_events, put_state/get_state, list_state, replace_thread_events

Prerequisites
- Run this from the repository root.
- Ensure scripts/setup_example.sh is executable: chmod +x scripts/setup_example.sh

Usage
- Run:
  ./scripts/setup_example.sh --project-dir=$(pwd)/examples/projects/38_Chat_History_Manager

Environment Variables
- AFK_MEMORY_BACKEND: "inmemory" (default for this example), "sqlite", "redis", "postgres"
- AFK_SQLITE_PATH: SQLite database path (when using sqlite backend)

Modes
1. Interactive chat session (multi-thread conversation)
2. Runner.resume() demonstration (checkpoint restoration)
