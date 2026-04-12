# Memory and State

AFK memory subsystem: event streams, key-value state, long-term memory,
retention, compaction, and vector search.

- Docs: https://afk.arpan.sh/library/memory | /checkpoint-schema
- Source: `src/afk/memory/store.py`, `types.py`, `adapters/*.py`, `lifecycle.py`, `factory.py`, `vector.py`
- Cross-refs: `agents-and-runner.md`, `llm-configuration.md`

---

## 1. MemoryStore ABC

All methods are `async`. Initialize via `setup()` or `async with` before use.

| Method | Returns | Purpose |
|--------|---------|---------|
| `append_event(event: MemoryEvent)` | `None` | Append one event to a thread |
| `get_recent_events(thread_id, limit=50)` | `list[MemoryEvent]` | Recent events, chronological order |
| `get_events_since(thread_id, since_ms, limit=500)` | `list[MemoryEvent]` | Events newer than `since_ms` (epoch ms) |
| `put_state(thread_id, key, value)` | `None` | Set a thread-scoped key-value pair |
| `get_state(thread_id, key)` | `JsonValue \| None` | Read a single state key |
| `list_state(thread_id, prefix=None)` | `dict[str, JsonValue]` | List state keys, optionally by prefix |
| `upsert_long_term_memory(memory, *, embedding=None)` | `None` | Insert or update a long-term memory |
| `delete_long_term_memory(user_id, memory_id)` | `None` | Delete one long-term memory |
| `list_long_term_memories(user_id, *, scope=None, limit=100)` | `list[LongTermMemory]` | List memories for user + optional scope |
| `search_long_term_memory_text(user_id, query, *, scope=None, limit=20)` | `list[LongTermMemory]` | Text search over long-term memories |
| `search_long_term_memory_vector(user_id, query_embedding, *, scope=None, limit=20, min_score=None)` | `list[tuple[LongTermMemory, float]]` | Vector similarity search |

```python
# CORRECT: Use async with for automatic setup/close
async with InMemoryMemoryStore() as store:
    await store.put_state("thread-1", "step", 3)
    val = await store.get_state("thread-1", "step")  # 3

# WRONG: Using the store without initialization
store = InMemoryMemoryStore()
await store.put_state("thread-1", "step", 3)  # RuntimeError: not initialized
```

---

## 2. Backend Adapters

| Backend | Class | Constructor | When to Use |
|---------|-------|-------------|-------------|
| In-memory | `InMemoryMemoryStore` | `InMemoryMemoryStore()` | Tests, local dev, ephemeral |
| SQLite | `SQLiteMemoryStore` | `SQLiteMemoryStore(db_path: str)` | Single-process, persistent |
| Redis | `RedisMemoryStore` | `RedisMemoryStore(url=None, host="localhost", port=6379, db=0, password=None)` | Multi-process, low-latency, TTL |
| Postgres | `PostgresMemoryStore` | `PostgresMemoryStore(dsn: str, ...)` | Production, pgvector, pooling |

| Capability | InMemory | SQLite | Redis | Postgres |
|------------|----------|--------|-------|----------|
| `text_search` / `vector_search` / `atomic_upsert` | Yes | Yes | Yes | Yes (pgvector) |
| `ttl` | No | No | Yes | No |

```python
# CORRECT: Import from the public API
from afk.memory import InMemoryMemoryStore, SQLiteMemoryStore
from afk.memory import RedisMemoryStore, PostgresMemoryStore  # optional extras

# WRONG: Import from internal adapter paths
from afk.memory.adapters.sqlite import SQLiteMemoryStore  # Never use internal paths
```

---

## 3. Factory and Environment Configuration

`create_memory_store_from_env()` selects a backend from environment variables.

```python
store = create_memory_store_from_env()  # reads AFK_MEMORY_BACKEND
async with store:
    events = await store.get_recent_events("thread-1")
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `AFK_MEMORY_BACKEND` | `sqlite` | `inmemory` \| `sqlite` \| `redis` \| `postgres` |
| `AFK_SQLITE_PATH` | `afk_memory.sqlite3` | SQLite file path (or `":memory:"`) |
| `AFK_REDIS_URL` | -- | Full Redis URL (precedence over individual vars) |
| `AFK_REDIS_HOST`/`PORT`/`DB`/`PASSWORD` | `localhost`/`6379`/`0`/-- | Redis connection parts |
| `AFK_PG_DSN` | -- | Full Postgres DSN (precedence over individual vars) |
| `AFK_PG_HOST`/`PORT`/`USER`/`PASSWORD`/`DB` | `localhost`/`5432`/`postgres`/--/`afk` | Postgres connection |
| `AFK_PG_SSL` / `POOL_MIN` / `POOL_MAX` | `false`/`1`/`10` | SSL and pool bounds |
| `AFK_VECTOR_DIM` | -- | Embedding dimension (**required** for Postgres) |

```python
# CORRECT: Let the factory pick the backend from env
os.environ["AFK_MEMORY_BACKEND"] = "postgres"
os.environ["AFK_PG_DSN"] = "postgresql://user:pass@db:5432/afk"
os.environ["AFK_VECTOR_DIM"] = "1536"
store = create_memory_store_from_env()

# WRONG: Hardcoding backend selection per environment
if env == "prod":
    store = PostgresMemoryStore(dsn=dsn, vector_dim=1536)
else:
    store = SQLiteMemoryStore("dev.sqlite3")  # Use the factory instead
```

---

## 4. Memory Types

### MemoryEvent -- short-term, per-thread, immutable, append-only

```python
@dataclass(frozen=True, slots=True)
class MemoryEvent:
    id: str                    # Unique event ID
    thread_id: str             # Conversation thread
    user_id: str | None        # Optional user association
    type: EventType            # "tool_call" | "tool_result" | "message" | "system" | "trace"
    timestamp: int             # Epoch milliseconds
    payload: JsonObject        # Arbitrary JSON data
    tags: list[str] = []       # Optional tags for filtering
```

```python
# CORRECT: Create events with typed MemoryEvent objects
from afk.memory import MemoryEvent
event = MemoryEvent(
    id="evt-001", thread_id="thread-abc", user_id="user-42",
    type="message", timestamp=1700000000000,
    payload={"role": "user", "content": "Hello"}, tags=["input"],
)
await store.append_event(event)

# WRONG: Passing a raw dict instead of a MemoryEvent
await store.append_event({"id": "evt-001", "type": "message", ...})  # TypeError
```

### LongTermMemory -- durable, user-scoped, for retrieval and RAG

```python
@dataclass(frozen=True, slots=True)
class LongTermMemory:
    id: str                    # Unique memory ID
    user_id: str | None        # Owner (None for system-wide)
    scope: str                 # e.g. "global", "org:123", "project:abc"
    data: JsonObject           # Structured memory content
    text: str | None = None    # Optional plaintext for search
    tags: list[str] = []       # Optional tags
    metadata: JsonObject = {}  # Extra metadata
    created_at: int            # Epoch milliseconds
    updated_at: int            # Epoch milliseconds
```

```python
# CORRECT: Include text field for searchability
from afk.memory import LongTermMemory
mem = LongTermMemory(
    id="mem-001", user_id="user-42", scope="project:web-app",
    data={"preference": "dark_mode", "value": True},
    text="User prefers dark mode for the web-app project",
    created_at=1700000000000, updated_at=1700000000000,
)
await store.upsert_long_term_memory(mem)

# WRONG: Omitting scope (required field)
mem = LongTermMemory(id="mem-001", user_id="user-42", data={...})  # TypeError
```

---

## 5. Thread Continuity

`thread_id` groups events and state across Runner invocations for multi-turn
conversations and checkpoint-based resume.

```python
# CORRECT: Reuse thread_id across Runner calls for continuity
result1 = await runner.run(agent, input="Hello", thread_id="t-100", memory=store)
result2 = await runner.run(agent, input="Follow up", thread_id="t-100", memory=store)

# WRONG: New thread_id per turn (loses context)
result1 = await runner.run(agent, input="Hello", thread_id="t-100", memory=store)
result2 = await runner.run(agent, input="Follow up", thread_id="t-200", memory=store)
# Agent has no memory of the first turn
```

Thread-scoped state is used by the Runner for checkpointing; store custom
application state under your own key prefixes:

```python
await store.put_state("t-100", "app:user_lang", "en")  # set
lang = await store.get_state("t-100", "app:user_lang")  # "en"
app_state = await store.list_state("t-100", prefix="app:")  # {"app:user_lang": "en"}
```

---

## 6. Retention and Compaction

```python
class RetentionPolicy:
    max_events_per_thread: int = 5000        # Hard cap per thread
    keep_event_types: list[str] = ["trace"]  # Protected types (never evicted first)
    scan_limit: int = 20_000                 # Max events to scan during compaction

class StateRetentionPolicy:
    max_runs: int = 100                      # Keep N most-recent runs
    max_runtime_states_per_run: int = 3      # runtime_state rows per run
    max_effect_entries_per_run: int = 3000   # Effect log entries per run
    always_keep_phases: list[str] = [...]    # Phases never evicted
    keep_state_prefixes: list[str] = []      # Custom prefixes always retained
```

`compact_thread_memory()` returns a `MemoryCompactionResult`:

| Field | Type | Meaning |
|-------|------|---------|
| `events_before` / `events_after` / `events_removed` | `int` | Event counts |
| `state_keys_before` / `state_keys_after` / `state_keys_removed` | `int` | State key counts |
| `state_keys_removed_effective` | `int` | Keys actually deleted (backend may not support deletion) |

```python
# CORRECT: Use compact_thread_memory with policies
from afk.memory import compact_thread_memory, RetentionPolicy, StateRetentionPolicy

result = await compact_thread_memory(
    store, thread_id="t-100",
    event_policy=RetentionPolicy(max_events_per_thread=1000),
    state_policy=StateRetentionPolicy(keep_state_prefixes=["app:"]),
)
print(f"Removed {result.events_removed} events, {result.state_keys_removed} state keys")

# WRONG: Manually deleting events instead of using compaction
events = await store.get_recent_events("t-100", limit=10000)
for e in events[:5000]:
    ...  # No delete_event API -- use compact_thread_memory()
```

---

## 7. Vector Search

All backends support vector search. Postgres uses native pgvector with HNSW
indexing; others compute cosine similarity in Python. Pass `embedding` when
upserting; if omitted on subsequent upserts the existing embedding is preserved.

```python
embedding = await my_model.embed("User prefers dark mode")  # list[float]
await store.upsert_long_term_memory(mem, embedding=embedding)
```

Returns `(LongTermMemory, float)` tuples -- cosine similarity in [-1, 1], higher is better.

```python
# CORRECT: Use min_score to filter low-relevance results server-side
results = await store.search_long_term_memory_vector(
    user_id="user-42", query_embedding=query_vec,
    scope="global", limit=5, min_score=0.7,
)
for memory, score in results:
    print(f"{memory.id}: {score:.3f} -- {memory.text}")

# WRONG: Fetching all memories and filtering manually in Python
all_mems = await store.list_long_term_memories("user-42", limit=10000)
for m in all_mems:
    sim = cosine_similarity(query_vec, get_embedding_somehow(m))
    # Defeats the purpose of indexed search -- use the store method
```

Standalone cosine utility: `cosine_similarity([1.0, 0.0], [0.0, 1.0])  # 0.0`

---

## Imports Quick Reference

```python
from afk.memory import (
    MemoryStore, InMemoryMemoryStore,       # ABC + default backend
    SQLiteMemoryStore,                      # SQLite backend
    RedisMemoryStore,                      # Redis backend (optional)
    PostgresMemoryStore,                   # Postgres backend (optional)
    MemoryEvent, LongTermMemory,            # Data models
    RetentionPolicy, StateRetentionPolicy,  # Retention policies
    MemoryCompactionResult,                # Compaction result
    compact_thread_memory, apply_event_retention, apply_state_retention,  # Lifecycle
    create_memory_store_from_env,           # Factory
    cosine_similarity,                    # Vector utility
)
```
