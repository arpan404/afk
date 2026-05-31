# AFK Python SDK Environment Variables

This reference documents environment defaults from the current source. Runtime configuration APIs remain primary.

## LLM Defaults

| Variable | Default | Description |
| --- | --- | --- |
| `AFK_LLM_PROVIDER` | `litellm` | Default provider id (`openai`, `litellm`, `anthropic_agent`) |
| `AFK_LLM_PROVIDER_ORDER` | _(none)_ | Comma-separated provider preference order |
| `AFK_LLM_MODEL` | `gpt-4.1-mini` | Default model |
| `AFK_EMBED_MODEL` | _(none)_ | Embedding model |
| `AFK_LLM_API_BASE_URL` | _(none)_ | Provider API base |
| `AFK_LLM_API_KEY` | _(none)_ | Provider API key |
| `AFK_LLM_TIMEOUT_S` | `30` | Request timeout seconds |
| `AFK_LLM_STREAM_IDLE_TIMEOUT_S` | `45` | Stream idle timeout seconds |
| `AFK_LLM_MAX_RETRIES` | `3` | Retry attempts |
| `AFK_LLM_BACKOFF_BASE_S` | `0.5` | Retry backoff base |
| `AFK_LLM_BACKOFF_JITTER_S` | `0.15` | Retry jitter |
| `AFK_LLM_JSON_MAX_RETRIES` | `2` | Structured output repair attempts |
| `AFK_LLM_MAX_INPUT_CHARS` | `200000` | Input truncation ceiling |

## Memory

| Variable | Default | Description |
| --- | --- | --- |
| `AFK_MEMORY_BACKEND` | `sqlite` | `memory`, `inmemory`, `sqlite`, `redis`, `postgres` |
| `AFK_SQLITE_PATH` | `afk_memory.sqlite3` | SQLite file path |
| `AFK_REDIS_URL` | _(none)_ | Redis URL |
| `AFK_REDIS_HOST` | `localhost` | Redis host when URL is not set |
| `AFK_REDIS_PORT` | `6379` | Redis port when URL is not set |
| `AFK_REDIS_DB` | `0` | Redis DB when URL is not set |
| `AFK_REDIS_PASSWORD` | _(none)_ | Redis password when URL is not set |
| `AFK_REDIS_EVENTS_MAX` | `2000` | Max Redis memory events per thread |
| `AFK_PG_DSN` | _(none)_ | PostgreSQL DSN |
| `AFK_PG_HOST` | `localhost` | PostgreSQL host when DSN is not set |
| `AFK_PG_PORT` | `5432` | PostgreSQL port when DSN is not set |
| `AFK_PG_USER` | `postgres` | PostgreSQL user when DSN is not set |
| `AFK_PG_PASSWORD` | _(none)_ | PostgreSQL password when DSN is not set |
| `AFK_PG_DB` | `afk` | PostgreSQL database when DSN is not set |
| `AFK_PG_SSL` | `false` | Enable PostgreSQL SSL |
| `AFK_PG_POOL_MIN` | `1` | PostgreSQL pool minimum size |
| `AFK_PG_POOL_MAX` | `10` | PostgreSQL pool maximum size |
| `AFK_VECTOR_DIM` | _(required for Postgres)_ | Vector dimension for Postgres memory search |

## Queue

| Variable | Default | Description |
| --- | --- | --- |
| `AFK_QUEUE_BACKEND` | `inmemory` | `inmemory`, `redis` |
| `AFK_QUEUE_REDIS_URL` | falls back to `AFK_REDIS_URL` | Redis URL for queue backend |
| `AFK_QUEUE_REDIS_HOST` | falls back to `AFK_REDIS_HOST`, then `localhost` | Redis queue host |
| `AFK_QUEUE_REDIS_PORT` | falls back to `AFK_REDIS_PORT`, then `6379` | Redis queue port |
| `AFK_QUEUE_REDIS_DB` | falls back to `AFK_REDIS_DB`, then `0` | Redis queue DB |
| `AFK_QUEUE_REDIS_PASSWORD` | falls back to `AFK_REDIS_PASSWORD` | Redis queue password |
| `AFK_QUEUE_REDIS_PREFIX` | `afk:queue` | Redis key prefix |
| `AFK_QUEUE_RETRY_BACKOFF_BASE_S` | `0.5` | Retry base delay |
| `AFK_QUEUE_RETRY_BACKOFF_MAX_S` | `30` | Retry max delay |
| `AFK_QUEUE_RETRY_BACKOFF_JITTER_S` | `0.2` | Retry jitter |

Execution contracts are configured in code via `TaskWorker(..., execution_contracts=...)`.

## Prompts

| Variable | Default | Description |
| --- | --- | --- |
| `AFK_AGENT_PROMPTS_DIR` | `.agents/prompt` | Prompt root directory |

## Runner and Command Tools

| Variable | Default | Description |
| --- | --- | --- |
| `AFK_ALLOWED_COMMANDS` | _(none)_ | Comma-separated default allowlist for runtime command tools |

## MCP Server

| Variable | Default | Description |
| --- | --- | --- |
| `AFK_CORS_ORIGINS` | _(none)_ | Comma-separated CORS origins |
| `AFK_MCP_NAME` | `afk-mcp-server` | Server name |
| `AFK_MCP_VERSION` | `1.0.0` | Server version |
| `AFK_MCP_HOST` | `0.0.0.0` | Bind host |
| `AFK_MCP_PORT` | `8000` | Bind port |
| `AFK_MCP_INSTRUCTIONS` | _(none)_ | Optional server instructions |
| `AFK_MCP_PATH` | `/mcp` | HTTP MCP endpoint path |
| `AFK_MCP_SSE_PATH` | `/mcp/sse` | SSE endpoint path |
| `AFK_MCP_HEALTH_PATH` | `/health` | Health endpoint path |
| `AFK_MCP_ENABLE_SSE` | `true` | Enable SSE endpoint |
| `AFK_MCP_ENABLE_HEALTH` | `true` | Enable health endpoint |
| `AFK_MCP_ALLOW_BATCH` | `true` | Allow batched MCP requests |

## A2A

No default environment variables are required. Configure A2A host/auth in code for explicit security posture.
