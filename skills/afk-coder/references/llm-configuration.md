# LLM Configuration

AFK LLM subsystem: provider-portable model strings, fluent builder API,
environment-driven settings, named profiles, and enterprise runtime policies.

- Docs: https://afk.arpan.sh/library/llm-interaction | https://afk.arpan.sh/library/configuration-reference
- Source: `src/afk/llms/builder.py`, `src/afk/llms/settings.py`, `src/afk/llms/profiles.py`, `src/afk/llms/runtime/client.py`
- Cross-refs: `agents-and-runner.md`, `streaming-and-interaction.md`

---

## 1. Overview

AFK uses **provider-portable model strings** so application code never couples
to a single LLM vendor. The `LLMBuilder` class provides a fluent API for
constructing production-ready `LLMClient` instances with retry, circuit
breaker, rate limiting, caching, and hedging policies baked in. `LLMSettings`
centralizes runtime defaults and loads them from environment variables.

Key public imports:

```python
from afk.llms import LLMBuilder, LLMSettings, LLMClient, LLM, MiddlewareStack
```

---

## 2. Model Strings

Format: `provider/model_name` or just `model_name` (uses default provider).

| String | Provider | Notes |
|--------|----------|-------|
| `gpt-4.1-mini` | litellm (default) | Routed through LiteLLM |
| `claude-sonnet-4` | litellm | Anthropic model via LiteLLM routing |
| `ollama_chat/llama3:8b` | litellm | Local Ollama backend |
| `anthropic/claude-sonnet-4` | litellm | Explicit LiteLLM provider prefix |

The default provider is `litellm`, which supports hundreds of model backends
through a unified interface. When a model string contains a `/`, LiteLLM
interprets the prefix as the backend provider.

```python
# CORRECT: Use portable model strings
agent = Agent(model="gpt-4.1-mini", ...)

# WRONG: Import and instantiate a provider-specific SDK client
import openai
client = openai.OpenAI()  # couples you to one vendor
```

---

## 3. Agent Model Configuration

Agents accept either a plain model string or a pre-built `LLM` instance.

```python
from afk.agents import Agent

# Option A: Simple string (recommended for most cases)
agent = Agent(
    model="gpt-4.1-mini",
    name="my-agent",
    instructions="You are a helpful assistant.",
)

# Option B: Pre-built LLM instance for advanced adapter control
from afk.llms import LLM
llm = LLM(model="gpt-4.1-mini", provider="litellm")
agent = Agent(
    model=llm,
    name="my-agent",
    instructions="You are a helpful assistant.",
)
```

The `model` parameter on `BaseAgent.__init__` is typed as `str | LLM`.
When a plain string is provided, the runner resolves it through the configured
`ModelResolver` at execution time.

---

## 4. LLMBuilder (Fluent API)

`LLMBuilder` is the recommended way to construct an `LLMClient` with full
runtime policy control.

```python
from afk.llms import LLMBuilder

client = (
    LLMBuilder()
    .provider("litellm")
    .model("gpt-4.1-mini")
    .profile("production")
    .with_cache(cache_backend)
    .with_middlewares(middleware_stack)
    .with_observers([my_observer])
    .build()
)
```

### Builder Methods

All builder methods return `self` for chaining. Call `.build()` last to
materialize the `LLMClient`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `provider` | `(str) -> LLMBuilder` | Set default provider id (e.g. `"litellm"`, `"openai"`, `"anthropic_agent"`) |
| `model` | `(str) -> LLMBuilder` | Override default model in builder settings |
| `settings` | `(LLMSettings) -> LLMBuilder` | Replace builder settings with an explicit `LLMSettings` instance |
| `profile` | `(str) -> LLMBuilder` | Apply a named runtime profile (`development`, `production`, `high_throughput`, `low_latency`) |
| `for_agent_runtime` | `() -> LLMBuilder` | Shortcut for `.profile("production")` -- one-call production baseline |
| `with_provider_settings` | `(str, Mapping) -> LLMBuilder` | Attach provider-specific configuration passed to provider factory hooks |
| `with_middlewares` | `(MiddlewareStack) -> LLMBuilder` | Configure middleware stacks for chat/stream/embed transport paths |
| `with_observers` | `(list[LLMObserver]) -> LLMBuilder` | Configure lifecycle observers for best-effort telemetry callbacks |
| `with_cache` | `(backend) -> LLMBuilder` | Select a cache backend instance or registered backend id |
| `with_router` | `(router) -> LLMBuilder` | Select a router instance or registered router id |
| `build` | `() -> LLMClient` | Materialize one configured `LLMClient` instance |

```python
# CORRECT: Use the builder for production clients
client = (
    LLMBuilder()
    .provider("litellm")
    .model("gpt-4.1-mini")
    .for_agent_runtime()
    .build()
)

# WRONG: Manually wiring retry/timeout/breaker everywhere
from afk.llms import LLMClient, RetryPolicy, TimeoutPolicy
client = LLMClient(
    provider="litellm",
    settings=LLMSettings(),
    retry_policy=RetryPolicy(max_retries=3, backoff_base_s=0.5, backoff_jitter_s=0.15),
    timeout_policy=TimeoutPolicy(request_timeout_s=30.0, stream_idle_timeout_s=45.0),
    # ... repetitive boilerplate
)
```

---

## 5. LLMSettings

`LLMSettings` is a frozen dataclass that holds all tunable runtime defaults.
Load from environment variables with `LLMSettings.from_env()` (called
automatically by `LLMBuilder()`).

| Field | Type | Default | Env Var |
|-------|------|---------|---------|
| `default_provider` | `str` | `"litellm"` | `AFK_LLM_PROVIDER` |
| `default_model` | `str` | `"gpt-4.1-mini"` | `AFK_LLM_MODEL` |
| `embedding_model` | `str \| None` | `None` | `AFK_EMBED_MODEL` |
| `api_base_url` | `str \| None` | `None` | `AFK_LLM_API_BASE_URL` |
| `api_key` | `str \| None` | `None` | `AFK_LLM_API_KEY` |
| `timeout_s` | `float` | `30.0` | `AFK_LLM_TIMEOUT_S` |
| `max_retries` | `int` | `3` | `AFK_LLM_MAX_RETRIES` |
| `backoff_base_s` | `float` | `0.5` | `AFK_LLM_BACKOFF_BASE_S` |
| `backoff_jitter_s` | `float` | `0.15` | `AFK_LLM_BACKOFF_JITTER_S` |
| `json_max_retries` | `int` | `2` | `AFK_LLM_JSON_MAX_RETRIES` |
| `max_input_chars` | `int` | `200000` | `AFK_LLM_MAX_INPUT_CHARS` |
| `stream_idle_timeout_s` | `float \| None` | `45.0` | `AFK_LLM_STREAM_IDLE_TIMEOUT_S` |

```python
from afk.llms import LLMSettings

# Load from environment (default behavior inside LLMBuilder)
settings = LLMSettings.from_env()

# Override specific fields
from dataclasses import replace
custom = replace(settings, timeout_s=60.0, max_retries=5)
client = LLMBuilder().settings(custom).build()
```

```python
# CORRECT: Use env vars for secrets
# export AFK_LLM_API_KEY=sk-...
settings = LLMSettings.from_env()

# WRONG: Hard-code API keys in source
settings = LLMSettings(api_key="sk-abc123hardcoded")
```

---

## 6. Profiles

Profiles are pre-built collections of runtime policies. Apply one via
`.profile(name)` on the builder.

| Profile | Retry | Timeout (req/stream) | Rate Limit (rps/burst) | Breaker (threshold/cooldown) | Hedging | Cache | Coalescing |
|---------|-------|----------------------|------------------------|------------------------------|---------|-------|------------|
| `development` | 1 retry, 0.2s base | 30s / 90s | 50 / 100 | 8 failures / 10s | Off | Off | On |
| `production` | 3 retries, 0.5s base | 30s / 45s | 20 / 40 | 5 failures / 30s | Off | Off | On |
| `high_throughput` | 2 retries, 0.3s base | 20s / 40s | 120 / 200 | 10 failures / 20s | Off | On (20s TTL) | On |
| `low_latency` | 1 retry, 0.2s base | 10s / 20s | 30 / 60 | 3 failures / 15s | On (0.08s delay) | On (10s TTL) | On |

```python
from afk.llms import LLMBuilder

# Development: relaxed timeouts, minimal retry for local iteration
dev_client = LLMBuilder().profile("development").build()

# Production: balanced retry + circuit breaker + rate limiting
prod_client = LLMBuilder().profile("production").build()

# High throughput: high concurrency with response caching
batch_client = LLMBuilder().profile("high_throughput").build()

# Low latency: fast timeout, hedging enabled, aggressive circuit breaker
realtime_client = LLMBuilder().profile("low_latency").build()
```

Profiles are defined in `afk.llms.PROFILES` and can be inspected at runtime:

```python
from afk.llms import PROFILES
print(list(PROFILES.keys()))  # ['development', 'production', 'high_throughput', 'low_latency']
```

---

## 7. Runtime Policies (LLMClient Internals)

`LLMClient` composes multiple enterprise execution policies. Each policy is
a frozen dataclass imported from `afk.llms`.

| Policy | Class | Key Fields | Purpose |
|--------|-------|------------|---------|
| Retry | `RetryPolicy` | `max_retries`, `backoff_base_s`, `backoff_jitter_s`, `require_idempotency_key` | Exponential backoff with jitter on transient errors |
| Timeout | `TimeoutPolicy` | `request_timeout_s`, `stream_idle_timeout_s` | Per-request and stream idle deadlines |
| Rate Limit | `RateLimitPolicy` | `requests_per_second`, `burst` | Token bucket per provider and operation |
| Circuit Breaker | `CircuitBreakerPolicy` | `failure_threshold`, `cooldown_s`, `half_open_max_calls` | Consecutive failure detection with half-open probes |
| Hedging | `HedgingPolicy` | `enabled`, `delay_s` | Speculative secondary call to reduce tail latency |
| Cache | `CachePolicy` | `enabled`, `ttl_s`, `namespace` | Response cache with configurable TTL and optional tenant/user namespace |
| Coalescing | `CoalescingPolicy` | `enabled` | In-flight request deduplication for identical payloads |

Policies are set via profiles or individually through the builder:

```python
from afk.llms import LLMBuilder, RetryPolicy, TimeoutPolicy

client = (
    LLMBuilder()
    .profile("production")
    .build()
)
# The production profile sets:
#   RetryPolicy(max_retries=3, backoff_base_s=0.5, backoff_jitter_s=0.15)
#   TimeoutPolicy(request_timeout_s=30.0, stream_idle_timeout_s=45.0)
#   CircuitBreakerPolicy(failure_threshold=5, cooldown_s=30.0)
#   RateLimitPolicy(requests_per_second=20.0, burst=40)
#   CoalescingPolicy(enabled=True)
```

---

## 8. LLMClient Usage

`LLMClient` is the high-level runtime entry point. It supports non-streaming
chat, streaming chat, stream handles, and embeddings.

```python
from afk.llms import LLMBuilder, LLMRequest, Message

client = LLMBuilder().provider("litellm").model("gpt-4.1-mini").for_agent_runtime().build()

# Non-streaming chat
request = LLMRequest(
    model="gpt-4.1-mini",
    messages=[Message(role="user", content="Hello!")],
)
response = await client.chat(request)
print(response.text)

# Synchronous variant
response = client.chat_sync(request)

# Streaming chat
async for event in await client.chat_stream(request):
    print(event)

# Stream handle with cancel/interrupt control
handle = await client.chat_stream_handle(request)
async for event in handle.events:
    print(event)
```

---

## 9. Environment Variable Quick Reference

Set these before process start or in your `.env` file:

```bash
# Required for remote providers
export AFK_LLM_API_KEY="sk-..."

# Optional overrides
export AFK_LLM_PROVIDER="litellm"          # default provider
export AFK_LLM_MODEL="gpt-4.1-mini"        # default model
export AFK_EMBED_MODEL="text-embedding-3-small"  # embedding model
export AFK_LLM_API_BASE_URL="https://my-proxy.example.com/v1"
export AFK_LLM_TIMEOUT_S="60"              # request timeout
export AFK_LLM_MAX_RETRIES="5"             # retry count
export AFK_LLM_BACKOFF_BASE_S="1.0"        # retry backoff base
export AFK_LLM_BACKOFF_JITTER_S="0.3"      # retry jitter
export AFK_LLM_JSON_MAX_RETRIES="3"        # structured output repair retries
export AFK_LLM_MAX_INPUT_CHARS="400000"    # input size limit
export AFK_LLM_STREAM_IDLE_TIMEOUT_S="60"  # stream idle deadline
```

---

## 10. CORRECT / WRONG Patterns

### Model portability

```python
# CORRECT: Use model strings for provider portability
agent = Agent(model="gpt-4.1-mini", ...)
agent = Agent(model="claude-sonnet-4", ...)
agent = Agent(model="ollama_chat/llama3:8b", ...)

# WRONG: Hard-code provider-specific SDK calls
import openai
response = openai.chat.completions.create(model="gpt-4.1-mini", ...)
```

### Production configuration

```python
# CORRECT: Use profiles for production deployments
client = LLMBuilder().for_agent_runtime().build()
client = LLMBuilder().profile("production").build()

# WRONG: Manually configuring timeout/retry/breaker in every call site
client = LLMClient(
    provider="litellm",
    settings=LLMSettings(),
    retry_policy=RetryPolicy(max_retries=3, ...),
    timeout_policy=TimeoutPolicy(request_timeout_s=30.0, ...),
    circuit_breaker_policy=CircuitBreakerPolicy(failure_threshold=5, ...),
    rate_limit_policy=RateLimitPolicy(requests_per_second=20.0, ...),
)
```

### Secret management

```python
# CORRECT: Use environment variables for API keys
# export AFK_LLM_API_KEY=sk-...
settings = LLMSettings.from_env()

# WRONG: Hard-code API keys in source code
settings = LLMSettings(api_key="sk-abc123-do-not-do-this")
```

### Settings override

```python
# CORRECT: Override settings via dataclasses.replace
from dataclasses import replace
settings = replace(LLMSettings.from_env(), timeout_s=60.0)
client = LLMBuilder().settings(settings).build()

# WRONG: Mutating settings (LLMSettings is frozen)
settings = LLMSettings.from_env()
settings.timeout_s = 60.0  # FrozenInstanceError
```

---

## 11. Cross-References

- **Agents and Runner**: See [agents-and-runner.md](./agents-and-runner.md) for how agents consume model configuration and how the runner resolves model strings.
- **Streaming and Interaction**: See [streaming-and-interaction.md](./streaming-and-interaction.md) for `LLMStreamHandle`, stream events, and session control.
- **Memory and State**: See [memory-and-state.md](./memory-and-state.md) for the memory subsystem that agents use alongside LLM calls.

---

## 12. Source Files

| File | Purpose |
|------|---------|
| `src/afk/llms/__init__.py` | Public API surface and re-exports |
| `src/afk/llms/builder.py` | `LLMBuilder` fluent construction API |
| `src/afk/llms/settings.py` | `LLMSettings` dataclass and env loading |
| `src/afk/llms/profiles.py` | Named profile definitions (`PROFILES` dict) |
| `src/afk/llms/llm.py` | `LLM` base class (provider adapter contract) |
| `src/afk/llms/runtime/client.py` | `LLMClient` enterprise runtime |
| `src/afk/llms/runtime/contracts.py` | Policy dataclasses (`RetryPolicy`, `TimeoutPolicy`, etc.) |
| `src/afk/llms/middleware.py` | `MiddlewareStack` for transport-level interception |
| `src/afk/llms/observability.py` | `LLMObserver` and `LLMLifecycleEvent` |

Documentation:
- https://afk.arpan.sh/library/llm-interaction
- https://afk.arpan.sh/library/configuration-reference
- Doc source files: `docs/library/llm-interaction.mdx`, `docs/library/configuration-reference.mdx`
