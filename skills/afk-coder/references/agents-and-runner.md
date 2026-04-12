# Agents and Runner

AFK agent execution model: Agent declares intent, Runner executes it,
Runtime wires the lifecycle.

- Docs: https://afk.arpan.sh/library/agents | https://afk.arpan.sh/library/core-runner
- Source: `src/afk/agents/core/base.py`, `src/afk/core/runner/api.py`, `src/afk/core/runner/types.py`
- Cross-refs: `tools-system.md`, `memory-and-state.md`, `streaming-and-interaction.md`, `security-and-policies.md`

---

## Overview

AFK separates configuration from execution through three pillars:

| Pillar | Role | Key Class |
|--------|------|-----------|
| **Agent** | Stateless declaration of model, tools, instructions, limits, and skills | `Agent` |
| **Runner** | Stateful execution engine that drives the agent loop, manages memory, and enforces policies | `Runner` |
| **Runtime** | Internal orchestration layer (tool dispatch, subagent routing, checkpointing) -- not used directly | -- |

An `Agent` is a pure configuration object. It stores what the agent _is_ but
never runs anything itself. A `Runner` takes an `Agent`, boots the runtime
loop, calls the LLM, executes tools, manages memory, respects fail-safes,
and returns an `AgentResult` when the run terminates.

---

## Quick Start

```python
from afk.agents import Agent
from afk.core.runner import Runner

agent = Agent(model="gpt-4.1-mini", instructions="You are a helpful assistant.")
runner = Runner()
result = runner.run_sync(agent, user_message="Hello!")
print(result.final_text)
```

Async equivalent:

```python
import asyncio
from afk.agents import Agent
from afk.core.runner import Runner

async def main():
    agent = Agent(model="gpt-4.1-mini", instructions="You are a helpful assistant.")
    runner = Runner()
    result = await runner.run(agent, user_message="Hello!")
    print(result.final_text)

asyncio.run(main())
```

---

## Agent Declaration

`Agent` is the concrete class developers use. It extends `BaseAgent` with no
additional logic -- all constructor parameters live on `BaseAgent.__init__`.

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str \| LLM` | **(required)** | Model identifier (e.g. `"gpt-4.1-mini"`, `"claude-sonnet-4"`) or pre-built `LLM` adapter instance |
| `name` | `str \| None` | `None` | Logical agent name used in traces and subagent routing. Defaults to the class name |
| `tools` | `list[ToolLike] \| None` | `None` | Tools (or callables returning tools) the agent can invoke through LLM tool-calls |
| `subagents` | `list[BaseAgent] \| None` | `None` | Child agents available for delegation / subagent routing |
| `instructions` | `str \| InstructionProvider \| None` | `None` | Static instruction string or async/sync callable resolved per run |
| `instruction_file` | `str \| Path \| None` | `None` | Prompt filename/path under `prompts_dir`, used when `instructions` is not set |
| `prompts_dir` | `str \| Path \| None` | `None` | Root directory for system prompt files. Falls back to env `AFK_AGENT_PROMPTS_DIR`, then `.agents/prompt` |
| `context_defaults` | `dict[str, JSONValue] \| None` | `None` | Default JSON-safe context merged into each run before caller-provided context |
| `inherit_context_keys` | `list[str] \| None` | `None` | Context keys this agent accepts from a parent when used as a subagent |
| `model_resolver` | `ModelResolver \| None` | `None` | Optional override resolver for model string normalization |
| `skills` | `list[str] \| None` | `None` | Skill names to resolve under `skills_dir` (each skill has a `SKILL.md`) |
| `mcp_servers` | `list[MCPServerLike] \| None` | `None` | External MCP server refs (string URL, `name=url`, dict config, or `MCPServerRef`) whose tools are exposed to this agent |
| `skills_dir` | `str \| Path` | `".agents/skills"` | Root directory for skills (`<skill>/SKILL.md`) |
| `instruction_roles` | `list[InstructionRole] \| None` | `None` | Callbacks that append dynamic instruction text at run time |
| `policy_roles` | `list[PolicyRole] \| None` | `None` | Callbacks that can allow/deny/defer runtime actions |
| `policy_engine` | `PolicyEngine \| None` | `None` | Deterministic rule engine applied before policy roles |
| `subagent_router` | `SubagentRouter \| None` | `None` | Router callback deciding which subagents to execute |
| `max_steps` | `int` | `20` | Maximum reasoning/tool loop steps. Must be >= 1 |
| `tool_parallelism` | `int \| None` | `None` | Max concurrent tool calls. When `None`, uses `fail_safe.max_parallel_tools` |
| `subagent_parallelism_mode` | `SubagentParallelismMode` | `"configurable"` | `"single"`, `"parallel"`, or `"configurable"` (follows router decision) |
| `fail_safe` | `FailSafeConfig \| None` | `None` | Runtime limits and failure policies. Defaults to `FailSafeConfig(max_steps=max_steps)` |
| `reasoning_enabled` | `bool \| None` | `None` | Enable extended thinking / reasoning mode |
| `reasoning_effort` | `str \| None` | `None` | Reasoning effort label (provider-specific) |
| `reasoning_max_tokens` | `int \| None` | `None` | Max token budget for reasoning/thinking |
| `skill_tool_policy` | `SkillToolPolicy \| None` | `None` | Security/limits policy for built-in skill tools. Defaults to `SkillToolPolicy()` |
| `enable_skill_tools` | `bool` | `True` | Whether to auto-register built-in skill tools |
| `enable_mcp_tools` | `bool` | `True` | Whether to auto-register tools from configured external MCP servers |
| `runner` | `Runner \| None` | `None` | Optional runner override; defaults to `Runner()` when calling `agent.call()` |

### Agent.call() Shorthand

`Agent.call()` is an async convenience that creates a `Runner` (or uses the
one attached via the `runner` parameter) and calls `runner.run()`.

```python
from afk.agents import Agent

agent = Agent(model="gpt-4.1-mini", instructions="Be concise.")
result = await agent.call("Hello!", context={"user_id": "u1"}, thread_id="t1")
print(result.final_text)
```

Signature:

```python
async def call(
    self,
    user_message: str | None = None,
    *,
    context: dict[str, JSONValue] | None = None,
    thread_id: str | None = None,
) -> AgentResult
```

---

## Runner API

### Runner Constructor

```python
from afk.core.runner import Runner, RunnerConfig

runner = Runner(
    memory_store=my_store,              # MemoryStore | None
    interaction_provider=my_provider,   # InteractionProvider | None
    policy_engine=my_engine,            # PolicyEngine | None
    telemetry="jsonl",                  # str | TelemetrySink | None
    telemetry_config={"path": "logs"}, # Mapping[str, JSONValue] | None
    config=RunnerConfig(               # RunnerConfig | None
        interaction_mode="headless",
        debug=True,
    ),
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `memory_store` | `MemoryStore \| None` | `None` | Memory backend. When `None`, resolved from environment on first use (may fall back to in-memory) |
| `interaction_provider` | `InteractionProvider \| None` | `None` | Human-in-the-loop provider. Required when `interaction_mode` is not `"headless"` |
| `policy_engine` | `PolicyEngine \| None` | `None` | Deterministic policy engine shared across all runs on this runner |
| `telemetry` | `str \| TelemetrySink \| None` | `None` | Telemetry sink instance or backend identifier string |
| `telemetry_config` | `Mapping[str, JSONValue] \| None` | `None` | Backend-specific sink configuration |
| `config` | `RunnerConfig \| None` | `None` | Runner configuration. Defaults to `RunnerConfig()` |

### Execution Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `run(agent, *, user_message, context, thread_id)` | `AgentResult` | Async -- execute an agent run and wait for terminal result |
| `run_sync(agent, *, user_message, context, thread_id)` | `AgentResult` | Blocking -- convenience wrapper for scripts/CLIs without an event loop |
| `run_stream(agent, *, user_message, context, thread_id)` | `AgentStreamHandle` | Async -- start a run and return a stream handle for real-time events |
| `run_handle(agent, *, user_message, context, thread_id)` | `AgentRunHandle` | Async -- start execution and return a handle for full lifecycle control (pause, cancel, interrupt) |
| `resume(agent, *, run_id, thread_id, context)` | `AgentResult` | Async -- resume a previously checkpointed run and wait for completion |
| `resume_handle(agent, *, run_id, thread_id, context)` | `AgentRunHandle` | Async -- resume a run and return a live handle |
| `compact_thread(*, thread_id, event_policy, state_policy)` | `MemoryCompactionResult` | Async -- compact retained memory records for a thread |

### RunnerConfig

`RunnerConfig` is a frozen dataclass controlling runner behavior and safety
defaults. Pass it to `Runner(config=RunnerConfig(...))`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `interaction_mode` | `InteractionMode` | `"headless"` | `"headless"`, `"interactive"`, or `"external"` |
| `approval_timeout_s` | `float` | `300.0` | Timeout in seconds for deferred approval decisions |
| `input_timeout_s` | `float` | `300.0` | Timeout in seconds for deferred user-input decisions |
| `approval_fallback` | `DecisionKind` | `"deny"` | Fallback decision when approval times out |
| `input_fallback` | `DecisionKind` | `"deny"` | Fallback decision when user input times out |
| `sanitize_tool_output` | `bool` | `True` | Sanitize tool output before forwarding to the model |
| `untrusted_tool_preamble` | `bool` | `True` | Inject untrusted-data warning preamble into tool results |
| `tool_output_max_chars` | `int` | `12_000` | Max characters of tool output forwarded to model |
| `default_sandbox_profile` | `SandboxProfile \| None` | `None` | Default sandbox profile for tool execution |
| `sandbox_profile_provider` | `SandboxProfileProvider \| None` | `None` | Runtime sandbox profile resolver |
| `secret_scope_provider` | `SecretScopeProvider \| None` | `None` | Secret injector per tool call |
| `default_allowlisted_commands` | `tuple[str, ...]` | `(ls, cat, head, tail, rg, pwd, echo)` | Allowed shell commands |
| `max_parallel_subagents_global` | `int` | `64` | Global cap for concurrently executing subagent tasks |
| `max_parallel_subagents_per_parent` | `int` | `8` | Per-parent-run cap for concurrent subagent fanout |
| `max_parallel_subagents_per_target_agent` | `int` | `4` | Per-target-agent cap under broad fanout |
| `subagent_queue_backpressure_limit` | `int` | `512` | Max pending subagent nodes before backpressure |
| `checkpoint_async_writes` | `bool` | `True` | Enable asynchronous checkpoint/state writes |
| `checkpoint_queue_maxsize` | `int` | `1024` | Maximum queued checkpoint writes |
| `checkpoint_flush_timeout_s` | `float` | `10.0` | Timeout for terminal checkpoint flush |
| `checkpoint_coalesce_runtime_state` | `bool` | `True` | Coalesce runtime-state writes by key |
| `debug` | `bool` | `False` | Enable debug instrumentation for run events |
| `debug_config` | `RunnerDebugConfig \| None` | `None` | Advanced debug settings |
| `background_tools_enabled` | `bool` | `True` | Allow tools to be deferred into background execution |
| `background_tool_default_grace_s` | `float` | `0.0` | Grace window before backgrounding |
| `background_tool_max_pending` | `int` | `256` | Max unresolved background tools per run |
| `background_tool_poll_interval_s` | `float` | `0.5` | Poll interval for persisted tool state |
| `background_tool_result_ttl_s` | `float` | `3600.0` | TTL for pending background tool tickets |
| `background_tool_interrupt_on_resolve` | `bool` | `True` | Hint loop wake-up on resolution |

`RunnerConfig` also has a `from_env()` class method that loads configuration from environment variables.

---

## FailSafeConfig

`FailSafeConfig` defines runtime limits and failure-recovery policies for an
agent run. Pass it as `Agent(fail_safe=FailSafeConfig(...))`. If not provided,
the agent defaults to `FailSafeConfig(max_steps=max_steps)`.

```python
from afk.agents import Agent, FailSafeConfig

agent = Agent(
    model="gpt-4.1-mini",
    fail_safe=FailSafeConfig(
        max_steps=10,
        max_wall_time_s=60.0,
        max_total_cost_usd=0.50,
    ),
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm_failure_policy` | `FailurePolicy` | `"retry_then_fail"` | Strategy when LLM calls fail |
| `tool_failure_policy` | `FailurePolicy` | `"continue_with_error"` | Strategy when tool calls fail (one broken tool does not abort the run) |
| `subagent_failure_policy` | `FailurePolicy` | `"continue"` | Strategy when subagent calls fail (parent can still produce a result) |
| `approval_denial_policy` | `FailurePolicy` | `"skip_action"` | Strategy when approval is denied or times out |
| `max_steps` | `int` | `20` | Maximum run loop iterations |
| `max_wall_time_s` | `float` | `300.0` | Maximum wall-clock runtime in seconds |
| `max_llm_calls` | `int` | `50` | Maximum number of LLM invocations per run |
| `max_tool_calls` | `int` | `200` | Maximum number of tool invocations per run |
| `max_parallel_tools` | `int` | `16` | Max concurrent tools per batch |
| `max_subagent_depth` | `int` | `3` | Maximum subagent recursion depth |
| `max_subagent_fanout_per_step` | `int` | `4` | Maximum selected subagents per step |
| `max_total_cost_usd` | `float \| None` | `None` | Optional cost ceiling in USD for run termination |
| `fallback_model_chain` | `list[str]` | `[]` | Ordered fallback model list for LLM retries |
| `breaker_failure_threshold` | `int` | `5` | Consecutive failures before opening the circuit breaker |
| `breaker_cooldown_s` | `float` | `30.0` | Cooldown window in seconds before retrying after circuit breaker opens |

---

## AgentResult

`AgentResult` is the frozen dataclass returned by `runner.run()`,
`runner.run_sync()`, `runner.resume()`, and `agent.call()`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `run_id` | `str` | -- | Unique run identifier |
| `thread_id` | `str` | -- | Thread identifier for memory continuity |
| `state` | `AgentState` | -- | Terminal agent state (e.g. `"completed"`, `"failed"`) |
| `final_text` | `str` | -- | Final assistant text output |
| `requested_model` | `str \| None` | `None` | User-requested model identifier |
| `normalized_model` | `str \| None` | `None` | Effective model identifier used at runtime |
| `provider_adapter` | `str \| None` | `None` | Adapter/provider id used for execution |
| `final_structured` | `dict[str, JSONValue] \| None` | `None` | Final structured output payload when available |
| `llm_response` | `LLMResponse \| None` | `None` | Final raw normalized LLM response |
| `tool_executions` | `list[ToolExecutionRecord]` | `[]` | Ordered tool execution records |
| `subagent_executions` | `list[SubagentExecutionRecord]` | `[]` | Ordered subagent execution records |
| `skills_used` | `list[str]` | `[]` | Skill names enabled for this run |
| `skill_reads` | `list[SkillReadRecord]` | `[]` | Skill file read audit records |
| `skill_command_executions` | `list[CommandExecutionRecord]` | `[]` | Skill command execution records |
| `usage_aggregate` | `UsageAggregate` | `UsageAggregate()` | Total token usage across LLM calls (`input_tokens`, `output_tokens`, `total_tokens`) |
| `total_cost_usd` | `float \| None` | `None` | Aggregated cost in USD |
| `session_token` | `str \| None` | `None` | Provider session token for resume |
| `checkpoint_token` | `str \| None` | `None` | Provider checkpoint token for resume |
| `state_snapshot` | `dict[str, JSONValue]` | `{}` | Terminal runtime snapshot payload |

---

## CORRECT / WRONG Examples

### CORRECT -- Use public imports

```python
from afk.agents import Agent, FailSafeConfig, AgentResult
from afk.core.runner import Runner, RunnerConfig
```

### WRONG -- Internal module paths

```python
from afk.agents.core.base import BaseAgent       # WRONG: internal path
from afk.core.runner.api import RunnerAPIMixin    # WRONG: internal mixin
from afk.agents.types.policy import FailSafeConfig  # WRONG: internal submodule
```

---

### CORRECT -- Explicit fail-safe limits

```python
from afk.agents import Agent, FailSafeConfig

agent = Agent(
    model="gpt-4.1-mini",
    instructions="Summarize the input text.",
    fail_safe=FailSafeConfig(
        max_steps=10,
        max_wall_time_s=60.0,
        max_total_cost_usd=0.25,
    ),
)
```

### WRONG -- Rely on unset limits for production workloads

```python
from afk.agents import Agent

# No fail_safe means defaults apply (20 steps, 300s wall time, no cost cap).
# Fine for development, but production agents SHOULD set explicit budgets.
agent = Agent(model="gpt-4.1-mini", instructions="Do a complex task.")
```

---

### CORRECT -- Thread continuity with memory

```python
from afk.agents import Agent
from afk.core.runner import Runner
from afk.memory import InMemoryMemoryStore

async with InMemoryMemoryStore() as store:
    runner = Runner(memory_store=store)
    agent = Agent(model="gpt-4.1-mini", instructions="Remember context.")

    r1 = await runner.run(agent, user_message="My name is Alice.", thread_id="t1")
    r2 = await runner.run(agent, user_message="What is my name?", thread_id="t1")
    # r2 has full context from r1 because they share thread_id and memory_store
```

### WRONG -- New thread per turn (loses context)

```python
r1 = await runner.run(agent, user_message="My name is Alice.", thread_id="t1")
r2 = await runner.run(agent, user_message="What is my name?", thread_id="t2")
# Agent has no memory of r1 -- different thread_id means a fresh conversation
```

---

### CORRECT -- Streaming events

```python
from afk.agents import Agent
from afk.core.runner import Runner

agent = Agent(model="gpt-4.1-mini", instructions="Tell a short story.")
runner = Runner()

handle = await runner.run_stream(agent, user_message="Tell me a story.")
async for event in handle:
    if event.type == "text_delta":
        print(event.text_delta, end="", flush=True)
result = handle.result
```

### WRONG -- Blocking on run_sync when you need streaming

```python
# run_sync blocks until the entire response is complete.
# Use run_stream() for real-time text deltas.
result = runner.run_sync(agent, user_message="Tell me a story.")
print(result.final_text)  # Works, but no incremental output
```

---

### CORRECT -- Resuming a checkpointed run

```python
from afk.agents import Agent
from afk.core.runner import Runner

runner = Runner(memory_store=my_store)
agent = Agent(model="gpt-4.1-mini", instructions="Process data.")

# First run -- may checkpoint mid-execution
r1 = await runner.run(agent, user_message="Start processing.", thread_id="t1")

# Resume using the run_id and thread_id from the first result
r2 = await runner.resume(
    agent,
    run_id=r1.run_id,
    thread_id=r1.thread_id,
)
```

### WRONG -- Trying to resume without a memory store

```python
runner = Runner()  # No memory_store -- checkpoints go to ephemeral in-memory
result = runner.run_sync(agent, user_message="Start.", thread_id="t1")
# After process restart, in-memory state is gone -- resume will fail
```

---

### CORRECT -- Subagent delegation

```python
from afk.agents import Agent

researcher = Agent(
    model="gpt-4.1-mini",
    name="researcher",
    instructions="Find relevant information.",
)
writer = Agent(
    model="gpt-4.1-mini",
    name="writer",
    instructions="Write a polished summary.",
)

orchestrator = Agent(
    model="gpt-4.1-mini",
    name="orchestrator",
    instructions="Coordinate research and writing tasks.",
    subagents=[researcher, writer],
    max_steps=30,
)
```

### WRONG -- Nesting agents by calling Runner inside a tool

```python
# Do NOT manually run sub-agents inside tool functions.
# Use the subagents parameter on the parent Agent instead.
@tool
def research(query: str) -> str:
    sub = Agent(model="gpt-4.1-mini", instructions="Research.")
    result = Runner().run_sync(sub, user_message=query)  # WRONG: bypasses runtime
    return result.final_text
```

---

## Cross-References

- **Tools**: See [tools-system.md](./tools-system.md) for `@tool` decorator, `ToolRegistry`, and tool middleware
- **Memory**: See [memory-and-state.md](./memory-and-state.md) for `MemoryStore`, backends, retention, and vector search
- **Streaming**: See [streaming-and-interaction.md](./streaming-and-interaction.md) for `AgentStreamHandle`, stream events, and HITL
- **Policies**: See [security-and-policies.md](./security-and-policies.md) for `PolicyEngine`, `PolicyRole`, and approval flows

## Source and Docs

- Source: `src/afk/agents/core/base.py`, `src/afk/core/runner/api.py`, `src/afk/core/runner/types.py`
- Docs: https://afk.arpan.sh/library/agents, https://afk.arpan.sh/library/core-runner
- Doc files: `docs/library/agents.mdx`, `docs/library/core-runner.mdx`
