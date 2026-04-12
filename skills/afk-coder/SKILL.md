---
name: afk-coder
description: Build production-grade AI agents with the AFK Python library. Covers Agent declaration, Runner execution, tools, memory, streaming, policies, multi-agent delegation, LLM configuration, and evaluation.
---

# AFK Coder Skill

Use this skill when building AI agents with the **AFK** Python library (`afk-py`).

## When to Use This Skill

- Creating new agents with `Agent()` and running them with `Runner`
- Defining custom tools with `@tool` and composing hook/middleware pipelines
- Configuring LLM providers, models, and runtime settings
- Adding conversation memory (SQLite, Postgres, Redis, in-memory)
- Implementing streaming output and human-in-the-loop workflows
- Setting up multi-agent delegation and subagent orchestration
- Applying security policies, sandbox profiles, and fail-safe guards
- Writing eval suites to test agent behavior
- Deploying agents as MCP or A2A servers

## Reference Files

Read these files for detailed API references. Recommended order:

| # | File | Purpose |
|---|------|---------|
| 1 | [agents-and-runner.md](./references/agents-and-runner.md) | Agent declaration, Runner API, RunnerConfig, AgentResult |
| 2 | [tools-system.md](./references/tools-system.md) | @tool decorator, ToolResult, hooks, middleware, ToolRegistry |
| 3 | [llm-configuration.md](./references/llm-configuration.md) | Model strings, LLMBuilder, LLMSettings, provider system |
| 4 | [memory-and-state.md](./references/memory-and-state.md) | MemoryStore backends, factory, vector search, retention |
| 5 | [streaming-and-interaction.md](./references/streaming-and-interaction.md) | AgentStreamHandle, AgentRunHandle, InteractionProvider |
| 6 | [security-and-policies.md](./references/security-and-policies.md) | PolicyEngine, SandboxProfile, SkillToolPolicy, fail-safe |
| 7 | [multi-agent-and-delegation.md](./references/multi-agent-and-delegation.md) | Subagents, DelegationPlan, A2A protocol, MCP |
| 8 | [evals-and-testing.md](./references/evals-and-testing.md) | EvalCase, EvalSuite, assertions, budgets, testing patterns |
| 9 | [debugger.md](./references/debugger.md) | Debugger, DebuggerConfig, debug instrumentation |
| 10 | [queues.md](./references/queues.md) | TaskQueue, TaskWorker, retry policies |
| 11 | [environment-variables.md](./references/environment-variables.md) | All AFK_* env vars, loading from .env |
| 12 | [cookbook-examples.md](./references/cookbook-examples.md) | 11 complete runnable examples |

## Architecture

AFK uses a **three-pillar** architecture:

```
Agent (stateless config)  -->  Runner (stateful execution)  -->  Runtime (LLM, tools, memory, telemetry)
```

**Three tiers**:
- **Orchestration** (`afk.core`): Runner, streaming, interaction
- **Adapters** (`afk.llms`, `afk.tools`, `afk.memory`): Provider-portable integrations
- **Extensions** (`afk.evals`, `afk.observability`, `afk.queues`, `afk.mcp`, `afk.messaging`): Optional capabilities

## Core Philosophy

1. **Progressive disclosure** -- Simple things are simple (5-line agent), advanced things are possible
2. **Composition over inheritance** -- Middleware, hooks, policies, registries
3. **Contract-first** -- Pydantic models, Protocols, ABCs at every boundary
4. **Provider-portable** -- Zero lock-in via normalized LLM types
5. **Safety by default** -- Deny fallbacks, output sanitization, sandbox profiles
6. **Observable** -- Built-in telemetry, structured events, cost tracking
7. **Async-native** -- `async/await` throughout with `run_sync()` for scripts

## Essential Patterns

### Minimal Agent

```python
from afk.agents import Agent, Runner

agent = Agent(model="gpt-4.1-mini", instructions="You are helpful.")
result = Runner().run_sync(agent, user_message="Hello")
print(result.final_text)
```

### Agent with Tools

```python
from pydantic import BaseModel
from afk.agents import Agent, Runner
from afk.tools import tool

class SearchArgs(BaseModel):
    query: str

@tool(args_model=SearchArgs)
async def search(args: SearchArgs):
    """Search the web."""
    return {"results": [f"Result for {args.query}"]}

agent = Agent(model="gpt-4.1-mini", tools=[search])
result = Runner().run_sync(agent, user_message="Search for AFK docs")
```

### Streaming

```python
handle = await runner.run_stream(agent, user_message="Explain AFK")
async for event in handle:
    if event.type == "text_delta":
        print(event.text_delta, end="", flush=True)
```

### Multi-Agent

```python
researcher = Agent(model="gpt-4.1-mini", name="researcher", instructions="Research topics.")
writer = Agent(model="gpt-4.1-mini", name="writer", instructions="Write articles.")
lead = Agent(model="gpt-4.1", subagents=[researcher, writer], instructions="Coordinate.")
result = Runner().run_sync(lead, user_message="Write about AI agents")
```

### Memory

```python
from afk.memory import create_memory_store

store = create_memory_store("sqlite", database_path="./memory.db")
await store.setup()
runner = Runner(memory_store=store)
result = await runner.run(agent, user_message="Hi", thread_id="thread-1")
```

## Mandatory Guardrails

When generating AFK agent code, always ensure:

1. **Use public imports only** -- `from afk.agents import Agent`, never internal paths
2. **Pydantic v2 args models** -- Every `@tool` requires `args_model=SomeBaseModel`
3. **Async by default** -- Use `async def` for tools; sync tools run in threadpool
4. **Handle tool errors** -- Default `raise_on_error=False` returns `ToolResult(success=False)`; enable strict mode for critical tools
5. **Set `max_steps`** -- Always configure `Agent(max_steps=N)` to prevent runaway loops
6. **Thread IDs for memory** -- Pass `thread_id=` to `runner.run()` when using memory
7. **Sandbox in production** -- Configure `SandboxProfile` and `RunnerConfig` for deployed agents
8. **Test with evals** -- Write `EvalCase` assertions for expected behavior

## Decision Workflow

When building an AFK agent, follow this sequence:

1. **Define the agent's purpose** -- What instructions and capabilities does it need?
2. **Choose tools** -- Define `@tool` functions or use prebuilts
3. **Pick a model** -- Model string or `LLMBuilder` for advanced config
4. **Add memory** (if needed) -- Select backend, call `create_memory_store()`
5. **Configure safety** -- `FailSafeConfig`, `SandboxProfile`, `PolicyEngine`
6. **Set up interaction** (if HITL needed) -- Implement `InteractionProvider`
7. **Add subagents** (if multi-agent) -- Define specialists, attach to parent
8. **Configure runner** -- `RunnerConfig` with timeouts, limits, debug settings
9. **Write evals** -- `EvalCase` + `EvalSuite` for automated testing
10. **Deploy** -- MCP server, A2A server, or direct integration

## Documentation

- **Web docs**: https://afk.arpan.sh
- **Library docs**: https://afk.arpan.sh/library/agents
- **Doc files** (for direct reading):
  - `docs/library/agents.mdx` -- Agent and Runner
  - `docs/library/tools.mdx` -- Tools system
  - `docs/library/memory.mdx` -- Memory stores
  - `docs/library/a2a.mdx` -- Agent-to-Agent protocol
  - `docs/library/mcp.mdx` -- MCP integration
  - `docs/library/full-module-reference.mdx` -- Complete module reference

## Source Paths

Key source files for implementation details:

| Module | Path |
|--------|------|
| Agent core | `src/afk/agents/core/base.py` |
| Agent types | `src/afk/agents/types/` |
| Runner API | `src/afk/core/runner/api.py` |
| Runner config | `src/afk/core/runner/types.py` |
| Runner execution | `src/afk/core/runner/execution.py` |
| Tool base | `src/afk/tools/core/base.py` |
| Tool decorator | `src/afk/tools/core/decorator.py` |
| Tool registry | `src/afk/tools/registry.py` |
| Tool errors | `src/afk/tools/core/errors.py` |
| Tool security | `src/afk/tools/security.py` |
| LLM builder | `src/afk/llms/builder.py` |
| LLM settings | `src/afk/llms/settings.py` |
| LLM client | `src/afk/llms/runtime/client.py` |
| Memory store | `src/afk/memory/store.py` |
| Memory factory | `src/afk/memory/factory.py` |
| Memory backends | `src/afk/memory/adapters/` |
| Memory lifecycle | `src/afk/memory/lifecycle.py` |
| Streaming | `src/afk/core/streaming.py` |
| Interaction | `src/afk/core/interaction.py` |
| Policy engine | `src/afk/agents/policy/engine.py` |
| Delegation | `src/afk/core/runtime/dispatcher.py` |
| Evals | `src/afk/evals/` |
| Debugger | `src/afk/debugger/` |
| Queues | `src/afk/queues/` |
| MCP server | `src/afk/mcp/server/` |
| A2A protocol | `src/afk/agents/a2a/` |

## Utilities

- **Search docs**: `python agent-skill/coder/scripts/search_afk_docs.py "query"`
- **LLM reference**: `agent-skill/coder/llms.txt` (self-contained API reference)
- **Config**: `agent-skill/coder/assets/coder-config.yaml`
