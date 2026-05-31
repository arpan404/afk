---
name: afk-coder
description: Build applications with the AFK Python SDK. Use when creating or modifying AFK agents, runners, tools, memory, streaming, policy/HITL, LLM runtime configuration, queues, MCP/A2A integrations, evals, or production examples using public `afk.*` imports.
---

# AFK Coder

Use this skill to build with AFK as an application developer. Keep examples runnable, public-import only, and aligned with the current docs index bundled in this skill.

## First Steps

1. Search bundled docs before guessing API details:

   ```bash
   python skills/afk-coder/scripts/search_afk_docs.py "runner streaming tools"
   ```

2. Read only the references needed for the task.
3. Prefer the simplest working AFK pattern before adding memory, subagents, queues, or custom runtime configuration.
4. Validate examples against public imports when practical.

## Reference Map

| Need | Read |
| --- | --- |
| Agent/Runner/result basics | `references/agents-and-runner.md` |
| Tools, hooks, middleware, registries | `references/tools-system.md` |
| LLM providers, builder, settings | `references/llm-configuration.md` |
| Memory, checkpoints, retention | `references/memory-and-state.md` |
| Streaming and interaction providers | `references/streaming-and-interaction.md` |
| Policies, sandboxing, fail-safes | `references/security-and-policies.md` |
| Subagents, delegation, MCP, A2A | `references/multi-agent-and-delegation.md` |
| Evals and test patterns | `references/evals-and-testing.md` |
| Queues and workers | `references/queues.md` |
| Env vars | `references/environment-variables.md` |
| Complete examples | `references/cookbook-examples.md` and `references/afk-examples.md` |
| Current generated docs index | `references/afk-docs/docs-index.jsonl` |

## Core Pattern

```python
from afk.agents import Agent
from afk.core import Runner

agent = Agent(
    name="assistant",
    model="gpt-4.1-mini",
    instructions="Answer directly with concrete detail.",
)

result = Runner().run_sync(agent, user_message="What is an error budget?")
print(result.final_text)
```

## Tool Pattern

```python
from pydantic import BaseModel

from afk.agents import Agent, FailSafeConfig
from afk.core import Runner
from afk.tools import tool


class SearchArgs(BaseModel):
    query: str


@tool(args_model=SearchArgs, name="search_docs", description="Search docs.")
async def search_docs(args: SearchArgs) -> dict:
    return {"results": [args.query]}


agent = Agent(
    name="researcher",
    model="gpt-4.1-mini",
    instructions="Use search_docs before answering documentation questions.",
    tools=[search_docs],
    fail_safe=FailSafeConfig(max_steps=8, max_tool_calls=4, max_total_cost_usd=0.10),
)

result = Runner().run_sync(agent, user_message="Find docs about tools.")
```

## Guardrails

- Use public imports: `afk.agents`, `afk.core`, `afk.tools`, `afk.llms`, `afk.memory`, `afk.queues`, `afk.mcp`, `afk.messaging`, `afk.evals`.
- Import `Runner` from `afk.core`, not `afk.agents`.
- Do not use `src.afk` or deep internal imports in user-facing examples.
- Use Python 3.13+ assumptions.
- Use `afk-py` for package installation and `afk` for imports.
- Give every `@tool` a Pydantic `args_model`.
- Use `thread_id=` for multi-turn memory.
- Prefer `await runner.run(...)` in async services and `runner.run_sync(...)` in scripts.
- Read cost and tokens from `result.total_cost_usd` and `result.usage_aggregate`.
- Set production limits with `FailSafeConfig`.
- Add evals for behavior that must not regress.

## Build Workflow

1. Define the narrow user task and success criteria.
2. Start with one `Agent` and one `Runner`.
3. Add typed tools only when the agent must take action or fetch external data.
4. Add memory only when turns need continuity.
5. Add streaming only for user-facing progress.
6. Add policies, sandboxing, and budgets before production.
7. Add subagents only when roles are genuinely different.
8. Add queues for durable background work.
9. Add evals and telemetry before release.
