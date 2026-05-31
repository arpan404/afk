# Agent Forge Kit (AFK) Python SDK

AFK is a Python 3.13+ SDK for building reliable AI agents with typed tools, runtime controls, memory, streaming, evals, and observability.

Documentation: [afk.arpan.sh](https://afk.arpan.sh)

## Install

The distribution package is `afk-py`; the import package is `afk`.

```bash
python -m pip install afk-py
```

For repository development:

```bash
python -m pip install --upgrade pip
python -m pip install -e . pytest
```

## Quick Start

```python
from afk.agents import Agent
from afk.core import Runner

agent = Agent(
    name="ops-bot",
    model="gpt-4.1-mini",
    instructions="You are a helpful SRE assistant.",
)

result = Runner().run_sync(
    agent,
    user_message="What is an error budget?",
)

print(result.state)
print(result.final_text)
```

## Core Model

AFK separates agent behavior from runtime execution:

- `Agent` describes identity, model, instructions, tools, subagents, skills, MCP servers, and fail-safe limits.
- `Runner` executes agents synchronously, asynchronously, or as a stream.
- Runtime subsystems provide LLM adapters, tool execution, memory, queues, policy, and telemetry.
- `AgentResult` records final text, state, run/thread ids, tool/subagent executions, usage, and cost.

## Add a Tool

```python
from pydantic import BaseModel

from afk.agents import Agent, FailSafeConfig
from afk.core import Runner
from afk.tools import tool


class LookupArgs(BaseModel):
    order_id: str


@tool(args_model=LookupArgs, name="lookup_order", description="Look up an order.")
def lookup_order(args: LookupArgs) -> dict:
    return {"order_id": args.order_id, "status": "shipped"}


agent = Agent(
    name="support-agent",
    model="gpt-4.1-mini",
    instructions="Use lookup_order when users ask about orders.",
    tools=[lookup_order],
    fail_safe=FailSafeConfig(max_steps=8, max_tool_calls=4, max_total_cost_usd=0.10),
)

result = Runner().run_sync(agent, user_message="Where is order A123?")
print(result.final_text)
```

## Common Commands

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src pytest -q tests/agents/test_agent_runtime.py
ruff check src tests
ruff format src tests
./scripts/docs_dev.sh
./scripts/build_agentic_ai_assets.sh
```

## Install AFK Codex Skills

Install AFK skills with Vercel's Skills CLI:

```bash
npx skills add https://github.com/arpan404/afk --skill afk-coder
npx skills add https://github.com/arpan404/afk --skill afk-maintainer
```

The `skills` CLI installs the selected skill into the configured agent environment, including Codex.

## Docs Paths

- Start building: [Quickstart](https://afk.arpan.sh/library/quickstart)
- Guided tutorial: [Learn AFK in 15 Minutes](https://afk.arpan.sh/library/learn-in-15-minutes)
- Public imports: [API Reference](https://afk.arpan.sh/library/api-reference)
- Contributor workflow: [Developer Guide](https://afk.arpan.sh/library/developer-guide)
- Environment variables: [Environment Variables](https://afk.arpan.sh/library/environment-variables)

## When to Use AFK

Use AFK when your agent needs tools, multi-step execution, streaming, memory, approvals, cost limits, telemetry, evals, queues, or provider portability.

A direct provider SDK may be simpler for one-off single-turn text generation.
