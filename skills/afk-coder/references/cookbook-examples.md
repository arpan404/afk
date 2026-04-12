# Cookbook Examples

> Complete, runnable examples for common AFK patterns. Copy-paste and adapt.

- **Doc page**: https://afk.arpan.sh/library/agents
- **Source root**: `src/afk/`

All examples use **public imports only** and target Python 3.13+.

---

## 1. Minimal Agent

The simplest possible AFK agent. Five lines of code.

```python
from afk.agents import Agent, Runner

agent = Agent(
    model="gpt-4.1-mini",
    instructions="You are a helpful assistant.",
)

runner = Runner()
result = runner.run_sync(agent, user_message="What is 2 + 2?")
print(result.final_text)
```

**Key points**:
- `run_sync()` blocks until completion -- use for scripts and CLIs
- No tools, no memory, no streaming -- just LLM + instructions
- Default `RunnerConfig` is used (headless mode, deny fallbacks)

---

## 2. Agent with Custom Tools

Define and attach tools using the `@tool` decorator.

```python
from pydantic import BaseModel
from afk.agents import Agent, Runner
from afk.tools import tool, ToolContext

class WeatherArgs(BaseModel):
    city: str
    units: str = "celsius"

@tool(args_model=WeatherArgs)
async def get_weather(args: WeatherArgs, ctx: ToolContext):
    """Get current weather for a city."""
    # Replace with real API call
    return {"city": args.city, "temp": 22, "units": args.units}

class CalculateArgs(BaseModel):
    expression: str

@tool(args_model=CalculateArgs)
async def calculate(args: CalculateArgs):
    """Evaluate a math expression."""
    return {"result": eval(args.expression)}

agent = Agent(
    model="gpt-4.1-mini",
    instructions="You help with weather and math.",
    tools=[get_weather, calculate],
)

runner = Runner()
result = runner.run_sync(agent, user_message="What's the weather in Tokyo?")
print(result.final_text)
print(f"Tools used: {[t.tool_name for t in result.tool_executions]}")
```

---

## 3. Multi-Agent Delegation

Parent agent delegates to specialist subagents.

```python
from afk.agents import Agent, Runner

researcher = Agent(
    model="gpt-4.1-mini",
    name="researcher",
    instructions="You research topics thoroughly and return findings.",
)

writer = Agent(
    model="gpt-4.1-mini",
    name="writer",
    instructions="You write polished articles from research notes.",
)

editor = Agent(
    model="gpt-4.1",
    name="editor",
    instructions=(
        "You coordinate research and writing. "
        "Delegate research to 'researcher', then send findings to 'writer'."
    ),
    subagents=[researcher, writer],
)

runner = Runner()
result = runner.run_sync(editor, user_message="Write an article about quantum computing")
print(result.final_text)
print(f"Subagents used: {[s.agent_name for s in result.subagent_executions]}")
```

**Key points**:
- Subagents are passed via `subagents=[...]`
- The parent agent decides when and how to delegate
- See [multi-agent-and-delegation.md](./multi-agent-and-delegation.md) for DAG delegation and concurrency

---

## 4. Streaming Output

Real-time token-by-token output with run events.

```python
import asyncio
from afk.agents import Agent, Runner

async def main():
    agent = Agent(
        model="gpt-4.1-mini",
        instructions="You explain concepts clearly.",
    )
    runner = Runner()

    handle = await runner.run_stream(agent, user_message="Explain recursion")
    async for event in handle:
        match event.type:
            case "text_delta":
                print(event.text_delta, end="", flush=True)
            case "step_started":
                print(f"\n--- Step {event.step} ---")
            case "tool_started":
                print(f"\n[Calling {event.tool_name}...]")
            case "tool_completed":
                status = "ok" if event.tool_success else "failed"
                print(f"[{event.tool_name}: {status}]")
            case "completed":
                print(f"\n\nDone. Cost: ${event.result.total_cost_usd}")

asyncio.run(main())
```

---

## 5. Memory-Backed Agent

Persist conversation history with SQLite.

```python
from afk.agents import Agent, Runner
from afk.memory import create_memory_store

store = create_memory_store("sqlite", database_path="./agent_memory.db")
await store.setup()

agent = Agent(
    model="gpt-4.1-mini",
    instructions="You remember previous conversations.",
)

runner = Runner(memory_store=store)

# First conversation
result1 = await runner.run(
    agent,
    user_message="My name is Alice and I love Python.",
    thread_id="alice-thread",
)

# Later conversation (same thread_id loads history)
result2 = await runner.run(
    agent,
    user_message="What's my name and what do I love?",
    thread_id="alice-thread",
)
print(result2.final_text)  # "Your name is Alice and you love Python."
```

**Key points**:
- `thread_id` links conversations across runs
- Memory is loaded automatically at run start
- See [memory-and-state.md](./memory-and-state.md) for all backends

---

## 6. Production Safety Configuration

Sandbox profiles, tool output limits, and fail-safe settings.

```python
from afk.agents import Agent, Runner, FailSafeConfig
from afk.core import RunnerConfig
from afk.tools.security import SandboxProfile

sandbox = SandboxProfile(
    profile_id="production",
    allow_network=False,
    allow_command_execution=True,
    allowed_command_prefixes=["ls", "cat", "head", "tail"],
    deny_shell_operators=True,
    allowed_paths=["/app/data"],
    denied_paths=["/etc", "/var", "/root"],
    command_timeout_s=30.0,
    max_output_chars=10_000,
)

config = RunnerConfig(
    interaction_mode="headless",
    approval_fallback="deny",
    sanitize_tool_output=True,
    untrusted_tool_preamble=True,
    tool_output_max_chars=8_000,
    default_sandbox_profile=sandbox,
    debug=False,
)

agent = Agent(
    model="gpt-4.1-mini",
    instructions="You assist with data analysis.",
    tools=[...],
    max_steps=10,
    fail_safe=FailSafeConfig(
        max_steps=10,
        max_wall_time_s=60.0,
        max_total_cost_usd=0.50,
    ),
)

runner = Runner(config=config)
result = runner.run_sync(agent, user_message="Analyze sales data")
```

---

## 7. LLM Configuration

Custom LLM setup with builder pattern.

```python
from afk.agents import Agent, Runner
from afk.llms import LLMBuilder, LLMSettings

# Simple -- just set model string
agent = Agent(model="claude-sonnet-4-20250514")

# Advanced -- full LLM builder
settings = LLMSettings(
    default_provider="litellm",
    default_model="gpt-4.1",
    timeout_s=60.0,
    max_retries=5,
)

llm = (
    LLMBuilder()
    .provider("litellm")
    .model("gpt-4.1")
    .settings(settings)
    .build()
)

agent = Agent(
    model=llm,  # pass LLM instance directly
    instructions="You are a senior engineer.",
)

runner = Runner()
result = runner.run_sync(agent, user_message="Review this code")
```

See [llm-configuration.md](./llm-configuration.md) for profiles, routing, and env vars.

---

## 8. Human-in-the-Loop (HITL)

Interactive approval flow for sensitive operations.

```python
from afk.agents import Agent, Runner, PolicyRule, PolicyRuleCondition, PolicyEngine
from afk.agents.types import ApprovalDecision
from afk.core import InteractionProvider, RunnerConfig

class TerminalApprovalProvider:
    """Simple terminal-based approval provider."""

    async def request_approval(self, request):
        print(f"\n[APPROVAL NEEDED] {request.reason}")
        response = input("Allow? (y/n): ").strip().lower()
        kind = "allow" if response == "y" else "deny"
        return ApprovalDecision(kind=kind)

    async def request_user_input(self, request):
        from afk.agents.types import UserInputDecision
        print(f"\n[INPUT NEEDED] {request.prompt}")
        value = input("> ")
        return UserInputDecision(kind="allow", value=value)

    async def await_deferred(self, token, *, timeout_s):
        return None

    async def notify(self, event):
        pass

# Policy: require approval for delete operations
policy = PolicyEngine(rules=[
    PolicyRule(
        rule_id="approve_deletes",
        action="request_approval",
        priority=200,
        condition=PolicyRuleCondition(tool_name_pattern="delete_*"),
        reason="Delete operations require approval",
    ),
])

runner = Runner(
    interaction_provider=TerminalApprovalProvider(),
    policy_engine=policy,
    config=RunnerConfig(
        interaction_mode="interactive",
        approval_timeout_s=120.0,
    ),
)

agent = Agent(model="gpt-4.1-mini", tools=[...])
result = runner.run_sync(agent, user_message="Clean up old records")
```

---

## 9. Eval Suite

Automated evaluation of agent behavior.

```python
from afk.agents import Agent
from afk.evals import (
    EvalCase,
    EvalSuiteConfig,
    EvalBudget,
    StateCompletedAssertion,
    FinalTextContainsAssertion,
    arun_suite,
)

cases = [
    EvalCase(
        name="greeting",
        agent=Agent(model="gpt-4.1-mini", instructions="Greet the user."),
        user_message="Say hello",
    ),
    EvalCase(
        name="math",
        agent=Agent(model="gpt-4.1-mini", instructions="Answer math questions."),
        user_message="What is 15 * 7?",
    ),
]

result = await arun_suite(
    runner_factory=None,  # or pass Runner class
    cases=cases,
    config=EvalSuiteConfig(
        assertions=(
            StateCompletedAssertion(),
            FinalTextContainsAssertion("hello", name="contains_greeting"),
        ),
    ),
)
print(f"Passed: {result.passed}/{result.total}")
for row in result.results:
    print(f"  {row.case}: {'PASS' if row.passed else 'FAIL'}")
```

See [evals-and-testing.md](./evals-and-testing.md) for assertions, scorers, and datasets.

---

## 10. Background Tools

Long-running tools that execute in the background.

```python
import uuid
from pydantic import BaseModel
from afk.agents import Agent
from afk.core import Runner, RunnerConfig
from afk.tools import tool, ToolResult, ToolDeferredHandle

class ReportArgs(BaseModel):
    report_type: str

@tool(args_model=ReportArgs)
async def generate_report(args: ReportArgs):
    """Generate a report (takes several minutes)."""
    ticket = str(uuid.uuid4())
    # Start background job (your infrastructure)
    start_report_job(ticket, args.report_type)
    return ToolResult(
        output=f"Report generation started: {ticket}",
        deferred=ToolDeferredHandle(
            ticket_id=ticket,
            tool_name="generate_report",
            status="pending",
            poll_after_s=10.0,
            summary=f"Generating {args.report_type} report",
        ),
    )

agent = Agent(
    model="gpt-4.1-mini",
    tools=[generate_report],
    instructions="You generate reports. Use generate_report for long tasks.",
)

runner = Runner(
    config=RunnerConfig(
        background_tools_enabled=True,
        background_tool_max_pending=10,
        background_tool_result_ttl_s=7200.0,
    ),
)

result = runner.run_sync(agent, user_message="Generate a sales report")

# Later, resolve the background tool
pending = await runner.list_background_tools(result.run_id)
for ticket in pending:
    report_data = await fetch_report_result(ticket.ticket_id)
    await runner.resolve_background_tool(
        result.run_id, ticket.ticket_id, output=report_data
    )
```

---

## 11. MCP Server

Expose an agent as an MCP (Model Context Protocol) server.

```python
from afk.agents import Agent
from afk.mcp.server import MCPServer

agent = Agent(
    model="gpt-4.1-mini",
    name="assistant",
    instructions="You help with code review.",
    tools=[...],
)

server = MCPServer(agent=agent)
server.run(host="0.0.0.0", port=8080)
```

See the MCP docs at `docs/library/mcp.mdx` for full server configuration.

---

## Common Patterns Summary

| Pattern | Key Imports | Reference |
|---------|-------------|-----------|
| Minimal agent | `Agent`, `Runner` | [agents-and-runner.md](./agents-and-runner.md) |
| Custom tools | `@tool`, `ToolContext`, `ToolResult` | [tools-system.md](./tools-system.md) |
| Multi-agent | `Agent(subagents=[...])` | [multi-agent-and-delegation.md](./multi-agent-and-delegation.md) |
| Streaming | `runner.run_stream()` | [streaming-and-interaction.md](./streaming-and-interaction.md) |
| Memory | `create_memory_store()` | [memory-and-state.md](./memory-and-state.md) |
| Safety | `SandboxProfile`, `RunnerConfig` | [security-and-policies.md](./security-and-policies.md) |
| LLM config | `LLMBuilder`, `LLMSettings` | [llm-configuration.md](./llm-configuration.md) |
| HITL | `InteractionProvider`, `PolicyEngine` | [streaming-and-interaction.md](./streaming-and-interaction.md) |
| Evals | `EvalCase`, `EvalSuite` | [evals-and-testing.md](./evals-and-testing.md) |
| Background tools | `ToolDeferredHandle` | [tools-system.md](./tools-system.md) |
