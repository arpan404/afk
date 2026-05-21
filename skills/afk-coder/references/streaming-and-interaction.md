# Streaming and Interaction

> Real-time streaming, run lifecycle controls, and human-in-the-loop interaction.

- **Doc page**: https://afk.arpan.sh/library/agents
- **Doc files**: `docs/library/agents.mdx`, `docs/library/tools.mdx`
- **Source files**: `src/afk/core/streaming.py`, `src/afk/core/interaction.py`, `src/afk/agents/types/interaction.py`, `src/afk/core/runner/types.py`

---

## 1. Streaming with `run_stream()`

`Runner.run_stream()` returns an `AgentStreamHandle` that yields real-time events.

```python
from afk.agents import Agent
from afk.core import Runner

agent = Agent(model="gpt-5.2-mini", instructions="You are helpful.")
runner = Runner()

handle = await runner.run_stream(agent, user_message="Explain async/await")
async for event in handle:
    if event.type == "text_delta":
        print(event.text_delta, end="", flush=True)
    elif event.type == "tool_started":
        print(f"\n[Tool: {event.tool_name}]")
    elif event.type == "completed":
        print(f"\n\nDone. State: {event.result.state}")

# Or collect all text at once
handle2 = await runner.run_stream(agent, user_message="Hello")
text = await handle2.collect_text()
```

### AgentStreamEvent

Every event is an `AgentStreamEvent` dataclass.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `AgentStreamEventType` | Event category (see table below) |
| `text_delta` | `str \| None` | Incremental text chunk |
| `tool_name` | `str \| None` | Tool name for tool events |
| `tool_call_id` | `str \| None` | Tool call identifier |
| `tool_success` | `bool \| None` | Whether tool succeeded |
| `tool_output` | `JSONValue \| None` | Tool output payload |
| `tool_error` | `str \| None` | Tool error message |
| `tool_ticket_id` | `str \| None` | Background tool ticket ID |
| `step` | `int \| None` | Current execution step |
| `state` | `AgentState \| None` | Current agent state |
| `run_event` | `AgentRunEvent \| None` | Full run event for `run_event` type |
| `result` | `AgentResult \| None` | Terminal result for `completed` type |
| `error` | `str \| None` | Error message for `error` type |
| `data` | `dict[str, JSONValue]` | Additional JSON-safe payload |

### Event Types

| Type | When Emitted | Key Fields |
|------|-------------|------------|
| `text_delta` | Token-by-token LLM output | `text_delta`, `step` |
| `tool_started` | Tool call begins | `tool_name`, `tool_call_id`, `step` |
| `tool_completed` | Tool call finishes | `tool_name`, `tool_success`, `tool_output`, `tool_error` |
| `tool_deferred` | Tool execution deferred to background | `tool_name`, `tool_ticket_id` |
| `tool_background_resolved` | Background tool completed | `tool_name`, `tool_ticket_id` |
| `tool_background_failed` | Background tool failed | `tool_name`, `tool_ticket_id` |
| `step_started` | Runner step begins | `step`, `state` |
| `step_completed` | Runner step ends | `step`, `state` |
| `status_update` | General status change | `state`, `data` |
| `run_event` | Full run event forwarded | `run_event` |
| `completed` | Run finished | `result` |
| `error` | Run failed | `error` |

### AgentStreamHandle

| Method / Property | Description |
|-------------------|-------------|
| `async for event in handle` | Iterate over stream events |
| `await handle.collect_text()` | Consume all events, return concatenated text |
| `handle.result` | Terminal `AgentResult` (available after stream completes) |
| `handle.done` | Whether stream has ended |

**Source**: `src/afk/core/streaming.py`

---

## 2. Run Handle with `run_handle()`

`Runner.run_handle()` returns an `AgentRunHandle` for full lifecycle control: pause, resume, cancel, interrupt.

```python
handle = await runner.run_handle(agent, user_message="Write a long report")

# Consume events
async for event in handle.events:
    print(event)

# Or control the run
await handle.pause()
# ... later
await handle.resume()

# Or cancel
await handle.cancel()

# Get terminal result
result = await handle.await_result()  # AgentResult | None
```

### AgentRunHandle Protocol

| Method | Signature | Description |
|--------|-----------|-------------|
| `events` | `property -> AsyncIterator[AgentRunEvent]` | Single-consumer event stream |
| `pause()` | `async -> None` | Pause at safe boundaries |
| `resume()` | `async -> None` | Resume after pause |
| `cancel()` | `async -> None` | Cancel and resolve with `None` |
| `interrupt()` | `async -> None` | Interrupt in-flight operations |
| `await_result()` | `async -> AgentResult \| None` | Await terminal result |

**Source**: `src/afk/agents/types/interaction.py`, `src/afk/core/runner/types.py`

### CORRECT vs WRONG

```python
# CORRECT - single consumer for events
handle = await runner.run_handle(agent, user_message="Go")
async for event in handle.events:
    process(event)
result = await handle.await_result()
```

```python
# WRONG - multiple consumers on same handle
handle = await runner.run_handle(agent, user_message="Go")
asyncio.create_task(consumer_one(handle.events))  # first consumer
asyncio.create_task(consumer_two(handle.events))  # RuntimeError!
```

---

## 3. Text Delta Streaming

Enable text deltas for token-by-token output in `run_handle()`:

```python
handle = await runner.run_handle(agent, user_message="Hello")
# Text deltas are emitted as AgentRunEvent with event_type containing delta info
# For structured streaming, use run_stream() instead
```

For most streaming use cases, prefer `run_stream()` which provides typed `AgentStreamEvent` objects with the `text_delta` field.

---

## 4. InteractionProvider Protocol

The `InteractionProvider` protocol enables human-in-the-loop (HITL) workflows. Implement this protocol to connect the runner to your approval UI, chat interface, or webhook system.

```python
from afk.core.interaction import InteractionProvider
```

### Protocol Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `request_approval` | `(ApprovalRequest) -> ApprovalDecision \| DeferredDecision` | Request human approval |
| `request_user_input` | `(UserInputRequest) -> UserInputDecision \| DeferredDecision` | Request user text input |
| `await_deferred` | `(token: str, *, timeout_s: float) -> Decision \| None` | Wait for deferred decision |
| `notify` | `(event: AgentRunEvent) -> None` | Receive lifecycle notifications |

**Source**: `src/afk/core/interaction.py`

### Request Types

**ApprovalRequest**:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Run identifier |
| `thread_id` | `str` | Thread identifier |
| `step` | `int` | Current execution step |
| `reason` | `str` | Reason shown to approver |
| `payload` | `dict[str, JSONValue]` | Additional context for UI |

**UserInputRequest**:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Run identifier |
| `thread_id` | `str` | Thread identifier |
| `step` | `int` | Current execution step |
| `prompt` | `str` | Prompt text for human |
| `payload` | `dict[str, JSONValue]` | Additional context |

### Decision Types

**ApprovalDecision**:

| Field | Type | Description |
|-------|------|-------------|
| `kind` | `DecisionKind` | `"allow"`, `"deny"`, or `"defer"` |
| `reason` | `str \| None` | Explanation |

**UserInputDecision**:

| Field | Type | Description |
|-------|------|-------------|
| `kind` | `DecisionKind` | `"allow"`, `"deny"`, or `"defer"` |
| `value` | `str \| None` | User-provided text |
| `reason` | `str \| None` | Explanation |

**DeferredDecision**:

| Field | Type | Description |
|-------|------|-------------|
| `token` | `str` | Opaque token for later resolution |
| `message` | `str \| None` | Provider message for logs/UI |

---

## 5. Built-in Providers

### HeadlessInteractionProvider

Default autonomous provider. Returns immediate fallback decisions.

```python
from afk.core.interaction import HeadlessInteractionProvider

provider = HeadlessInteractionProvider(
    approval_fallback="deny",  # or "allow"
    input_fallback="deny",
)
runner = Runner(interaction_provider=provider)
```

### InMemoryInteractiveProvider

Test/development provider with in-memory deferred decisions.

```python
from afk.core.interaction import InMemoryInteractiveProvider

provider = InMemoryInteractiveProvider()

# Programmatically resolve deferred decisions
provider.set_deferred_result(
    "approval:run-1:3",
    ApprovalDecision(kind="allow"),
)

# Check captured notifications
events = provider.notifications()
```

---

## 6. Custom InteractionProvider

Implement the protocol for production HITL:

```python
from afk.core.interaction import InteractionProvider
from afk.agents.types import (
    ApprovalRequest, ApprovalDecision,
    UserInputRequest, UserInputDecision,
    DeferredDecision, AgentRunEvent,
)

class SlackInteractionProvider:
    """Route approvals to Slack, collect responses via webhook."""

    async def request_approval(self, request: ApprovalRequest):
        token = await send_slack_approval(
            channel="#approvals",
            reason=request.reason,
            run_id=request.run_id,
        )
        return DeferredDecision(token=token)

    async def request_user_input(self, request: UserInputRequest):
        token = await send_slack_prompt(
            channel="#agent-io",
            prompt=request.prompt,
        )
        return DeferredDecision(token=token)

    async def await_deferred(self, token: str, *, timeout_s: float):
        return await poll_slack_response(token, timeout_s=timeout_s)

    async def notify(self, event: AgentRunEvent):
        await post_to_slack(f"Agent event: {event.event_type}")

runner = Runner(
    interaction_provider=SlackInteractionProvider(),
    config=RunnerConfig(
        interaction_mode="external",
        approval_timeout_s=600.0,
        input_timeout_s=300.0,
    ),
)
```

---

## 7. Interaction Mode

Set via `RunnerConfig.interaction_mode`:

| Mode | Description | Default Behavior |
|------|-------------|-----------------|
| `"headless"` | No human interaction | Uses fallback decisions from config |
| `"interactive"` | In-process interaction | Calls provider synchronously |
| `"external"` | External system interaction | Supports deferred decisions with timeouts |

### Timeout and Fallback Configuration

```python
from afk.core.runner import RunnerConfig

config = RunnerConfig(
    interaction_mode="external",
    approval_timeout_s=300.0,     # 5 minutes to approve
    input_timeout_s=300.0,        # 5 minutes for user input
    approval_fallback="deny",     # Deny on timeout
    input_fallback="deny",        # Deny on timeout
)
```

---

## 8. Policy-Triggered Interactions

The `PolicyEngine` can trigger approval requests via `PolicyAction = "request_approval"`. When a policy rule matches with this action, the runner invokes the `InteractionProvider`.

```python
from afk.agents.policy import PolicyRule, PolicyRuleCondition, PolicyEngine

approval_rule = PolicyRule(
    rule_id="approve_destructive",
    action="request_approval",
    priority=200,
    condition=PolicyRuleCondition(
        tool_name_pattern="delete_*",
    ),
    reason="Destructive tool requires human approval",
)

engine = PolicyEngine(rules=[approval_rule])
runner = Runner(
    policy_engine=engine,
    interaction_provider=my_provider,
    config=RunnerConfig(interaction_mode="external"),
)
```

See [security-and-policies.md](./security-and-policies.md) for full policy engine reference.

---

## 9. Run Resumption

Resume a previously paused or checkpointed run:

```python
# Resume with checkpoint token
result = await runner.resume(
    agent,
    checkpoint_token=previous_result.checkpoint_token,
    user_message="Continue from where we left off",
)

# Resume with handle
handle = await runner.resume_handle(
    agent,
    checkpoint_token=token,
)
async for event in handle.events:
    process(event)
```

See [agents-and-runner.md](./agents-and-runner.md) for `Runner.resume()` and `Runner.resume_handle()`.

---

## Cross-References

- **Runner API**: [agents-and-runner.md](./agents-and-runner.md)
- **Tool deferred execution**: [tools-system.md](./tools-system.md)
- **Policy engine**: [security-and-policies.md](./security-and-policies.md)
- **Cookbook examples**: [cookbook-examples.md](./cookbook-examples.md)
