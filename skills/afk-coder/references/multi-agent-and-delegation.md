# Multi-Agent and Delegation

AFK multi-agent subsystem: hierarchical subagent trees, delegation DAGs with
fanout/fanin scheduling, A2A protocol for cross-service communication, and MCP
integration for external tool servers.

- Docs: https://afk.arpan.sh/library/a2a | https://afk.arpan.sh/library/mcp-server | https://afk.arpan.sh/library/agent-skills
- Source: `src/afk/agents/core/base.py`, `src/afk/agents/delegation.py`, `src/afk/core/runtime/dispatcher.py`, `src/afk/agents/a2a/`, `src/afk/mcp/`
- Cross-refs: `agents-and-runner.md`, `security-and-policies.md`, `cookbook-examples.md`

---

## 1. Overview

AFK supports hierarchical multi-agent systems where a parent agent can delegate
work to child subagents. Subagents are declared on the `Agent` constructor and
executed by the `Runner`. The framework provides:

- **Subagents**: Child agents attached to a parent for in-process delegation.
- **Delegation DAGs**: Directed acyclic graph plans with dependency edges,
  retry policies, and join semantics for complex multi-step workflows.
- **A2A Protocol**: Cross-service agent-to-agent communication over HTTP with
  authentication and authorization.
- **MCP Integration**: External tool servers via Model Context Protocol, whose
  tools are materialized as native AFK `Tool` instances.

Key public imports:

```python
from afk.agents import (
    Agent, BaseAgent,
    DelegationNode, DelegationEdge, DelegationPlan, RetryPolicy,
    DelegationNodeResult, DelegationResult, JoinPolicy,
    A2AServiceHost, A2AAuthProvider,
    APIKeyA2AAuthProvider, JWTA2AAuthProvider, AllowAllA2AAuthProvider,
    AgentCommunicationProtocol, AgentInvocationRequest, AgentInvocationResponse,
    FailSafeConfig, SkillToolPolicy,
)
from afk.core.runner import Runner, RunnerConfig
from afk.mcp.store import MCPServerRef, MCPStore, get_mcp_store
```

---

## 2. Subagent Basics

```python
from afk.agents import Agent

researcher = Agent(
    model="gpt-4.1-mini",
    name="researcher",
    instructions="You research topics thoroughly.",
)

writer = Agent(
    model="gpt-4.1-mini",
    name="writer",
    instructions="You write clear, concise content.",
)

orchestrator = Agent(
    model="gpt-4.1-mini",
    name="orchestrator",
    instructions="Delegate research to researcher, writing to writer.",
    subagents=[researcher, writer],
    subagent_parallelism_mode="configurable",
)
```

### Subagent Configuration

Key `Agent` constructor parameters for multi-agent orchestration:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subagents` | `list[BaseAgent] \| None` | `None` | Child agents available for delegation |
| `subagent_parallelism_mode` | `"configurable" \| "single" \| "parallel"` | `"configurable"` | How subagents execute: follow router decision, force serial, or force parallel |
| `subagent_router` | `SubagentRouter \| None` | `None` | Router callback selecting target subagents per step |
| `inherit_context_keys` | `list[str] \| None` | `None` | Context keys that flow from parent to this agent when used as a subagent |
| `fail_safe` | `FailSafeConfig \| None` | `None` | Runtime limits including subagent depth, fanout, and failure policies |

### SubagentRouter Protocol

The `SubagentRouter` protocol controls which subagents are selected and whether
they run in parallel. It receives a `RouterInput` and returns a `RouterDecision`.

```python
from afk.agents import Agent, FailSafeConfig

class MyRouter:
    def __call__(self, data):
        # data is a RouterInput with: run_id, thread_id, step, context, messages
        return RouterDecision(
            targets=["researcher"],
            parallel=False,
            metadata={"reason": "research needed first"},
        )

from afk.agents.types import RouterInput, RouterDecision

orchestrator = Agent(
    model="gpt-4.1-mini",
    name="orchestrator",
    instructions="Coordinate research and writing.",
    subagents=[researcher, writer],
    subagent_router=MyRouter(),
)
```

**RouterInput fields:**

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | Current run identifier |
| `thread_id` | `str` | Current thread identifier |
| `step` | `int` | Current loop step index |
| `context` | `dict[str, JSONValue]` | JSON-safe runtime context snapshot |
| `messages` | `list[dict[str, JSONValue]]` | Current message transcript |

**RouterDecision fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `targets` | `list[str]` | `[]` | Subagent names selected for execution |
| `parallel` | `bool` | `False` | Whether targets should execute in parallel |
| `metadata` | `dict[str, JSONValue]` | `{}` | Additional metadata for audit/debug |

---

## 3. Concurrency Controls

### RunnerConfig Limits

`RunnerConfig` (from `afk.core.runner`) controls global scheduler concurrency:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_parallel_subagents_global` | `int` | `64` | Global cap across all runs |
| `max_parallel_subagents_per_parent` | `int` | `8` | Per-parent-run concurrent subagent cap |
| `max_parallel_subagents_per_target_agent` | `int` | `4` | Per-target cap to prevent overloading one specialist |
| `subagent_queue_backpressure_limit` | `int` | `512` | Maximum pending subagent nodes before backpressure error |

### FailSafeConfig Limits

`FailSafeConfig` (from `afk.agents`) sets per-agent safety boundaries:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_subagent_depth` | `int` | `3` | Maximum subagent recursion depth |
| `max_subagent_fanout_per_step` | `int` | `4` | Maximum subagents selected per step |
| `subagent_failure_policy` | `FailurePolicy` | `"continue"` | Strategy when subagent calls fail |

`FailurePolicy` values: `"retry_then_fail"`, `"retry_then_degrade"`, `"fail_fast"`,
`"continue_with_error"`, `"retry_then_continue"`, `"continue"`, `"fail_run"`, `"skip_action"`.

```python
from afk.agents import Agent, FailSafeConfig

orchestrator = Agent(
    model="gpt-4.1-mini",
    name="orchestrator",
    instructions="Coordinate subagents safely.",
    subagents=[researcher, writer],
    fail_safe=FailSafeConfig(
        max_subagent_depth=2,
        max_subagent_fanout_per_step=3,
        subagent_failure_policy="continue_with_error",
    ),
)
```

---

## 4. Delegation DAG

For complex multi-step workflows, AFK supports delegation plans where subagents
form a directed acyclic graph with explicit dependency edges. The
`DelegationPlanner` builds plans, the `GraphValidator` validates them, and the
`DelegationScheduler` executes them with bounded parallelism.

### Core Types

```python
from afk.agents import (
    DelegationNode, DelegationEdge, DelegationPlan, RetryPolicy,
)

plan = DelegationPlan(
    nodes=[
        DelegationNode(
            node_id="research",
            target_agent="researcher",
            input_binding={"topic": "quantum computing"},
            retry_policy=RetryPolicy(max_attempts=2),
        ),
        DelegationNode(
            node_id="write",
            target_agent="writer",
            input_binding={},
        ),
    ],
    edges=[
        DelegationEdge(
            from_node="research",
            to_node="write",
            output_key_map={"findings": "source_material"},
        ),
    ],
    join_policy="all_required",
    max_parallelism=2,
)
```

### DelegationNode

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `node_id` | `str` | required | Unique node identifier within the plan |
| `target_agent` | `str` | required | Name of the subagent to execute |
| `input_binding` | `dict[str, JSONValue]` | `{}` | Static input payload for this node |
| `timeout_s` | `float \| None` | `60.0` | Per-node execution timeout |
| `retry_policy` | `RetryPolicy` | `RetryPolicy()` | Retry controls for this node |
| `required` | `bool` | `True` | Whether failure of this node blocks dependents |

### DelegationEdge

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `from_node` | `str` | required | Source node id |
| `to_node` | `str` | required | Target node id (depends on source) |
| `output_key_map` | `dict[str, str]` | `{}` | Map source output keys to target input keys |

### RetryPolicy

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_attempts` | `int` | `1` | Maximum execution attempts |
| `backoff_base_s` | `float` | `0.25` | Base backoff duration (seconds) |
| `max_backoff_s` | `float` | `5.0` | Maximum backoff cap |
| `jitter_s` | `float` | `0.0` | Random jitter added to backoff |

### Join Policies

| Policy | Description |
|--------|-------------|
| `"all_required"` | Wait for all required nodes to complete successfully |
| `"allow_optional_failures"` | Complete when all required nodes finish; optional failures are tolerated |
| `"first_success"` | Proceed when the first node completes successfully |
| `"quorum"` | Proceed when `quorum` count of nodes succeed (set `DelegationPlan.quorum`) |

### DelegationPlan

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `nodes` | `list[DelegationNode]` | required | All executable nodes |
| `edges` | `list[DelegationEdge]` | `[]` | Dependency edges forming the DAG |
| `join_policy` | `JoinPolicy` | `"all_required"` | How to aggregate completion |
| `max_parallelism` | `int` | `1` | Maximum concurrent nodes |
| `quorum` | `int \| None` | `None` | Required success count for quorum policy |

---

## 5. A2A Protocol (Agent-to-Agent)

AFK exposes agents as HTTP services using the A2A protocol. The `A2AServiceHost`
creates a FastAPI application with authenticated endpoints for invocation,
streaming, task retrieval, and cancellation.

### Service Host

```python
from afk.agents import (
    A2AServiceHost,
    APIKeyA2AAuthProvider,
    AgentCommunicationProtocol,
)

# protocol implements AgentCommunicationProtocol
host = A2AServiceHost(
    protocol=my_protocol,
    auth_provider=APIKeyA2AAuthProvider(
        key_to_subject={"secret-key-1": "service-a"},
        key_to_roles={"secret-key-1": ("a2a:all",)},
    ),
    service_name="my-agent-service",
    production_mode=True,
)

app = host.create_app()  # Returns a FastAPI app
```

### A2A Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent-card` | GET | Service discovery and capability advertisement |
| `/a2a/invoke` | POST | Synchronous agent invocation |
| `/a2a/invoke/stream` | POST | Streaming agent invocation |
| `/a2a/tasks/{task_id}` | GET | Retrieve task metadata |
| `/a2a/tasks/{task_id}/cancel` | POST | Cancel a running task |

### Auth Providers

| Provider | Class | Auth Mechanism |
|----------|-------|----------------|
| API Key | `APIKeyA2AAuthProvider` | `x-api-key` header with role-based authorization |
| JWT | `JWTA2AAuthProvider` | Bearer token with claim-driven roles |
| Allow All | `AllowAllA2AAuthProvider` | **Dev only** -- allows every request (blocked in production mode) |

```python
from afk.agents import APIKeyA2AAuthProvider, JWTA2AAuthProvider

# API Key auth with role mapping
api_key_auth = APIKeyA2AAuthProvider(
    key_to_subject={"key-abc": "frontend-service", "key-xyz": "backend-service"},
    key_to_roles={"key-abc": ("a2a:invoke",), "key-xyz": ("a2a:all",)},
    header_name="x-api-key",
)

# JWT auth with configurable claims
jwt_auth = JWTA2AAuthProvider(
    secret="my-jwt-secret",
    algorithms=("HS256",),
    audience="my-service",
    issuer="auth-server",
    role_claim="roles",
    subject_claim="sub",
)
```

### AgentCommunicationProtocol

The `AgentCommunicationProtocol` is the abstract contract for agent message
exchange. Both `InternalA2AProtocol` and `GoogleA2AProtocolAdapter` implement it.

| Method | Signature | Description |
|--------|-----------|-------------|
| `invoke` | `(AgentInvocationRequest) -> AgentInvocationResponse` | Send one request, return one response |
| `invoke_stream` | `(AgentInvocationRequest) -> AsyncIterator[AgentProtocolEvent]` | Stream protocol events until terminal |
| `get_task` | `(task_id: str) -> dict` | Fetch task metadata |
| `cancel_task` | `(task_id: str) -> dict` | Request task cancellation |

---

## 6. MCP Integration

External tool servers via Model Context Protocol. AFK resolves MCP server
references, discovers remote tools, and materializes them as native `Tool`
instances so existing runtime policies (sandbox, fail-safe, replay) apply.

```python
from afk.agents import Agent

# Option A: String URL
agent = Agent(
    model="gpt-4.1-mini",
    mcp_servers=["http://localhost:3000/mcp"],
    enable_mcp_tools=True,
)

# Option B: Name=URL shorthand
agent = Agent(
    model="gpt-4.1-mini",
    mcp_servers=["my-server=http://localhost:3000/mcp"],
    enable_mcp_tools=True,
)

# Option C: Dict config
agent = Agent(
    model="gpt-4.1-mini",
    mcp_servers=[{
        "name": "my-server",
        "url": "http://localhost:3000/mcp",
    }],
    enable_mcp_tools=True,
)

# Option D: MCPServerRef object
from afk.mcp.store import MCPServerRef
ref = MCPServerRef(
    name="my-server",
    url="http://localhost:3000/mcp",
    headers={"Authorization": "Bearer token"},
    timeout_s=20.0,
    prefix_tools=True,
)
agent = Agent(model="gpt-4.1-mini", mcp_servers=[ref])
```

### MCPServerRef

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | required | Stable local alias for the server |
| `url` | `str` | required | JSON-RPC endpoint URL |
| `headers` | `dict[str, str]` | `{}` | HTTP headers sent with each request |
| `timeout_s` | `float` | `20.0` | HTTP timeout for tool calls |
| `prefix_tools` | `bool` | `True` | Prefix tool names with server name |
| `tool_name_prefix` | `str \| None` | `None` | Explicit prefix override |

### MCPServerLike Resolution

`MCPServerLike` is the union type accepted by `Agent.mcp_servers`:

| Format | Example | Resolution |
|--------|---------|------------|
| String URL | `"http://localhost:3000/mcp"` | Name derived from host |
| Name=URL | `"my-server=http://localhost:3000/mcp"` | Explicit name and URL |
| Dict | `{"name": "x", "url": "..."}` | Mapped to `MCPServerRef` fields |
| `MCPServerRef` | `MCPServerRef(name="x", url="...")` | Used directly |

### MCP Server Hosting

Expose tools as an MCP server using `MCPServer`:

```python
from afk.tools import ToolRegistry, tool
from afk.mcp.server import MCPServer, MCPServerConfig

registry = ToolRegistry()

@tool(name="greet", description="Greet someone")
def greet(name: str) -> str:
    return f"Hello, {name}!"

registry.register(greet)

server = MCPServer(
    registry,
    config=MCPServerConfig(
        host="0.0.0.0",
        port=8080,
    ),
)

server.run()
```

### MCPServerConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | `str` | `"0.0.0.0"` | Server host |
| `port` | `int` | `8000` | Server port |
| `title` | `str` | `"AFK MCP Server"` | Server title |
| `version` | `str` | `"1.0.0"` | Server version |

### create_mcp_server

Create an MCP server from a pre-built registry:

```python
from afk.tools import ToolRegistry, tool
from afk.mcp.server import create_mcp_server

registry = ToolRegistry()

@tool(name="greet", description="Greet someone")
def greet(name: str) -> str:
    return f"Hello, {name}!"

registry.register(greet)

server = create_mcp_server(
    registry,
    config=MCPServerConfig(host="0.0.0.0", port=8080),
)
server.run()
```

---

## 7. Agent Skills

Skills inject domain-specific instructions and tools from a directory of
`SKILL.md` files. Skill tools (like `run_skill_command`) are gated by
`SkillToolPolicy`.

```python
from afk.agents import Agent, SkillToolPolicy

agent = Agent(
    model="gpt-4.1-mini",
    skills=["afk-coder"],
    skills_dir=".agents/skills",
    enable_skill_tools=True,
    skill_tool_policy=SkillToolPolicy(
        command_allowlist=["python", "pytest"],
        deny_shell_operators=True,
        command_timeout_s=30.0,
        max_stdout_chars=20_000,
        max_stderr_chars=20_000,
    ),
)
```

### SkillToolPolicy

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `command_allowlist` | `list[str]` | `[]` | Allowed command prefixes for skill command tools |
| `deny_shell_operators` | `bool` | `True` | Block shell chaining operators (`&&`, `\|`, `;`) |
| `max_stdout_chars` | `int` | `20_000` | Maximum stdout characters retained |
| `max_stderr_chars` | `int` | `20_000` | Maximum stderr characters retained |
| `command_timeout_s` | `float` | `30.0` | Maximum command execution time (seconds) |

---

## 8. CORRECT / WRONG Patterns

### Subagent depth limits

```python
# CORRECT: Set max_subagent_depth for safety
agent = Agent(
    model="gpt-4.1-mini",
    subagents=[child_a, child_b],
    fail_safe=FailSafeConfig(max_subagent_depth=2),
)

# WRONG: Unlimited subagent nesting with no fail_safe
agent = Agent(
    model="gpt-4.1-mini",
    subagents=[child_a, child_b],
    # No fail_safe -- defaults to depth 3, but be explicit for clarity
)
```

### Subagent routing

```python
# CORRECT: Use SubagentRouter for controlled delegation
class TaskRouter:
    def __call__(self, data):
        if "research" in str(data.messages):
            return RouterDecision(targets=["researcher"], parallel=False)
        return RouterDecision(targets=["writer"], parallel=False)

agent = Agent(model="gpt-4.1-mini", subagents=[...], subagent_router=TaskRouter())

# WRONG: Rely on LLM to pick subagents without any router or guardrails
agent = Agent(model="gpt-4.1-mini", subagents=[a, b, c, d, e, f])
```

### A2A authentication

```python
# CORRECT: Auth provider for A2A with role-based access
host = A2AServiceHost(
    protocol=my_protocol,
    auth_provider=APIKeyA2AAuthProvider(
        key_to_subject={"prod-key": "frontend"},
        key_to_roles={"prod-key": ("a2a:invoke",)},
    ),
    production_mode=True,
)

# WRONG: Open A2A endpoints without auth in production
host = A2AServiceHost(
    protocol=my_protocol,
    auth_provider=AllowAllA2AAuthProvider(),
    production_mode=True,  # raises A2AServiceHostError
)
```

### MCP server configuration

```python
# CORRECT: Use MCPServerRef for full control over headers and timeouts
from afk.mcp.store import MCPServerRef
agent = Agent(
    model="gpt-4.1-mini",
    mcp_servers=[MCPServerRef(name="tools", url="http://localhost:3000/mcp", timeout_s=15.0)],
)

# WRONG: Hard-code auth tokens in string URLs without header support
agent = Agent(
    model="gpt-4.1-mini",
    mcp_servers=["http://secret-token@localhost:3000/mcp"],
)
```

---

## 9. Cross-References

- **Agents and Runner**: See [agents-and-runner.md](./agents-and-runner.md) for core `Agent` and `Runner` configuration.
- **Security and Policies**: See [security-and-policies.md](./security-and-policies.md) for `PolicyEngine`, `PolicyRole`, and sandbox profiles.
- **Cookbook Examples**: See [cookbook-examples.md](./cookbook-examples.md) for end-to-end multi-agent patterns.
- **LLM Configuration**: See [llm-configuration.md](./llm-configuration.md) for model strings and builder API.

---

## 10. Source Files

| File | Purpose |
|------|---------|
| `src/afk/agents/core/base.py` | `BaseAgent` and `Agent` constructors |
| `src/afk/agents/delegation.py` | `DelegationNode`, `DelegationEdge`, `DelegationPlan`, `RetryPolicy` |
| `src/afk/core/runtime/dispatcher.py` | `DelegationPlanner`, `GraphValidator`, `DelegationScheduler` |
| `src/afk/agents/a2a/server.py` | `A2AServiceHost` FastAPI endpoint host |
| `src/afk/agents/a2a/auth.py` | Auth providers: `APIKeyA2AAuthProvider`, `JWTA2AAuthProvider` |
| `src/afk/agents/a2a/internal_protocol.py` | `InternalA2AProtocol` in-process transport |
| `src/afk/agents/contracts.py` | `AgentCommunicationProtocol`, invocation/response types |
| `src/afk/agents/types/config.py` | `RouterInput`, `RouterDecision`, `SkillToolPolicy` |
| `src/afk/agents/types/protocols.py` | `SubagentRouter`, `InstructionRole`, `PolicyRole` |
| `src/afk/agents/types/policy.py` | `FailSafeConfig` and failure policy definitions |
| `src/afk/core/runner/types.py` | `RunnerConfig` with scheduler concurrency limits |
| `src/afk/mcp/store/registry.py` | `MCPStore` server registry and tool materialization |
| `src/afk/mcp/store/types.py` | `MCPServerRef`, `MCPRemoteTool` |

Documentation:
- https://afk.arpan.sh/library/a2a
- https://afk.arpan.sh/library/mcp-server
- https://afk.arpan.sh/library/agent-skills
- Doc source files: `docs/library/a2a.mdx`, `docs/library/mcp-server.mdx`, `docs/library/agent-skills.mdx`
