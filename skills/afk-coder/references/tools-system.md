# Tools System

> How to define, compose, and execute tools in AFK agents.

- **Doc page**: https://afk.arpan.sh/library/tools
- **Doc file**: `docs/library/tools.mdx`
- **Source files**: `src/afk/tools/core/base.py`, `src/afk/tools/core/decorator.py`, `src/afk/tools/core/errors.py`, `src/afk/tools/registry.py`, `src/afk/tools/security.py`

---

## 1. The `@tool` Decorator

The primary way to create tools. Requires a **Pydantic v2 args model**.

```python
from pydantic import BaseModel
from afk.tools import tool, ToolContext

class SearchArgs(BaseModel):
    query: str
    max_results: int = 10

@tool(args_model=SearchArgs)
async def web_search(args: SearchArgs, ctx: ToolContext):
    """Search the web for relevant results."""
    return {"results": [f"Result for '{args.query}'"]}
```

### Decorator Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `args_model` | `type[BaseModel]` | **required** | Pydantic v2 model for argument validation |
| `name` | `str \| None` | function name | Override tool name |
| `description` | `str \| None` | first docstring line | Override tool description |
| `timeout` | `float \| None` | `None` | Per-tool timeout in seconds |
| `prehooks` | `list[PreHook]` | `None` | Argument transformers run before execution |
| `posthooks` | `list[PostHook]` | `None` | Output transformers run after execution |
| `middlewares` | `list[Middleware]` | `None` | Wrap the execution pipeline |
| `raise_on_error` | `bool` | `False` | Raise exceptions instead of returning error results |

### Function Signatures

Tool functions accept these signatures (sync or async):

```python
# Args only
@tool(args_model=MyArgs)
def my_tool(args: MyArgs): ...

# Args + context
@tool(args_model=MyArgs)
async def my_tool(args: MyArgs, ctx: ToolContext): ...

# Context + args
@tool(args_model=MyArgs)
async def my_tool(ctx: ToolContext, args: MyArgs): ...
```

### CORRECT vs WRONG

```python
# CORRECT - Pydantic v2 model with explicit types
class CalculateArgs(BaseModel):
    expression: str

@tool(args_model=CalculateArgs)
async def calculate(args: CalculateArgs):
    left, op, right = args.expression.split()
    operations = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b,
    }
    return operations[op](float(left), float(right))
```

```python
# WRONG - raw dict, no model
@tool()
async def calculate(expression: str):
    left, op, right = expression.split()
    return {"+": lambda a, b: a + b}[op](float(left), float(right))
```

---

## 2. ToolResult

All tool executions return `ToolResult[ReturnT]`. The runner reads this to build model messages.

```python
from afk.tools import ToolResult
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output` | `ReturnT \| None` | `None` | Tool return value |
| `success` | `bool` | `True` | Whether execution succeeded |
| `error_message` | `str \| None` | `None` | Error description on failure |
| `tool_name` | `str \| None` | `None` | Name of the tool that produced this result |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata for observability |
| `tool_call_id` | `str \| None` | `None` | Model-assigned tool call identifier |
| `deferred` | `ToolDeferredHandle \| None` | `None` | Background tool marker (see section 7) |

### Returning ToolResult Directly

Tools can return `ToolResult` explicitly for richer control:

```python
@tool(args_model=MyArgs)
async def my_tool(args: MyArgs, ctx: ToolContext):
    if not args.query:
        return ToolResult(output=None, success=False, error_message="Empty query")
    return ToolResult(output="done", metadata={"cached": True})
```

If a tool returns a plain value (string, dict, etc.), it is automatically wrapped in `ToolResult(output=value, success=True)`.

---

## 3. ToolContext

Contextual information available during execution. Keep it serializable.

```python
from afk.tools import ToolContext
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `request_id` | `str \| None` | `None` | Current request/run identifier |
| `user_id` | `str \| None` | `None` | User identifier for access control |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary runtime metadata |

The runner populates `ToolContext` automatically from the run state. Tools can read it for user-specific logic.

---

## 4. ToolSpec

Stable metadata used for registry listing and LLM tool-calling export.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique tool name |
| `description` | `str` | Human/model-readable description |
| `parameters_schema` | `dict[str, Any]` | JSON Schema derived from args model |

Access via `tool_instance.spec`.

---

## 5. Hooks and Middleware

AFK tools support a three-layer execution pipeline: **prehooks** -> **middleware** -> **core** -> **posthooks**.

### PreHooks

Transform arguments before the tool executes. Must return a `dict` compatible with the main tool's args model.

```python
from afk.tools import prehook

class SanitizeArgs(BaseModel):
    query: str

@prehook(args_model=SanitizeArgs)
def sanitize_query(args: SanitizeArgs):
    """Strip dangerous characters."""
    return {"query": args.query.replace(";", "")}

@tool(args_model=SearchArgs, prehooks=[sanitize_query])
async def safe_search(args: SearchArgs):
    return {"results": [args.query]}
```

### PostHooks

Transform tool output after execution. Receive `{"output": <value>, "tool_name": <name>}`.

```python
from afk.tools import posthook

class PostArgs(BaseModel):
    output: Any
    tool_name: str | None = None

@posthook(args_model=PostArgs)
def redact_output(args: PostArgs):
    """Redact sensitive patterns from output."""
    text = str(args.output)
    return text.replace("SECRET", "***")

@tool(args_model=SearchArgs, posthooks=[redact_output])
async def search_with_redaction(args: SearchArgs):
    return "Result contains SECRET data"
```

### Tool-Level Middleware

Wraps the core execution with cross-cutting logic (logging, retries, etc.).

```python
from afk.tools import middleware

@middleware(name="timing")
async def timing_middleware(call_next, args, ctx: ToolContext):
    import time
    start = time.time()
    result = await call_next(args, ctx)
    elapsed = time.time() - start
    print(f"Tool took {elapsed:.2f}s")
    return result

@tool(args_model=SearchArgs, middlewares=[timing_middleware])
async def timed_search(args: SearchArgs):
    return {"results": []}
```

Middleware signatures (sync or async):
- `(call_next, args)`
- `(call_next, args, ctx)`
- `(args, ctx, call_next)`
- `(ctx, args, call_next)`

Where `call_next` is `async (args, ctx) -> output`.

---

## 6. ToolRegistry

Central registry for managing and executing tools. Used internally by the runner.

```python
from afk.tools import ToolRegistry
```

### Constructor

```python
ToolRegistry(
    max_concurrency=32,       # Semaphore limit for parallel tool calls
    default_timeout=None,     # Fallback timeout for all tools
    policy=None,              # ToolPolicy callback for allow/deny
    enable_plugins=False,     # Auto-discover via entry points
    plugin_entry_point_group="afk.tools",
    middlewares=None,          # Registry-level middlewares
)
```

### Core Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `register` | `(tool, *, overwrite=False)` | Register a single tool |
| `register_many` | `(tools, *, overwrite=False)` | Register multiple tools |
| `unregister` | `(name)` | Remove a tool by name |
| `get` | `(name) -> Tool` | Get tool by name (raises `ToolNotFoundError`) |
| `list` | `() -> list[Tool]` | List all registered tools |
| `names` | `() -> list[str]` | List all tool names |
| `has` | `(name) -> bool` | Check if a tool is registered |
| `call` | `(name, raw_args, *, ctx, timeout, tool_call_id)` | Execute a tool by name |
| `call_many` | `(calls, *, ctx, timeout, ...)` | Execute multiple tools concurrently |
| `specs` | `() -> list[ToolSpec]` | Get all tool specs |
| `to_openai_function_tools` | `() -> list[dict]` | Export in OpenAI format |

### Registry-Level Middleware

Wraps **all** tool calls through the registry, regardless of individual tool configuration.

```python
from afk.tools import registry_middleware

@registry_middleware(name="audit_log")
async def audit_log(call_next, tool, raw_args, ctx):
    print(f"Calling tool: {tool.spec.name}")
    result = await call_next(tool, raw_args, ctx, None, None)
    print(f"Result: {result.success}")
    return result

registry = ToolRegistry(middlewares=[audit_log])
```

Registry middleware signatures (sync or async):
- `(call_next, tool, raw_args, ctx)` -- 4 params
- `(call_next, tool, raw_args, ctx, timeout, tool_call_id)` -- 6 params
- `(tool, raw_args, ctx, call_next)` -- 4 params
- `(tool, raw_args, ctx, call_next, timeout, tool_call_id)` -- 6 params

Where `call_next` is `async (tool, raw_args, ctx, timeout, tool_call_id) -> ToolResult`.

### Timeout Precedence

1. `registry.call(timeout=...)` -- explicit call-site timeout
2. `tool.default_timeout` -- per-tool timeout from decorator
3. `registry._default_timeout` -- registry-wide fallback

### Plugin Discovery

Register tools from Python entry points in `pyproject.toml`:

```toml
[project.entry-points."afk.tools"]
my_tool = "my_pkg.tools:my_tool"
```

Enable with `ToolRegistry(enable_plugins=True)`.

---

## 7. Background Tools (Deferred Execution)

Tools can signal deferred/background execution by returning a `ToolDeferredHandle`.

```python
from afk.tools import ToolResult, ToolDeferredHandle

@tool(args_model=LongTaskArgs)
async def long_running_task(args: LongTaskArgs):
    ticket_id = start_background_work(args.task_id)
    return ToolResult(
        output="Task started",
        deferred=ToolDeferredHandle(
            ticket_id=ticket_id,
            tool_name="long_running_task",
            status="pending",
            poll_after_s=5.0,
            summary="Processing in background",
        ),
    )
```

### ToolDeferredHandle Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ticket_id` | `str` | **required** | Unique identifier for the background task |
| `tool_name` | `str` | **required** | Name of the deferred tool |
| `status` | `str` | `"pending"` | One of: `pending`, `running`, `completed`, `failed` |
| `poll_after_s` | `float \| None` | `None` | Suggested polling interval |
| `summary` | `str \| None` | `None` | Human-readable status summary |
| `resume_hint` | `str \| None` | `None` | Hint for the runner on how to resume |

### Runner Background Tool Config

Controlled via `RunnerConfig`:

| Field | Default | Description |
|-------|---------|-------------|
| `background_tools_enabled` | `True` | Allow deferred tool execution |
| `background_tool_default_grace_s` | `0.0` | Grace window before backgrounding |
| `background_tool_max_pending` | `256` | Max unresolved background tools per run |
| `background_tool_poll_interval_s` | `0.5` | Poll interval for persisted state |
| `background_tool_result_ttl_s` | `3600.0` | TTL for pending background tickets |
| `background_tool_interrupt_on_resolve` | `True` | Wake loop on resolution |

See [agents-and-runner.md](./agents-and-runner.md) for `Runner.list_background_tools()`, `Runner.resolve_background_tool()`, `Runner.fail_background_tool()`.

---

## 8. Prebuilt Tools

AFK ships with runtime and skill tools available via public imports.

```python
from afk.tools.prebuilts import runtime_tools, skill_tools
```

- **Runtime tools**: Shell command execution, file operations (controlled by `RunnerConfig.default_allowlisted_commands`)
- **Skill tools**: Read skill files, execute skill commands (controlled by `SkillToolPolicy`)

See [security-and-policies.md](./security-and-policies.md) for `SkillToolPolicy` and `SandboxProfile`.

---

## 9. Error Hierarchy

All tool errors inherit from `AFKToolError`:

```
AFKToolError
├── ToolValidationError     # Invalid arguments / bad signature
├── ToolExecutionError      # Runtime failure during execution
├── ToolTimeoutError        # Timeout exceeded
├── ToolPolicyError         # Blocked by policy hook / sandbox
├── ToolNotFoundError       # Tool not in registry
├── ToolAlreadyRegisteredError  # Duplicate registration
└── ToolPermissionError     # Access denied
```

**Source**: `src/afk/tools/core/errors.py`

### Error Behavior

- **Default** (`raise_on_error=False`): Errors are caught and returned as `ToolResult(success=False, error_message=...)`
- **Strict** (`raise_on_error=True`): Errors propagate as exceptions

```python
# CORRECT - let runner handle errors gracefully
@tool(args_model=MyArgs)
async def safe_tool(args: MyArgs):
    return do_work(args)  # errors become ToolResult(success=False)

# CORRECT - strict mode for critical tools
@tool(args_model=MyArgs, raise_on_error=True)
async def critical_tool(args: MyArgs):
    return do_critical_work(args)  # errors propagate as exceptions
```

---

## 10. Passing Tools to Agents

Tools are passed to agents via the `tools` parameter. The `ToolLike` type alias accepts `Tool` instances and `BaseTool` subclasses.

```python
from afk.agents import Agent

agent = Agent(
    model="gpt-4.1-mini",
    tools=[web_search, calculate, safe_search],
    instructions="You are a helpful research assistant.",
)
```

The agent internally builds a `ToolRegistry` from the provided tools. You can also access `agent.build_tool_registry()` for custom registry configuration.

### CORRECT vs WRONG

```python
# CORRECT - pass decorated tool objects
agent = Agent(model="gpt-4.1-mini", tools=[web_search, calculate])

# WRONG - pass raw functions (not decorated)
agent = Agent(model="gpt-4.1-mini", tools=[my_raw_function])
```

---

## Cross-References

- **Agent + Runner integration**: [agents-and-runner.md](./agents-and-runner.md)
- **Sandbox and policy enforcement**: [security-and-policies.md](./security-and-policies.md)
- **Background tool lifecycle**: [streaming-and-interaction.md](./streaming-and-interaction.md)
- **Cookbook examples**: [cookbook-examples.md](./cookbook-examples.md)
