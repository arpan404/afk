# Security and Policies

AFK defense-in-depth security: `PolicyEngine` for deterministic rules,
`SandboxProfile` for tool execution boundaries, `SkillToolPolicy` for skill
command limits, and `FailSafeConfig` for runtime budgets.

- Docs: https://afk.arpan.sh/library/security-model | https://afk.arpan.sh/library/failure-policy-matrix
- Source: `src/afk/agents/policy/engine.py`, `src/afk/tools/security.py`, `src/afk/agents/types/policy.py`, `src/afk/agents/types/config.py`
- Cross-refs: `agents-and-runner.md`, `tools-system.md`

---

## 1. Overview

AFK security is layered. Each layer handles a different concern:

| Layer | Class | Purpose |
|-------|-------|---------|
| Deterministic rules | `PolicyEngine` | Block, approve, or rewrite actions before execution |
| Tool sandbox | `SandboxProfile` | Restrict filesystem, network, and command access |
| Skill commands | `SkillToolPolicy` | Limit skill shell command execution |
| Runtime budgets | `FailSafeConfig` | Cap steps, time, cost, and concurrency |
| Secret injection | `SecretScopeProvider` | Inject secrets per-tool without exposing them in context |

Key public imports:

```python
from afk.agents import (
    PolicyEngine, PolicyRule, PolicyRuleCondition, PolicyEvaluation,
    PolicyDecision, PolicyEvent, PolicyRole,
    FailSafeConfig, SkillToolPolicy,
)
from afk.tools.security import SandboxProfile, SecretScopeProvider
```

---

## 2. PolicyEngine

`PolicyEngine` evaluates deterministic rules against runtime events. Rules are
sorted by priority (descending), then `rule_id` (ascending). The highest-priority
matching rule wins. If no rules match, the default decision is `allow`.

```python
from afk.agents import PolicyEngine, PolicyRule, PolicyRuleCondition

engine = PolicyEngine(rules=[
    PolicyRule(
        rule_id="block-dangerous-tools",
        action="deny",
        priority=200,
        subjects=["tool_call"],
        condition=PolicyRuleCondition(tool_name_pattern="dangerous_*"),
        reason="Dangerous tools are blocked",
    ),
    PolicyRule(
        rule_id="require-approval-for-writes",
        action="request_approval",
        priority=100,
        subjects=["tool_call"],
        condition=PolicyRuleCondition(tool_name="write_file"),
        reason="File writes require approval",
    ),
])
```

### PolicyRule Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rule_id` | `str` | required | Unique identifier for the rule |
| `action` | `PolicyAction` | required | Action to take when matched |
| `priority` | `int` | `100` | Higher = evaluated first |
| `enabled` | `bool` | `True` | Disabled rules are skipped |
| `subjects` | `list[PolicySubject]` | `["any"]` | Event channels this rule applies to |
| `reason` | `str \| None` | `None` | Human-readable explanation |
| `request_payload` | `dict[str, JSONValue]` | `{}` | Payload for approval/input flows |
| `updated_tool_args` | `dict[str, JSONValue] \| None` | `None` | Rewritten tool args for execution |
| `condition` | `PolicyRuleCondition` | `PolicyRuleCondition()` | Match conditions |

**PolicyAction** values: `"allow"`, `"deny"`, `"defer"`, `"request_approval"`, `"request_user_input"`

**PolicySubject** values: `"llm_call"`, `"tool_call"`, `"subagent_call"`, `"interaction"`, `"any"`

### PolicyRuleCondition Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `event_type` | `str \| None` | `None` | Match specific event type |
| `tool_name` | `str \| None` | `None` | Exact tool name match |
| `tool_name_pattern` | `str \| None` | `None` | Glob pattern for tool names |
| `subagent_name` | `str \| None` | `None` | Exact subagent name match |
| `context_equals` | `dict[str, JSONValue]` | `{}` | Context key/value equality checks |
| `context_has_keys` | `list[str]` | `[]` | Required context keys |
| `metadata_equals` | `dict[str, JSONValue]` | `{}` | Metadata key/value equality checks |

### PolicyDecision

Returned by `PolicyEngine.evaluate()` inside `PolicyEvaluation.decision`:

| Field | Type | Default |
|-------|------|---------|
| `action` | `PolicyAction` | `"allow"` |
| `reason` | `str \| None` | `None` |
| `updated_tool_args` | `dict[str, JSONValue] \| None` | `None` |
| `request_payload` | `dict[str, JSONValue]` | `{}` |
| `policy_id` | `str \| None` | `None` |
| `matched_rules` | `list[str]` | `[]` |

---

## 3. PolicyRole Protocol

For dynamic policy logic beyond declarative rules, implement the `PolicyRole`
protocol. Supports both sync and async implementations.

```python
from afk.agents import PolicyRole, PolicyEvent, PolicyDecision

class MyPolicyRole:
    """Deny a specific tool and allow everything else."""

    def __call__(self, event: PolicyEvent) -> PolicyDecision:
        if event.tool_name == "risky_tool":
            return PolicyDecision(action="deny", reason="Not allowed")
        return PolicyDecision(action="allow")
```

`PolicyEvent` fields available for matching:

| Field | Type |
|-------|------|
| `event_type` | `str` |
| `run_id` | `str` |
| `thread_id` | `str` |
| `step` | `int` |
| `context` | `dict[str, JSONValue]` |
| `tool_name` | `str \| None` |
| `tool_args` | `dict[str, JSONValue] \| None` |
| `subagent_name` | `str \| None` |
| `metadata` | `dict[str, JSONValue]` |

---

## 4. FailSafeConfig (Runtime Budgets)

`FailSafeConfig` sets hard limits on agent execution and defines failure
strategies for each subsystem.

```python
from afk.agents import FailSafeConfig

failsafe = FailSafeConfig(
    max_steps=30,
    max_wall_time_s=600.0,
    max_total_cost_usd=5.0,
    max_llm_calls=100,
    llm_failure_policy="retry_then_fail",
    tool_failure_policy="continue_with_error",
    breaker_failure_threshold=5,
    breaker_cooldown_s=30.0,
)
```

### All FailSafeConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `llm_failure_policy` | `FailurePolicy` | `"retry_then_fail"` | Strategy when LLM calls fail |
| `tool_failure_policy` | `FailurePolicy` | `"continue_with_error"` | Strategy when tool calls fail |
| `subagent_failure_policy` | `FailurePolicy` | `"continue"` | Strategy when subagent calls fail |
| `approval_denial_policy` | `FailurePolicy` | `"skip_action"` | Strategy when approval is denied |
| `max_steps` | `int` | `20` | Maximum run loop iterations |
| `max_wall_time_s` | `float` | `300.0` | Maximum wall-clock seconds |
| `max_llm_calls` | `int` | `50` | Maximum LLM invocations |
| `max_tool_calls` | `int` | `200` | Maximum tool invocations |
| `max_parallel_tools` | `int` | `16` | Max concurrent tools per batch |
| `max_subagent_depth` | `int` | `3` | Maximum subagent recursion depth |
| `max_subagent_fanout_per_step` | `int` | `4` | Maximum subagents selected per step |
| `max_total_cost_usd` | `float \| None` | `None` | Cost ceiling for run termination |
| `fallback_model_chain` | `list[str]` | `[]` | Ordered fallback models for LLM retries |
| `breaker_failure_threshold` | `int` | `5` | Consecutive failures to open breaker |
| `breaker_cooldown_s` | `float` | `30.0` | Cooldown seconds after breaker opens |

**FailurePolicy** values: `"retry_then_fail"`, `"retry_then_degrade"`, `"fail_fast"`, `"continue_with_error"`, `"retry_then_continue"`, `"continue"`, `"fail_run"`, `"skip_action"`

---

## 5. SandboxProfile

`SandboxProfile` restricts what tools can do at execution time: filesystem
paths, network access, shell commands, and output size.

```python
from afk.tools.security import SandboxProfile

profile = SandboxProfile(
    profile_id="restricted",
    allow_network=False,
    allow_command_execution=True,
    allowed_command_prefixes=["ls", "cat", "head"],
    deny_shell_operators=True,
    allowed_paths=["/workspace"],
    denied_paths=["/etc", "/root"],
    max_output_chars=20_000,
)
```

### All SandboxProfile Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `profile_id` | `str` | `"default"` | Identifier for this profile |
| `allow_network` | `bool` | `False` | Allow network-accessing tools and URL args |
| `allow_command_execution` | `bool` | `True` | Allow command execution tools |
| `allowed_command_prefixes` | `list[str]` | `[]` | Allowlisted command prefixes (empty = all allowed) |
| `deny_shell_operators` | `bool` | `True` | Block `&&`, `\|\|`, `;`, `\|`, backticks, `$(`, redirects |
| `allowed_paths` | `list[str]` | `[]` | Allowed filesystem paths (empty = no restriction) |
| `denied_paths` | `list[str]` | `[]` | Denied filesystem paths (checked before allowlist) |
| `file_allowed_extensions` | `list[str]` | `[]` | Allowed file extensions (empty = no restriction) |
| `file_max_size_mb` | `int \| None` | `None` | Maximum file size in MB for file operations |
| `network_allowed_domains` | `list[str]` | `[]` | Allowed domains for network access (empty = no restriction) |
| `command_timeout_s` | `float \| None` | `None` | Command execution timeout |
| `max_output_chars` | `int` | `20_000` | Maximum output characters (truncated beyond this) |

---

## 6. SkillToolPolicy

`SkillToolPolicy` controls what the `run_skill_command` tool can execute when
skills invoke shell commands.

```python
from afk.agents import SkillToolPolicy

skill_policy = SkillToolPolicy(
    command_allowlist=["npm", "pytest", "cargo"],
    deny_shell_operators=True,
    max_stdout_chars=20_000,
    max_stderr_chars=20_000,
    command_timeout_s=30.0,
)
```

### All SkillToolPolicy Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `command_allowlist` | `list[str]` | `[]` | Allowed command prefixes (empty = all allowed) |
| `deny_shell_operators` | `bool` | `True` | Block shell chaining operators |
| `max_stdout_chars` | `int` | `20_000` | Maximum stdout characters retained |
| `max_stderr_chars` | `int` | `20_000` | Maximum stderr characters retained |
| `command_timeout_s` | `float` | `30.0` | Maximum command execution time in seconds |

---

## 7. SecretScopeProvider Protocol

`SecretScopeProvider` injects secrets per-tool invocation without exposing them
in the agent context or message history.

```python
from afk.tools.security import SecretScopeProvider

class MySecretProvider:
    """Resolve secrets based on tool name and run context."""

    def resolve(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, any],
        run_context: dict[str, any],
    ) -> dict[str, str]:
        if tool_name == "call_api":
            return {"API_KEY": "secret-value"}
        return {}
```

The provider's `resolve` method receives the tool name, arguments, and run
context, and returns a dictionary of secret key-value pairs injected into the
tool's execution environment.

---

## 8. Production Security Checklist

- Set `FailSafeConfig` with explicit `max_steps`, `max_wall_time_s`, and `max_total_cost_usd`
- Use `SandboxProfile` with `allowed_paths` and `denied_paths` to restrict filesystem access
- Set `max_total_cost_usd` for cost control on every production agent
- Use `PolicyEngine` with deny rules for dangerous or unaudited tools
- Set `deny_shell_operators=True` on both `SandboxProfile` and `SkillToolPolicy`
- Use `SecretScopeProvider` for API keys instead of passing them in context
- Set `allowed_command_prefixes` to restrict which commands tools can execute
- Set `allow_network=False` on `SandboxProfile` for agents that do not need internet

---

## 9. CORRECT / WRONG Patterns

### PolicyEngine rules

```python
# CORRECT: Explicit PolicyEngine rules for production
from afk.agents import PolicyEngine, PolicyRule, PolicyRuleCondition

engine = PolicyEngine(rules=[
    PolicyRule(
        rule_id="block-shell",
        action="deny",
        priority=200,
        subjects=["tool_call"],
        condition=PolicyRuleCondition(tool_name="run_command"),
        reason="Shell commands are blocked in production",
    ),
])

# WRONG: No policies in production -- any tool can execute anything
engine = PolicyEngine()  # empty rules = allow everything
```

### SandboxProfile restrictions

```python
# CORRECT: SandboxProfile with explicit allowlists
from afk.tools.security import SandboxProfile

profile = SandboxProfile(
    allow_network=False,
    allowed_command_prefixes=["ls", "cat"],
    allowed_paths=["/workspace"],
    denied_paths=["/etc", "/root", "/var"],
)

# WRONG: Unrestricted network and no path boundaries
profile = SandboxProfile(
    allow_network=True,
    deny_shell_operators=False,
    # no allowed_paths, no denied_paths -- everything is reachable
)
```

### FailSafeConfig budgets

```python
# CORRECT: FailSafeConfig with cost limits and bounded steps
from afk.agents import FailSafeConfig

failsafe = FailSafeConfig(
    max_steps=25,
    max_wall_time_s=300.0,
    max_total_cost_usd=2.0,
    max_llm_calls=50,
)

# WRONG: Unlimited budgets -- agent can run forever and spend without limit
failsafe = FailSafeConfig(
    max_steps=999999,
    max_wall_time_s=999999.0,
    max_total_cost_usd=None,
)
```

---

## 10. Cross-References

- **Agents and Runner**: See [agents-and-runner.md](./agents-and-runner.md) for how `PolicyEngine`, `FailSafeConfig`, and `SandboxProfile` are wired into the agent runner.
- **Tools**: See [tools-system.md](./tools-system.md) for tool registration, execution, and how sandbox validation hooks into the tool registry.
- **LLM Configuration**: See [llm-configuration.md](./llm-configuration.md) for LLM-level retry and circuit breaker policies that complement `FailSafeConfig`.

---

## 11. Source Files

| File | Purpose |
|------|---------|
| `src/afk/agents/policy/engine.py` | `PolicyEngine`, `PolicyRule`, `PolicyRuleCondition`, `PolicyEvaluation` |
| `src/afk/tools/security.py` | `SandboxProfile`, `SecretScopeProvider`, `SandboxProfileProvider`, sandbox validation |
| `src/afk/agents/types/policy.py` | `FailSafeConfig`, `PolicyDecision`, `PolicyEvent`, `AgentRunEvent` |
| `src/afk/agents/types/config.py` | `SkillToolPolicy`, `SkillRef` |
| `src/afk/agents/types/common.py` | `PolicyAction`, `FailurePolicy` literal type definitions |
| `src/afk/agents/types/protocols.py` | `PolicyRole`, `InstructionRole`, `SubagentRouter` protocol interfaces |

Documentation:
- https://afk.arpan.sh/library/security-model
- https://afk.arpan.sh/library/failure-policy-matrix
- Doc source files: `docs/library/security-model.mdx`, `docs/library/failure-policy-matrix.mdx`
