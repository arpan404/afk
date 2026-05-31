---
name: afk-maintainer
description: Maintain and review the AFK Python SDK repository. Use when changing AFK internals, reviewing PRs, triaging issues, planning releases, updating docs/skills, or making architecture, public API, safety, dependency, testing, and compatibility decisions for AFK.
---

# AFK Maintainer

Use this skill as the repository governance guide for AFK itself. It is for framework changes, not for building downstream applications.

## First Steps

1. Identify the affected subsystem: agents, core runner, tools, LLM runtime, memory, queues, observability, evals, MCP, A2A, docs, or skills.
2. Search the generated docs when the public behavior or docs contract matters:

   ```bash
   python skills/afk-maintainer/scripts/search_afk_docs.py "public imports runner"
   ```

3. Read the minimum reference files needed for the task.
4. Preserve public API boundaries and avoid unrelated churn.
5. Validate with targeted tests first; broaden only when the change crosses subsystem boundaries.

## Reference Map

| Need | Read |
| --- | --- |
| Architecture principles and anti-patterns | `references/coding-principles-and-patterns.md` |
| PR protocol, risk, red flags | `references/maintainer-operating-rules.md` |
| DX, docs, examples, extensibility | `references/repo-design-and-quality-standards.md` |
| Review checklists | `references/code-review-checklist.md` |
| Release, triage, backport, emergency flow | `references/release-and-triage-playbook.md` |
| Dependencies and compatibility | `references/dependency-and-compatibility-rules.md` |
| Claude Agent SDK integrations | `references/claude-agent-sdk-playbook.md` |
| LiteLLM integrations | `references/litellm-playbook.md` |
| Example review/triage/release text | `references/examples.md` |
| Current generated docs index | `references/afk-docs/docs-index.jsonl` |

## Non-Negotiable Boundaries

| Boundary | Rule |
| --- | --- |
| Agent | Configuration only: identity, model, instructions, tools, subagents, policies, skills |
| Runner | Execution only: step loop, tool dispatch, memory, streaming, checkpoints, policy/HITL |
| Runtime | Capabilities only: LLM adapters, tools, memory, queues, telemetry, MCP/A2A |
| Public docs | Public `afk.*` imports only |
| Provider code | Provider-specific behavior stays in adapter/provider modules |

## Review Workflow

1. Classify risk: low, medium, high.
2. Identify user-visible behavior and public API impact.
3. Check package-boundary violations and internal imports in examples.
4. Check failure semantics: timeout, retry, cancellation, degraded state, partial completion, persistence.
5. Check docs/examples/env vars when behavior is user-visible.
6. Require tests for the changed contract.
7. Regenerate agent-facing docs when docs, snippets, navigation, or skill metadata change:

   ```bash
   ./scripts/build_agentic_ai_assets.sh
   ```

## High-Risk Areas

- `src/afk/core/runner/`: execution loop, checkpoints, resume, policy/failure routing.
- `src/afk/core/streaming.py`: stream events and handle lifecycle.
- `src/afk/tools/core/base.py` and `src/afk/tools/registry.py`: tool invocation semantics.
- `src/afk/tools/security.py`: sandbox, secret scope, output limiting.
- `src/afk/llms/runtime/`: retries, circuit breakers, rate limits, caching, routing.
- `src/afk/memory/`: persistence, checkpoint keys, compaction, vector search.
- `src/afk/queues/`: execution contracts, retry/DLQ, worker lifecycle.
- `src/afk/agents/a2a/`: auth, delivery guarantees, protocol compatibility.

## Quality Bar

- Public functions and classes have type annotations.
- Errors are actionable and classified where the runner needs policy decisions.
- Async code does not block the event loop.
- Defaults are safe and useful for the zero-config path.
- New configuration has docs and tests.
- Breaking changes include migration notes.
- Docs examples use `afk-py` for install and `afk` for imports.

## Common Validation Commands

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src pytest -q tests/agents/test_agent_runtime.py
PYTHONPATH=src pytest -q tests/llms/test_llm_settings.py
PYTHONPATH=src pytest -q tests/queues/test_queue_factory.py
ruff check src tests
./scripts/build_agentic_ai_assets.sh
```
