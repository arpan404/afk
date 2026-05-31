# AFK Repository Guide for Coding Agents

This file is the first stop for LLM agents working in this repository. It summarizes the repo shape, stable commands, and documentation workflow so agentic edits stay grounded in the current codebase.

## Project Snapshot

AFK is a Python 3.13 SDK for building reliable AI agents. The public package imports from `afk.*`; source lives under `src/afk/`.

Core boundaries:

- `afk.agents`: agent definitions, policy, prompts, skills, lifecycle, workflow, A2A.
- `afk.core`: `Runner`, interaction providers, streaming handles, telemetry contracts.
- `afk.llms`: provider-portable LLM runtime, adapters, retry/timeout/cache/routing policies.
- `afk.tools`: typed tools, decorators, registry, sandbox and output limiting.
- `afk.memory`: memory stores, checkpoints, retention, compaction, vector helpers.
- `afk.queues`: async task queues, execution contracts, workers, retry/DLQ behavior.
- `afk.observability`: telemetry collectors, projectors, exporters.
- `afk.mcp`, `afk.messaging`, `afk.debugger`, `afk.evals`: optional integration and quality layers.

## Ground Rules

- Use public imports in docs and examples, such as `from afk.agents import Agent` and `from afk.core import Runner`.
- Do not import from `src.afk` or internal module paths in user-facing examples unless the document is explicitly an internals reference.
- Keep Agent/Runner/Runtime boundaries intact: agents are configuration, runners execute, adapters/tools/memory provide runtime capabilities.
- Prefer typed contracts: Pydantic models, dataclasses, protocols, and explicit error classes.
- Add or update docs when changing user-visible behavior, especially runner lifecycle, tools, policy, memory, queues, LLM routing, or env vars.
- The worktree may contain user changes. Do not revert unrelated edits.

## Common Commands

Install for local development:

```bash
python -m pip install --upgrade pip
python -m pip install -e . pytest
```

Run tests:

```bash
PYTHONPATH=src pytest -q
```

Run targeted tests:

```bash
PYTHONPATH=src pytest -q tests/llms/test_llm_settings.py
PYTHONPATH=src pytest -q tests/queues/test_queue_factory.py
PYTHONPATH=src pytest -q tests/agents/test_agent_runtime.py
```

Lint and format when available:

```bash
ruff check src tests
ruff format src tests
```

Preview docs:

```bash
./scripts/docs_dev.sh
```

Regenerate agent-friendly docs and skill indexes:

```bash
./scripts/build_agentic_ai_assets.sh
```

Install AFK skills with Vercel's Skills CLI:

```bash
npx skills add https://github.com/arpan404/afk --skill afk-coder
npx skills add https://github.com/arpan404/afk --skill afk-maintainer
```

## Documentation Map

- `README.md`: repository landing page and quickest human orientation.
- `CONTRIBUTING.md`: local setup, tests, PR and docs workflow.
- `ENV_VARS.md`: environment variable reference grounded in runtime factories/settings.
- `docs/docs.json`: Mintlify navigation. Add new docs pages here or they are hard to discover.
- `docs/index.mdx`: public docs landing page.
- `docs/library/developer-guide.mdx`: framework contributor workflow.
- `docs/library/building-with-ai.mdx`: application builder playbook.
- `docs/library/api-reference.mdx`: public import contract.
- `docs/library/full-module-reference.mdx`: generated/source-level symbol map.
- `docs/library/examples/index.mdx` and `docs/library/snippets/*.mdx`: runnable examples.
- `ai-index/`: generated searchable docs records for agents.
- `skills/afk-coder/`: skill for developers building with AFK.
- `skills/afk-maintainer/`: skill for maintainers reviewing or changing AFK itself.

## Documentation Quality Checklist

Before finishing docs work:

- New pages are included in `docs/docs.json`.
- Code examples import `Runner` from `afk.core`, not `afk.agents`.
- Package install guidance distinguishes distribution install from local editable install.
- Environment variable names match `src/afk/llms/settings.py`, `src/afk/memory/factory.py`, and `src/afk/queues/factory.py`.
- Agent-facing docs mention where to search: `python skills/afk-coder/scripts/search_afk_docs.py "query"`.
- Generated assets are refreshed when navigation, snippets, or skill metadata changes.

## High-Risk Areas

Use extra care and targeted tests when touching:

- `src/afk/core/runner/`: execution loop, checkpoints, resume, policy/failure routing.
- `src/afk/core/streaming.py`: stream events and handle lifecycle.
- `src/afk/tools/core/base.py` and `src/afk/tools/registry.py`: tool invocation semantics.
- `src/afk/tools/security.py`: sandbox, secret scope, output limiting.
- `src/afk/llms/runtime/`: retries, circuit breakers, rate limits, caching, routing.
- `src/afk/memory/`: persistence, checkpoint keys, compaction, vector search.
- `src/afk/queues/`: execution contracts, retry/DLQ, worker lifecycle.
- `src/afk/agents/a2a/`: auth, delivery guarantees, protocol compatibility.
