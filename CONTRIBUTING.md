# Contributing to AFK

Thanks for contributing to AFK (Agent Forge Kit).

## Development Status

> **Note:** AFK is in **fast-paced development mode**.
> Internals and public APIs are still evolving. Expect frequent changes and keep PRs focused.

## Prerequisites

- Python `3.13`
- `pip` (or `uv`, optional)
- Git

## Local Setup

```bash
python -m pip install --upgrade pip
python -m pip install -e . pytest
```

## Run Tests

Use the same command as CI:

```bash
PYTHONPATH=src pytest -q
```

Run a specific test file:

```bash
PYTHONPATH=src pytest -q tests/agents/test_agent_runtime.py
```

## Docs Workflow

- Docs live under `docs/`
- Mintlify config is `docs/docs.json`
- Main landing page is `docs/index.mdx`
- AI docs index output is generated under `ai-index/`
- Coding-agent skills are stored in `skills/`
- Repository instructions for coding agents live in `AGENTS.md`

Local docs preview:

```bash
./scripts/docs_dev.sh
```

Mintlify currently requires an LTS Node runtime and fails on Node 25+. The
script above runs the preview with Node 22 even if your global `node` points to a
newer development release.

Build AI-searchable docs index + skill metadata:

```bash
./scripts/build_agentic_ai_assets.sh
```

Search the bundled agent docs index:

```bash
python skills/afk-coder/scripts/search_afk_docs.py "runner resume"
```

Install the repository skills with Vercel's Skills CLI:

```bash
npx skills add https://github.com/arpan404/afk --skill afk-coder
npx skills add https://github.com/arpan404/afk --skill afk-maintainer
```

Use `afk-coder` when building with AFK. Use `afk-maintainer` when reviewing or changing AFK itself.

## Contribution Guidelines

- Use public imports (`afk.*`) in examples and docs.
- Keep changes scoped to one concern when possible.
- Update docs for behavior changes, especially runtime, tools, and policy semantics.
- Add or update tests for bug fixes and behavior changes.
- For prompt-loader changes, update `tests/agents/test_prompt_loader.py` and `/docs/library/system-prompts.mdx`.
- Avoid destructive git operations in shared branches.

## Pull Request Checklist

- Code builds and tests pass locally.
- New behavior is covered by tests.
- Relevant docs are updated.
- Add a changelog-ready note for `CHANGELOG.md` (`[Unreleased]`) when there is user-visible impact.
- PR description explains:
  - what changed
  - why it changed
  - any migration impact

Changelog entry format template: `.github/CHANGELOG_ENTRY_TEMPLATE.md`

## Reporting Issues

When filing an issue, include:

- environment (OS, Python version)
- minimal reproducible example
- expected behavior
- actual behavior
- logs or traceback

## Security

If you discover a security issue, please report it privately to maintainers first rather than posting full details publicly.

## Maintainer Contact

- GitHub: `arpan404@github` (handle: `@arpan404`)
- LinkedIn: `arpanbhandari`
- Email: `contact@arpan.sh`
- Docs: `https://afk.arpan.sh`
