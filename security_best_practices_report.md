# AFK Production Security and Readiness Review

Date: 2026-05-31

Scope: `src/afk`, tests, packaging metadata, runtime defaults, persistence, queues, MCP/A2A-adjacent integration boundaries, and public SDK production posture.

## Executive Summary

The framework is in good baseline shape for correctness: the full test suite passed (`1677 passed, 4 skipped`) and `ruff check src tests` passed. The main production risks are not broad code-quality failures; they are default-privilege and deployment-safety issues:

- AFK currently auto-registers local filesystem runtime tools for every runner, rooted at the process working directory.
- The MCP server defaults to listening on all interfaces, accepts wildcard CORS, and has no authentication gate before tool invocation.
- `litellm==1.81.13` currently has multiple known vulnerabilities with fixes available in newer releases.
- Remote MCP server references are only lightly validated, creating SSRF/internal-network risk if untrusted configuration can influence them.
- Audit and memory persistence can store sensitive payloads without deep redaction, retention controls, or file-permission hardening.

## Verification Performed

- `PYTHONPATH=src pytest -q`
  - Result: `1677 passed, 4 skipped in 4.73s`
- `ruff check src tests`
  - Result: all checks passed
- `bandit -r src/afk`
  - Result: highlighted the runtime/tooling items discussed below, plus several low-risk scanner findings.
- `pip-audit . --format json --progress-spinner off --desc off --aliases off`
  - Result: 6 known vulnerabilities in `litellm==1.81.13`
- `python -m build --outdir /tmp/afk-py-dist`
  - Result: wheel and sdist built successfully
- `python -m twine check /tmp/afk-py-dist/*`
  - Result: both artifacts passed metadata checks

## High Severity

### AFK-SEC-001: Local filesystem tools are registered by default for every runner

Evidence:

- `src/afk/core/runner/execution.py:314` starts per-turn tool resolution.
- `src/afk/core/runner/execution.py:347` unconditionally appends `build_runtime_tools(root_dir=Path.cwd())`.
- `src/afk/tools/prebuilts/runtime.py:47` exposes `list_directory`.
- `src/afk/tools/prebuilts/runtime.py:80` exposes `read_file`.
- `src/afk/core/runner/types.py:103` has `default_sandbox_profile=None`.
- `src/afk/core/runner/execution.py:1267` only applies sandboxing when an effective sandbox profile exists.
- `src/afk/agents/security/__init__.py:23` redacts prompt-injection markers, not secrets.

Impact:

Any agent run receives model-callable filesystem tools even if the application developer did not explicitly grant file access. The tools are root-bound to the current working directory, but that directory commonly contains `.env`, local config, checked-out source, credentials, test fixtures, and private customer data. Tool outputs can be sent back to the model provider, audit logs, telemetry, or memory stores.

Recommendation:

- Do not auto-register runtime filesystem tools by default.
- Add an explicit opt-in setting such as `RunnerConfig(enable_runtime_tools=False, runtime_tool_root=...)`.
- Prefer per-tool allowlists over broad current-working-directory access.
- Apply sandbox profiles to runtime tools by default when enabled.
- Add secret-pattern redaction for model-visible tool output, not only prompt-injection marker redaction.

Suggested production gate:

- No release should ship with implicit file tools unless the default root is empty, synthetic, or explicitly configured by the host application.

### AFK-SEC-002: MCP server defaults permit unauthenticated network tool execution

Evidence:

- `src/afk/mcp/server/runtime.py:68` defaults to `host="0.0.0.0"`.
- `src/afk/mcp/server/runtime.py:71` defaults to `cors_origins=["*"]`.
- `src/afk/mcp/server/runtime.py:199` handles `/mcp` POST requests.
- `src/afk/mcp/server/protocol.py:149` dispatches `tools/call`.
- `src/afk/mcp/server/protocol.py:163` calls the registered tool.
- `src/afk/mcp/server/runtime.py:283` installs CORS middleware with wildcard methods and headers.
- `src/afk/mcp/server/runtime.py:317` runs the service with the configured host and port.

Impact:

If a developer exposes the MCP server using the default configuration, any reachable client can invoke registered tools. If sensitive tools are present, this can become remote data access, remote mutation, or remote command execution depending on the tool set. `allow_credentials=True` combined with permissive CORS is also unsafe for browser-adjacent deployments.

Recommendation:

- Change the default host to `127.0.0.1`.
- Require an auth provider, bearer token, or explicit insecure-development flag before accepting `tools/call` over HTTP.
- Reject `cors_origins=["*"]` when credentials are enabled.
- Add startup warnings or hard failures for production-unsafe combinations.
- Add tests proving unauthenticated calls are rejected in production mode.

### AFK-SEC-003: Vulnerable `litellm` dependency pin

Evidence:

- `pyproject.toml:15` pins `litellm==1.81.13`.
- `uv.lock:795` locks `litellm` at `1.81.13`.
- `pip-audit` reported:
  - `CVE-2026-35029`, fixed in `1.83.0`
  - `CVE-2026-35030`, fixed in `1.83.0`
  - `GHSA-69x8-hrgq-fjj8`, fixed in `1.83.0`
  - `CVE-2026-42203`, fixed in `1.83.7`
  - `CVE-2026-42271`, fixed in `1.83.7`
  - `CVE-2026-40217`, fixed in `1.83.10`

Impact:

The LLM adapter dependency is directly in the SDK runtime dependency graph. Known vulnerabilities in this layer are production blockers because LiteLLM processes model requests, headers, provider configuration, and sometimes credentials.

Recommendation:

- Upgrade to `litellm>=1.83.10`.
- Regenerate the lockfile.
- Run the LLM adapter test suite and at least one integration smoke test per supported provider path.
- Add dependency-audit CI for both direct dependencies and lockfiles.

## Medium Severity

### AFK-SEC-004: Remote MCP client accepts arbitrary HTTP(S) targets with minimal SSRF protection

Evidence:

- `src/afk/mcp/store/utils.py:29` validates only that the URL scheme is `http` or `https`.
- `src/afk/mcp/store/utils.py:35` requires only `netloc`.
- `src/afk/mcp/store/registry.py:86` resolves and registers arbitrary refs.
- `src/afk/mcp/store/transport.py:68` builds the remote JSON-RPC request.
- `src/afk/mcp/store/transport.py:81` sends the request via `urllib.request.urlopen`.

Impact:

If untrusted users, tools, or configuration can influence MCP server references, AFK can be used to send requests to localhost, link-local metadata services, private subnets, or other internal services. This is a classic SSRF class even though the intended use is developer-configured MCP endpoints.

Recommendation:

- Enforce URL validation in the transport layer, not only the resolver.
- Deny loopback, link-local, multicast, private, and metadata-service IP ranges by default for remote refs.
- Add an explicit `allow_private_networks=True` escape hatch for local development.
- Require HTTPS for non-local production refs.
- Add tests for `127.0.0.1`, `[::1]`, `169.254.169.254`, RFC1918 ranges, DNS rebinding-sensitive hostnames, and direct `MCPServerRef` construction.

### AFK-SEC-005: Audit logging redaction is shallow and file permissions are not hardened

Evidence:

- `src/afk/agents/policy/audit.py:84` defaults `include_payloads=True`.
- `src/afk/agents/policy/audit.py:163` opens the audit log path for append.
- `src/afk/agents/policy/audit.py:194` performs redaction.
- `src/afk/agents/policy/audit.py:401` formats events for output.

Impact:

The redactor handles selected top-level metadata keys, but nested event payloads can still contain API keys, auth headers, tool arguments, customer prompts, or returned secrets. Log files are created with process-default permissions, which may be broader than intended on shared hosts.

Recommendation:

- Make redaction recursive for dictionaries, lists, dataclasses, and Pydantic models.
- Redact both sensitive keys and sensitive-looking string values.
- Default production audit logs to `0600` file permissions where supported.
- Consider `include_payloads=False` as the safer default for production, or require explicit opt-in.
- Add tests with nested `authorization`, `api_key`, `password`, `token`, and tool-output payloads.

### AFK-SEC-006: Memory persistence stores raw payloads without code-level privacy controls

Evidence:

- `src/afk/memory/factory.py:28` defaults `AFK_MEMORY_BACKEND` to `sqlite`.
- `src/afk/memory/factory.py:34` defaults the SQLite path to `afk_memory.sqlite3`.
- `src/afk/memory/adapters/sqlite.py:65` creates tables for raw JSON payloads.
- `src/afk/memory/adapters/sqlite.py:120` appends event payload JSON.
- `src/afk/memory/adapters/sqlite.py:165` persists state values.
- `src/afk/memory/adapters/postgres.py:89` creates raw JSONB event/state/memory storage.

Impact:

Production agents often process sensitive inputs and tool outputs. By default, memory can persist this data to a local SQLite database in the working directory, without encryption, redaction, retention limits, or environment-specific consent checks.

Recommendation:

- Make production persistence an explicit application choice.
- Add retention controls and documented defaults for local and server backends.
- Provide a redaction hook before memory writes.
- Document encryption-at-rest expectations for SQLite and Postgres deployments.
- Ensure generated local memory files are covered by `.gitignore` and docs.

### AFK-OPS-001: Queue worker cancellation can leave task outcome ambiguous

Evidence:

- `src/afk/queues/worker.py:281` starts shutdown and cancels the worker loop after timeout.
- `src/afk/queues/worker.py:295` cancels active task execution.
- `src/afk/queues/worker.py:388` executes an individual task.
- `src/afk/queues/worker.py:420` handles general exceptions, but `asyncio.CancelledError` is not handled there.

Impact:

When shutdown cancels active task execution, the queue may not consistently mark the task as failed, retryable, cancelled, or requeued. Redis-backed queues may recover through inflight recovery, but in-memory or custom queue implementations can observe ambiguous task state.

Recommendation:

- Handle `asyncio.CancelledError` in `_execute_task`.
- Decide whether shutdown cancellation should requeue, fail, or mark cancelled.
- Add tests for worker shutdown while a task is executing for every queue backend.

### AFK-OPS-002: Runtime dependency surface is broader than necessary

Evidence:

- `pyproject.toml:7` declares runtime dependencies.
- `pyproject.toml:13` includes `fastapi` as a mandatory runtime dependency.
- `pyproject.toml:14` includes `pytest` as a mandatory runtime dependency.
- `pyproject.toml:18` and `pyproject.toml:19` include Redis and Postgres dependencies as mandatory runtime dependencies.

Impact:

The installed SDK pulls in development and integration dependencies for every user, increasing supply-chain surface, install time, and vulnerability exposure. `pytest` should not normally be a runtime dependency of an SDK.

Recommendation:

- Move test dependencies to a `dev` extra.
- Move optional integrations to extras such as `mcp`, `redis`, `postgres`, `evals`, and `all`.
- Keep the base SDK dependency set minimal.
- Re-run build, twine check, import smoke tests, and a dependency audit after restructuring extras.

## Low Severity and Scanner Notes

### AFK-LOW-001: Jinja autoescape warning is not HTML XSS, but prompt templates are trusted code

Evidence:

- `src/afk/agents/prompts/store.py:172` creates a Jinja `Environment` with `autoescape=False`.
- `src/afk/agents/prompts/store.py:231` renders templates.
- `src/afk/agents/prompts/store.py:241` returns rendered prompt text.

Assessment:

Bandit reports this as high because Jinja without autoescape is dangerous for HTML output. AFK renders prompts, not HTML, so the direct XSS finding is likely a false positive. The actual risk is different: untrusted prompt templates can inspect exposed context and generate adversarial model instructions.

Recommendation:

- Document prompt templates as trusted configuration.
- Use `SandboxedEnvironment` if end users can edit templates.
- Keep autoescape disabled for non-HTML prompt output unless HTML rendering is added later.

### AFK-LOW-002: SQL f-string findings appear constrained, but should be documented

Evidence:

- `src/afk/memory/adapters/postgres.py:317`
- `src/afk/memory/adapters/postgres.py:459`
- `src/afk/memory/adapters/postgres.py:473`
- `src/afk/memory/adapters/sqlite.py:330`
- `src/afk/memory/adapters/sqlite.py:382`

Assessment:

The dynamic SQL paths reviewed use internal fragments or vector dimensions coerced through integer parsing rather than raw user strings. These do not look like exploitable SQL injection paths as written.

Recommendation:

- Bound-check vector dimensions.
- Keep all user values parameterized.
- Add precise `# nosec` comments only where the project intentionally accepts the pattern.

### AFK-LOW-003: Random jitter, asserts in testing helpers, and swallowed telemetry exceptions

Assessment:

Bandit also reported non-cryptographic `random` usage for retry jitter, `assert` in testing helpers, and swallowed exceptions in best-effort telemetry/audit paths. These are mostly acceptable as written, but swallowed exceptions should be observable in debug mode.

Recommendation:

- Keep retry jitter non-cryptographic unless used for security decisions.
- Keep asserts in test-only helpers.
- Add optional debug logging for swallowed telemetry/audit errors.

## Production Checklist

Before a production release:

- Disable implicit runtime filesystem tools, or make them explicit opt-in with a narrow root.
- Change MCP server network/auth defaults and add production guardrails.
- Upgrade `litellm` to at least `1.83.10` and regenerate the lockfile.
- Add dependency-audit CI.
- Add recursive redaction for audit logs and memory writes.
- Add retention and encryption-at-rest guidance for every persistence backend.
- Add queue shutdown tests for running tasks.
- Move test and integration dependencies out of the base runtime dependency set.

Recommended CI gates:

- `PYTHONPATH=src pytest -q`
- `ruff check src tests`
- `bandit -r src/afk`
- `pip-audit .`
- `python -m build`
- `python -m twine check dist/*`
- Import smoke tests for the minimal install and each optional extra.

Recommended security tests to add:

- A runner test proving filesystem tools are absent unless explicitly enabled.
- MCP server tests proving unauthenticated `tools/call` is rejected in production mode.
- MCP remote-ref tests rejecting private, loopback, link-local, and metadata IPs.
- Audit redaction tests for nested secrets.
- Memory redaction/retention tests.
- Queue cancellation tests for in-flight task shutdown.

