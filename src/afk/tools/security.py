"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Sandbox and secret-scoping primitives for tool execution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .core.base import ToolContext, ToolResult
from .core.errors import ToolPolicyError


@dataclass(frozen=True, slots=True)
class SandboxProfile:
    profile_id: str = "default"
    allow_network: bool = False
    allow_command_execution: bool = False
    allowed_command_prefixes: list[str] = field(default_factory=list)
    deny_shell_operators: bool = True
    allowed_paths: list[str] = field(default_factory=list)
    denied_paths: list[str] = field(default_factory=list)
    command_timeout_s: float | None = None
    max_output_chars: int = 20_000


class SecretScopeProvider(Protocol):
    def resolve(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        run_context: dict[str, Any],
    ) -> dict[str, str]: ...


class SandboxProfileProvider(Protocol):
    def resolve(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        run_context: dict[str, Any],
    ) -> SandboxProfile | None: ...


def validate_tool_args_against_sandbox(
    *,
    tool_name: str,
    tool_args: dict[str, Any],
    profile: SandboxProfile,
    cwd: Path,
) -> str | None:
    lowered_name = tool_name.lower()

    if not profile.allow_network:
        if lowered_name in {
            "webfetch",
            "websearch",
            "web_fetch",
            "web_search",
        }:
            return f"Network access denied by sandbox profile '{profile.profile_id}'"

        for key, value in _iter_leaf_values(tool_args):
            if (
                key in {"url", "uri"}
                and isinstance(value, str)
                and value.strip().startswith(("http://", "https://"))
            ):
                return f"Network URL argument denied by sandbox profile '{profile.profile_id}'"

    command_parts = _extract_command_parts(tool_args)
    if command_parts:
        if not profile.allow_command_execution:
            return f"Command execution denied by sandbox profile '{profile.profile_id}'"

        if profile.deny_shell_operators:
            blocked = {"&&", "||", ";", "|", "`", "$(", ">", ">>", "<", "<<", "&"}
            joined = " ".join(command_parts)
            for op in blocked:
                if op in joined:
                    return (
                        f"Command denied due to shell operator '{op}' by sandbox "
                        f"profile '{profile.profile_id}'"
                    )

        command = command_parts[0].strip()
        if not profile.allowed_command_prefixes:
            return (
                f"Command '{command}' not allowlisted by sandbox profile "
                f"'{profile.profile_id}'"
            )
        if not _is_command_allowed(command, profile.allowed_command_prefixes):
            return (
                f"Command '{command}' not allowlisted by sandbox profile "
                f"'{profile.profile_id}'"
            )

    denied_roots = [_resolve_root(path, cwd) for path in profile.denied_paths]
    allowed_roots = [_resolve_root(path, cwd) for path in profile.allowed_paths]

    for key, value in _iter_leaf_values(tool_args):
        if not isinstance(value, str):
            continue
        if not _looks_like_path_key(key):
            continue
        if value.strip().startswith(("http://", "https://")):
            continue

        candidate = _resolve_path(value, cwd)
        if any(_is_under(candidate, root) for root in denied_roots):
            return (
                f"Path '{candidate}' denied by sandbox profile '{profile.profile_id}'"
            )
        if allowed_roots and not any(
            _is_under(candidate, root) for root in allowed_roots
        ):
            return f"Path '{candidate}' not in allowlist for sandbox profile '{profile.profile_id}'"

    return None


def build_registry_sandbox_policy(
    *,
    profile: SandboxProfile,
    cwd: Path,
) -> Callable[[str, dict[str, Any], ToolContext], None]:
    """
    Build a ToolRegistry policy hook enforcing the sandbox profile at call time.
    """

    def _policy(tool_name: str, raw_args: dict[str, Any], ctx: ToolContext) -> None:
        _ = ctx
        violation = validate_tool_args_against_sandbox(
            tool_name=tool_name,
            tool_args=raw_args,
            profile=profile,
            cwd=cwd,
        )
        if violation is not None:
            raise ToolPolicyError(violation)

    return _policy


def build_registry_output_limit_middleware(
    *,
    profile: SandboxProfile,
) -> Callable[..., Any]:
    """
    Registry middleware to cap large tool outputs at execution boundary.
    """

    async def _mw(call_next, tool_obj, raw_args, ctx, timeout, tool_call_id):
        result = await call_next(tool_obj, raw_args, ctx, timeout, tool_call_id)
        output = _truncate_json_like(result.output, max_chars=profile.max_output_chars)
        error = (
            _truncate_text(result.error_message, max_chars=profile.max_output_chars)
            if isinstance(result.error_message, str)
            else result.error_message
        )
        metadata = dict(result.metadata)
        return ToolResult(
            output=output,
            success=result.success,
            error_message=error,
            tool_name=result.tool_name,
            metadata=metadata,
            tool_call_id=result.tool_call_id,
        )

    return _mw


def resolve_sandbox_profile(
    *,
    tool_name: str,
    tool_args: dict[str, Any],
    run_context: dict[str, Any],
    default_profile: SandboxProfile | None,
    provider: SandboxProfileProvider | None,
) -> SandboxProfile | None:
    """
    Resolve the effective sandbox profile for a single tool invocation.

    Resolution order:
    1) profile returned by provider (if any),
    2) default_profile fallback.
    """
    if provider is not None:
        resolved = provider.resolve(
            tool_name=tool_name,
            tool_args=tool_args,
            run_context=run_context,
        )
        if resolved is not None:
            return resolved
    return default_profile


def apply_tool_output_limits(
    result: ToolResult[Any],
    *,
    profile: SandboxProfile | None,
) -> ToolResult[Any]:
    """
    Apply profile-level output truncation to a tool result.
    """
    if profile is None:
        return result
    return ToolResult(
        output=_truncate_json_like(result.output, max_chars=profile.max_output_chars),
        success=result.success,
        error_message=(
            _truncate_text(result.error_message, max_chars=profile.max_output_chars)
            if isinstance(result.error_message, str)
            else result.error_message
        ),
        tool_name=result.tool_name,
        metadata=dict(result.metadata),
        tool_call_id=result.tool_call_id,
        deferred=result.deferred,
    )


def _looks_like_path_key(key: str) -> bool:
    normalized = key.strip().lower()
    return any(token in normalized for token in ("path", "file", "dir", "cwd", "root"))


def _resolve_root(value: str, cwd: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def _resolve_path(value: str, cwd: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _iter_leaf_values(payload: dict[str, Any]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                _walk(child_prefix, child)
            return
        if isinstance(value, list):
            for idx, child in enumerate(value):
                _walk(f"{prefix}[{idx}]", child)
            return
        out.append((prefix.split(".")[-1], value))

    _walk("", payload)
    return out


def _extract_command_parts(tool_args: dict[str, Any]) -> list[str]:
    command = tool_args.get("command")
    if not isinstance(command, str) or not command.strip():
        return []
    args_value = tool_args.get("args")
    parts = [command.strip()]
    if isinstance(args_value, list):
        parts.extend(str(item) for item in args_value)
    return parts


def _is_command_allowed(command: str, allowlist: list[str]) -> bool:
    for item in allowlist:
        allowed = item.strip()
        if not allowed:
            continue
        if (
            command == allowed
            or command.startswith(allowed + "/")
            or command.startswith(allowed + " ")
        ):
            return True
    return False


def _truncate_text(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}... [truncated {len(value) - max_chars} chars]"


def _truncate_json_like(value: Any, *, max_chars: int) -> Any:
    remaining = max(max_chars, 0)

    def _truncate(value: Any) -> Any:
        nonlocal remaining
        if remaining <= 0:
            return "[truncated]"
        if isinstance(value, str):
            if len(value) <= remaining:
                remaining -= len(value)
                return value
            omitted = len(value) - remaining
            prefix = value[:remaining]
            remaining = 0
            return f"{prefix}... [truncated {omitted} chars]"
        if isinstance(value, list):
            items = []
            for item in value:
                if remaining <= 0:
                    items.append("[truncated]")
                    break
                items.append(_truncate(item))
            return items
        if isinstance(value, dict):
            items = {}
            for key, item in value.items():
                if remaining <= 0:
                    items["..."] = "[truncated]"
                    break
                items[str(key)] = _truncate(item)
            return items
        return value

    return _truncate(value)
