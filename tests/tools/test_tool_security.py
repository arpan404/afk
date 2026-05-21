from __future__ import annotations

import asyncio

from pydantic import BaseModel

from afk.tools import (
    SandboxProfile,
    ToolContext,
    ToolRegistry,
    build_registry_output_limit_middleware,
    build_registry_sandbox_policy,
    tool,
    validate_tool_args_against_sandbox,
)
from afk.tools.core.errors import ToolPolicyError


def run_async(coro):
    return asyncio.run(coro)


class _PathArgs(BaseModel):
    file_path: str


@tool(args_model=_PathArgs, name="read_any")
def read_any(args: _PathArgs) -> dict[str, str]:
    return {"path": args.file_path}


class _LargeArgs(BaseModel):
    text: str


@tool(args_model=_LargeArgs, name="large_text")
def large_text(args: _LargeArgs) -> dict[str, str]:
    return {"text": args.text}


def test_validate_tool_args_against_sandbox_denies_path_and_command_operator(tmp_path):
    profile = SandboxProfile(
        profile_id="strict",
        allow_command_execution=True,
        allowed_command_prefixes=["ls"],
        allowed_paths=[str(tmp_path / "allowed")],
        denied_paths=["/etc"],
        deny_shell_operators=True,
    )

    denied_path = validate_tool_args_against_sandbox(
        tool_name="read_file",
        tool_args={"file_path": "/etc/passwd"},
        profile=profile,
        cwd=tmp_path,
    )
    denied_cmd = validate_tool_args_against_sandbox(
        tool_name="run_skill_command",
        tool_args={"command": "ls && whoami", "args": []},
        profile=profile,
        cwd=tmp_path,
    )

    assert isinstance(denied_path, str)
    assert "denied" in denied_path.lower()
    assert isinstance(denied_cmd, str)
    assert "shell operator" in denied_cmd.lower()


def test_registry_sandbox_policy_blocks_at_call_time(tmp_path):
    profile = SandboxProfile(
        profile_id="registry",
        denied_paths=["/etc"],
    )
    registry = ToolRegistry(
        policy=build_registry_sandbox_policy(profile=profile, cwd=tmp_path)
    )
    registry.register(read_any)

    with_error = False
    try:
        run_async(
            registry.call(
                "read_any",
                {"file_path": "/etc/passwd"},
                ctx=ToolContext(),
            )
        )
    except ToolPolicyError:
        with_error = True

    assert with_error is True


def test_registry_output_limit_middleware_truncates_strings():
    profile = SandboxProfile(profile_id="cap", max_output_chars=64)
    registry = ToolRegistry(
        middlewares=[build_registry_output_limit_middleware(profile=profile)]
    )
    registry.register(large_text)

    result = run_async(
        registry.call(
            "large_text",
            {"text": "x" * 400},
            ctx=ToolContext(),
        )
    )

    assert result.success is True
    assert isinstance(result.output, dict)
    text = result.output.get("text") if isinstance(result.output, dict) else None
    assert isinstance(text, str)
    assert "truncated" in text
    assert len(text) < 200
