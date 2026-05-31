"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Prebuilt tools for skill discovery, reading, and constrained command execution.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from afk.agents.errors import SkillAccessError, SkillCommandDeniedError
from afk.agents.skills import SkillStore, get_skill_store
from afk.agents.types import SkillRef, SkillToolPolicy
from afk.tools.core.base import Tool
from afk.tools.core.decorator import tool


class _EmptyArgs(BaseModel):
    pass


class _ReadSkillArgs(BaseModel):
    skill_name: str = Field(min_length=1)


class _ReadSkillFileArgs(BaseModel):
    skill_name: str = Field(min_length=1)
    relative_path: str = Field(min_length=1)
    max_chars: int = Field(default=20_000, ge=1, le=500_000)


class _RunSkillCommandArgs(BaseModel):
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    timeout_s: float | None = Field(default=None, gt=0)


def build_skill_tools(
    *,
    skills: list[SkillRef],
    policy: SkillToolPolicy,
    skill_store: SkillStore | None = None,
) -> list[Tool[Any, Any]]:
    """
    Build runtime-bound skill tools for one run.

    The returned tools are scoped to the provided resolved skills and enforce
    root-bound file access plus command policy checks.
    """
    # Reuse the process-wide cache unless an explicit test/store instance is provided.
    store = skill_store or get_skill_store()
    skills_by_name = {skill.name: skill for skill in skills}
    roots = {skill.name: Path(skill.skill_md_path).parent.resolve() for skill in skills}

    @tool(args_model=_EmptyArgs, name="list_skills")
    async def list_skills(args: _EmptyArgs) -> dict[str, Any]:
        """Return resolved skill metadata visible to the current run."""
        _ = args
        return {
            "skills": [
                {
                    "name": s.name,
                    "description": s.description,
                    "skill_md_path": s.skill_md_path,
                    "checksum": s.checksum,
                }
                for s in skills
            ]
        }

    @tool(args_model=_ReadSkillArgs, name="read_skill_md")
    async def read_skill_md(args: _ReadSkillArgs) -> dict[str, Any]:
        """Read one skill's `SKILL.md` content and checksum via the shared cache."""
        ref = skills_by_name.get(args.skill_name)
        if ref is None:
            raise SkillAccessError(f"Unknown skill '{args.skill_name}'")
        path = Path(ref.skill_md_path).resolve()
        # Enforce root-bound access even for internally resolved paths.
        _ensure_inside(path, roots[args.skill_name])
        doc = store.load_skill_md(path)
        return {
            "skill_name": ref.name,
            "path": str(path),
            "checksum": doc.checksum,
            "content": doc.content,
        }

    @tool(args_model=_ReadSkillFileArgs, name="read_skill_file")
    async def read_skill_file(args: _ReadSkillFileArgs) -> dict[str, Any]:
        """Read an additional file under a resolved skill directory."""
        root = roots.get(args.skill_name)
        if root is None:
            raise SkillAccessError(f"Unknown skill '{args.skill_name}'")
        target = (root / args.relative_path).resolve()
        # Prevent directory traversal outside of the skill root.
        _ensure_inside(target, root)
        if not target.exists() or not target.is_file():
            raise SkillAccessError(f"Skill file not found: {args.relative_path}")
        text, truncated = await asyncio.to_thread(
            _read_text_limited, target, args.max_chars
        )
        return {
            "skill_name": args.skill_name,
            "path": str(target),
            "content": text,
            "truncated": truncated,
        }

    @tool(args_model=_RunSkillCommandArgs, name="run_skill_command")
    async def run_skill_command(args: _RunSkillCommandArgs) -> dict[str, Any]:
        """Execute one allowlisted command with timeout and output limits."""
        command = args.command.strip()
        if not command:
            raise SkillCommandDeniedError("Command cannot be empty")
        if not _is_command_allowed(command, policy.command_allowlist):
            raise SkillCommandDeniedError(
                f"Command '{command}' is not allowlisted for skill execution"
            )

        all_parts = [command, *args.args]
        if policy.deny_shell_operators:
            # Keep execution deterministic and safe by denying shell chaining.
            forbidden = {"&&", "||", ";", "|", "`", "$("}
            for part in all_parts:
                if any(token in part for token in forbidden):
                    raise SkillCommandDeniedError(
                        "Command denied due to shell operator restrictions"
                    )

        cwd_path: Path
        if args.cwd:
            maybe = Path(args.cwd).expanduser().resolve()
            # Command working directory must stay inside one of the resolved skill roots.
            if not any(_is_inside(maybe, root) for root in roots.values()):
                raise SkillAccessError(
                    "Command cwd must be inside one of the skill roots"
                )
            cwd_path = maybe
        elif len(roots) == 1:
            cwd_path = next(iter(roots.values()))
        else:
            raise SkillAccessError(
                "Command cwd is required when multiple skill roots are available"
            )

        timeout_s = args.timeout_s or policy.command_timeout_s
        proc = await asyncio.create_subprocess_exec(
            command,
            *args.args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd_path),
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            raise SkillCommandDeniedError(
                f"Command timed out after {timeout_s} seconds: {command}"
            )

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        if len(stdout) > policy.max_stdout_chars:
            stdout = stdout[: policy.max_stdout_chars]
        if len(stderr) > policy.max_stderr_chars:
            stderr = stderr[: policy.max_stderr_chars]

        return {
            "command": [command, *args.args],
            "cwd": str(cwd_path),
            "exit_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

    return [list_skills, read_skill_md, read_skill_file, run_skill_command]


def _is_command_allowed(command: str, allowlist: list[str]) -> bool:
    """Return `True` when `command` matches an allowlisted executable prefix."""
    if not allowlist:
        return False
    for allowed in allowlist:
        prefix = allowed.strip()
        if not prefix:
            continue
        if command == prefix:
            return True
        if command.startswith(prefix + "/"):
            return True
    return False


def _ensure_inside(path: Path, root: Path) -> None:
    """Raise when `path` escapes the resolved skill `root`."""
    if not _is_inside(path, root):
        raise SkillAccessError(f"Path '{path}' escapes skill root '{root}'")


def _is_inside(path: Path, root: Path) -> bool:
    """Return `True` when `path` is contained within `root`."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_text_limited(path: Path, max_chars: int) -> tuple[str, bool]:
    """Read at most max_chars plus one sentinel character from a UTF-8 file."""
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read(max_chars + 1)
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True
