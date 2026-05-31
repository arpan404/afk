"""
Tests for afk.tools.prebuilts — FileAccessError, build_runtime_tools,
internal Pydantic models, and the _ensure_inside helper.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from afk.agents.types import SkillRef, SkillToolPolicy
from afk.tools.core.base import ToolContext, ToolResult
from afk.tools.core.errors import ToolExecutionError
from afk.tools.prebuilts.errors import FileAccessError
from afk.tools.prebuilts.runtime import (
    _ensure_inside,
    _ListDirectoryArgs,
    _ReadFileArgs,
    build_runtime_tools,
)
from afk.tools.prebuilts.skills import build_skill_tools

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):
    """Shorthand for running a coroutine in a fresh event loop."""
    return asyncio.run(coro)


def _call_tool(tool_obj, raw_args: dict, tool_name: str = "test") -> ToolResult:
    """Invoke a Tool via its async .call() and return the ToolResult."""
    return run(
        tool_obj.call(
            raw_args,
            ctx=ToolContext(),
            timeout=10.0,
            tool_call_id=f"call_{tool_name}",
        )
    )


# ===================================================================
# 1. FileAccessError
# ===================================================================

class TestFileAccessError:
    """Tests for afk.tools.prebuilts.errors.FileAccessError."""

    def test_is_subclass_of_tool_execution_error(self):
        assert issubclass(FileAccessError, ToolExecutionError)

    def test_can_be_instantiated_with_message(self):
        err = FileAccessError("cannot read /etc/passwd")
        assert isinstance(err, FileAccessError)
        assert isinstance(err, ToolExecutionError)

    def test_str_representation_contains_message(self):
        msg = "path escapes root"
        err = FileAccessError(msg)
        assert msg in str(err)


# ===================================================================
# 2. Pydantic internal models
# ===================================================================

class TestListDirectoryArgs:
    """Tests for _ListDirectoryArgs defaults and validation."""

    def test_defaults(self):
        args = _ListDirectoryArgs()
        assert args.path == "."
        assert args.max_entries == 200

    def test_custom_values(self):
        args = _ListDirectoryArgs(path="subdir", max_entries=10)
        assert args.path == "subdir"
        assert args.max_entries == 10

    def test_max_entries_must_be_positive(self):
        with pytest.raises(Exception):
            _ListDirectoryArgs(max_entries=0)

    def test_max_entries_upper_bound(self):
        with pytest.raises(Exception):
            _ListDirectoryArgs(max_entries=10_000)


class TestReadFileArgs:
    """Tests for _ReadFileArgs defaults and validation."""

    def test_requires_path(self):
        with pytest.raises(Exception):
            _ReadFileArgs()

    def test_path_provided(self):
        args = _ReadFileArgs(path="hello.txt")
        assert args.path == "hello.txt"
        assert args.max_chars == 20_000

    def test_custom_max_chars(self):
        args = _ReadFileArgs(path="f.py", max_chars=500)
        assert args.max_chars == 500

    def test_path_must_be_nonempty(self):
        with pytest.raises(Exception):
            _ReadFileArgs(path="")

    def test_max_chars_must_be_positive(self):
        with pytest.raises(Exception):
            _ReadFileArgs(path="f.py", max_chars=0)


# ===================================================================
# 3. _ensure_inside helper
# ===================================================================

class TestEnsureInside:
    """Tests for the _ensure_inside containment check."""

    def test_inside_path_passes(self):
        # /a/b/c is inside /a/b  =>  no error
        _ensure_inside(Path("/a/b/c"), Path("/a/b"))

    def test_exact_root_passes(self):
        # /a/b is inside /a/b  =>  no error (same directory)
        _ensure_inside(Path("/a/b"), Path("/a/b"))

    def test_outside_path_raises(self):
        # /a/b is NOT inside /a/b/c
        with pytest.raises(FileAccessError):
            _ensure_inside(Path("/a/b"), Path("/a/b/c"))

    def test_sibling_path_raises(self):
        with pytest.raises(FileAccessError):
            _ensure_inside(Path("/a/x"), Path("/a/b"))

    def test_parent_traversal_raises(self):
        # Even resolved, /a is not inside /a/b
        with pytest.raises(FileAccessError):
            _ensure_inside(Path("/a"), Path("/a/b"))


# ===================================================================
# 4. build_runtime_tools — basics
# ===================================================================

class TestBuildRuntimeTools:
    """Tests that build_runtime_tools returns the expected tool objects."""

    def test_returns_two_tools(self, tmp_path: Path):
        tools = build_runtime_tools(root_dir=tmp_path)
        assert len(tools) == 2

    def test_tool_names(self, tmp_path: Path):
        tools = build_runtime_tools(root_dir=tmp_path)
        names = {t.spec.name for t in tools}
        assert names == {"list_directory", "read_file"}


# ===================================================================
# 5. list_directory tool
# ===================================================================

class TestListDirectoryTool:
    """Tests for the list_directory runtime tool."""

    @pytest.fixture()
    def setup(self, tmp_path: Path):
        """Create a small directory tree and return tools + root."""
        (tmp_path / "file_a.txt").write_text("aaa")
        (tmp_path / "file_b.txt").write_text("bbb")
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested")

        tools = build_runtime_tools(root_dir=tmp_path)
        list_dir_tool = [t for t in tools if t.spec.name == "list_directory"][0]
        return list_dir_tool, tmp_path

    def test_lists_files_in_root(self, setup):
        tool, root = setup
        result = _call_tool(tool, {"path": "."}, "list_dir")

        assert result.success is True
        entries = result.output["entries"]
        names = {e["name"] for e in entries}
        assert "file_a.txt" in names
        assert "file_b.txt" in names
        assert "subdir" in names

    def test_entries_have_expected_keys(self, setup):
        tool, root = setup
        result = _call_tool(tool, {"path": "."}, "list_dir")

        assert result.success is True
        for entry in result.output["entries"]:
            assert "name" in entry
            assert "path" in entry
            assert "is_dir" in entry
            assert "is_file" in entry

    def test_entry_types_are_correct(self, setup):
        tool, root = setup
        result = _call_tool(tool, {"path": "."}, "list_dir")

        entries_by_name = {e["name"]: e for e in result.output["entries"]}
        assert entries_by_name["file_a.txt"]["is_file"] is True
        assert entries_by_name["file_a.txt"]["is_dir"] is False
        assert entries_by_name["subdir"]["is_dir"] is True
        assert entries_by_name["subdir"]["is_file"] is False

    def test_lists_subdirectory(self, setup):
        tool, root = setup
        result = _call_tool(tool, {"path": "subdir"}, "list_dir")

        assert result.success is True
        names = {e["name"] for e in result.output["entries"]}
        assert "nested.txt" in names

    def test_max_entries_limits_results(self, setup):
        tool, root = setup
        # The root has 3 items (file_a.txt, file_b.txt, subdir).
        # Limiting to 2 should cap the results.
        result = _call_tool(tool, {"path": ".", "max_entries": 2}, "list_dir")

        assert result.success is True
        assert len(result.output["entries"]) == 2

    def test_nonexistent_directory_fails(self, setup):
        tool, root = setup
        result = _call_tool(tool, {"path": "does_not_exist"}, "list_dir")

        assert result.success is False
        assert result.error_message is not None
        assert "does_not_exist" in result.error_message or "not found" in result.error_message.lower()

    def test_path_traversal_fails(self, setup):
        tool, root = setup
        result = _call_tool(tool, {"path": "../../etc"}, "list_dir")

        assert result.success is False
        assert result.error_message is not None
        assert "escapes" in result.error_message.lower() or "root" in result.error_message.lower()

    def test_absolute_path_outside_root_fails(self, setup):
        tool, root = setup
        result = _call_tool(tool, {"path": "/etc"}, "list_dir")

        assert result.success is False
        assert result.error_message is not None


# ===================================================================
# 6. read_file tool
# ===================================================================

class TestReadFileTool:
    """Tests for the read_file runtime tool."""

    @pytest.fixture()
    def setup(self, tmp_path: Path):
        """Create files and return tools + root."""
        (tmp_path / "hello.txt").write_text("Hello, world!")
        (tmp_path / "big.txt").write_text("x" * 50_000)

        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "deep.txt").write_text("deep content")

        tools = build_runtime_tools(root_dir=tmp_path)
        read_tool = [t for t in tools if t.spec.name == "read_file"][0]
        return read_tool, tmp_path

    def test_reads_file_content(self, setup):
        tool, root = setup
        result = _call_tool(tool, {"path": "hello.txt"}, "read_file")

        assert result.success is True
        assert result.output["content"] == "Hello, world!"
        assert result.output["truncated"] is False

    def test_reads_file_in_subdirectory(self, setup):
        tool, root = setup
        result = _call_tool(tool, {"path": "subdir/deep.txt"}, "read_file")

        assert result.success is True
        assert result.output["content"] == "deep content"

    def test_truncates_when_file_exceeds_max_chars(self, setup):
        tool, root = setup
        result = _call_tool(tool, {"path": "big.txt", "max_chars": 100}, "read_file")

        assert result.success is True
        assert result.output["truncated"] is True
        assert len(result.output["content"]) == 100

    def test_no_truncation_when_under_limit(self, setup):
        tool, root = setup
        result = _call_tool(tool, {"path": "hello.txt", "max_chars": 50_000}, "read_file")

        assert result.success is True
        assert result.output["truncated"] is False
        assert result.output["content"] == "Hello, world!"

    def test_nonexistent_file_fails(self, setup):
        tool, root = setup
        result = _call_tool(tool, {"path": "missing.txt"}, "read_file")

        assert result.success is False
        assert result.error_message is not None
        assert "missing.txt" in result.error_message or "not found" in result.error_message.lower()

    def test_path_traversal_fails(self, setup):
        tool, root = setup
        result = _call_tool(tool, {"path": "../../etc/passwd"}, "read_file")

        assert result.success is False
        assert result.error_message is not None
        assert "escapes" in result.error_message.lower() or "root" in result.error_message.lower()

    def test_absolute_path_outside_root_fails(self, setup):
        tool, root = setup
        result = _call_tool(tool, {"path": "/etc/passwd"}, "read_file")

        assert result.success is False
        assert result.error_message is not None

    def test_directory_as_file_fails(self, setup):
        tool, root = setup
        result = _call_tool(tool, {"path": "subdir"}, "read_file")

        assert result.success is False
        assert result.error_message is not None


class TestSkillCommandTool:
    """Tests for constrained skill command execution cwd behavior."""

    def _skill_ref(self, root: Path, name: str) -> SkillRef:
        root.mkdir(parents=True, exist_ok=True)
        skill_md = root / "SKILL.md"
        skill_md.write_text(f"---\nname: {name}\n---\n", encoding="utf-8")
        return SkillRef(
            name=name,
            description="test skill",
            root_dir=str(root),
            skill_md_path=str(skill_md),
        )

    def test_command_without_cwd_uses_single_skill_root(self, tmp_path: Path):
        skill_root = tmp_path / "skill-a"
        ref = self._skill_ref(skill_root, "skill-a")
        tools = build_skill_tools(
            skills=[ref],
            policy=SkillToolPolicy(command_allowlist=["pwd"]),
        )
        run_tool = [t for t in tools if t.spec.name == "run_skill_command"][0]

        result = _call_tool(run_tool, {"command": "pwd"}, "run_skill_command")

        assert result.success is True
        assert result.output["cwd"] == str(skill_root.resolve())
        assert result.output["stdout"].strip() == str(skill_root.resolve())

    def test_command_without_cwd_fails_for_multiple_skill_roots(self, tmp_path: Path):
        ref_a = self._skill_ref(tmp_path / "skill-a", "skill-a")
        ref_b = self._skill_ref(tmp_path / "skill-b", "skill-b")
        tools = build_skill_tools(
            skills=[ref_a, ref_b],
            policy=SkillToolPolicy(command_allowlist=["pwd"]),
        )
        run_tool = [t for t in tools if t.spec.name == "run_skill_command"][0]

        result = _call_tool(run_tool, {"command": "pwd"}, "run_skill_command")

        assert result.success is False
        assert "cwd is required" in (result.error_message or "")
