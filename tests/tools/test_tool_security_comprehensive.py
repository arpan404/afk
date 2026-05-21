"""
Comprehensive tests for afk.tools.security module.

Covers SandboxProfile defaults/custom values, validate_tool_args_against_sandbox,
build_registry_sandbox_policy, apply_tool_output_limits, resolve_sandbox_profile,
and private helpers (_is_command_allowed, _truncate_text, _truncate_json_like,
_looks_like_path_key, _iter_leaf_values, _extract_command_parts).
"""

from __future__ import annotations

import pytest

from afk.tools.core.base import ToolContext, ToolResult
from afk.tools.core.errors import ToolPolicyError
from afk.tools.security import (
    SandboxProfile,
    _extract_command_parts,
    _is_command_allowed,
    _iter_leaf_values,
    _looks_like_path_key,
    _truncate_json_like,
    _truncate_text,
    apply_tool_output_limits,
    build_registry_sandbox_policy,
    resolve_sandbox_profile,
    validate_tool_args_against_sandbox,
)


# ---------------------------------------------------------------------------
# 1. SandboxProfile
# ---------------------------------------------------------------------------
class TestSandboxProfile:
    """Tests for SandboxProfile dataclass defaults and custom values."""

    def test_default_values(self):
        profile = SandboxProfile()
        assert profile.profile_id == "default"
        assert profile.allow_network is False
        assert profile.allow_command_execution is False
        assert profile.allowed_command_prefixes == []
        assert profile.deny_shell_operators is True
        assert profile.allowed_paths == []
        assert profile.denied_paths == []
        assert profile.command_timeout_s is None
        assert profile.max_output_chars == 20_000

    def test_custom_values(self):
        profile = SandboxProfile(
            profile_id="custom",
            allow_network=True,
            allow_command_execution=False,
            allowed_command_prefixes=["git", "npm"],
            deny_shell_operators=False,
            allowed_paths=["/home/user/project"],
            denied_paths=["/etc", "/var"],
            command_timeout_s=30.0,
            max_output_chars=5000,
        )
        assert profile.profile_id == "custom"
        assert profile.allow_network is True
        assert profile.allow_command_execution is False
        assert profile.allowed_command_prefixes == ["git", "npm"]
        assert profile.deny_shell_operators is False
        assert profile.allowed_paths == ["/home/user/project"]
        assert profile.denied_paths == ["/etc", "/var"]
        assert profile.command_timeout_s == 30.0
        assert profile.max_output_chars == 5000


# ---------------------------------------------------------------------------
# 2. validate_tool_args_against_sandbox
# ---------------------------------------------------------------------------
class TestValidateToolArgsAgainstSandbox:
    """Tests for the main validation function."""

    # -- Network access denied --

    @pytest.mark.parametrize(
        "tool_name",
        ["webfetch", "websearch", "web_fetch", "web_search"],
    )
    def test_network_denied_blocks_network_tool_names(self, tmp_path, tool_name):
        profile = SandboxProfile(allow_network=False)
        result = validate_tool_args_against_sandbox(
            tool_name=tool_name,
            tool_args={},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is not None
        assert "Network access denied" in result

    @pytest.mark.parametrize(
        "tool_name",
        ["WebFetch", "WEBSEARCH", "Web_Fetch", "Web_Search"],
    )
    def test_network_denied_blocks_network_tool_names_case_insensitive(
        self, tmp_path, tool_name
    ):
        profile = SandboxProfile(allow_network=False)
        result = validate_tool_args_against_sandbox(
            tool_name=tool_name,
            tool_args={},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is not None
        assert "Network access denied" in result

    @pytest.mark.parametrize(
        "key,value",
        [
            ("url", "http://example.com"),
            ("url", "https://example.com"),
            ("uri", "http://evil.com/data"),
            ("uri", "https://api.example.com/v1"),
        ],
    )
    def test_network_denied_blocks_url_args(self, tmp_path, key, value):
        profile = SandboxProfile(allow_network=False)
        result = validate_tool_args_against_sandbox(
            tool_name="some_tool",
            tool_args={key: value},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is not None
        assert "Network URL argument denied" in result

    def test_network_allowed_passes_network_tools(self, tmp_path):
        profile = SandboxProfile(allow_network=True)
        result = validate_tool_args_against_sandbox(
            tool_name="webfetch",
            tool_args={"url": "https://example.com"},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is None

    # -- Command execution denied / allowed --

    def test_command_execution_denied_blocks_commands(self, tmp_path):
        profile = SandboxProfile(allow_command_execution=False)
        result = validate_tool_args_against_sandbox(
            tool_name="run_command",
            tool_args={"command": "ls -la"},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is not None
        assert "Command execution denied" in result

    def test_command_execution_allowed_passes(self, tmp_path):
        profile = SandboxProfile(
            allow_command_execution=True,
            allowed_command_prefixes=["ls"],
            deny_shell_operators=False,
        )
        result = validate_tool_args_against_sandbox(
            tool_name="run_command",
            tool_args={"command": "ls -la"},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is None

    # -- allowed_command_prefixes --

    def test_allowed_command_prefixes_exact_match(self, tmp_path):
        profile = SandboxProfile(
            allow_command_execution=True,
            allowed_command_prefixes=["git", "npm"],
            deny_shell_operators=False,
        )
        result = validate_tool_args_against_sandbox(
            tool_name="exec",
            tool_args={"command": "git"},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is None

    def test_allowed_command_prefixes_path_prefix_match(self, tmp_path):
        profile = SandboxProfile(
            allow_command_execution=True,
            allowed_command_prefixes=["git"],
            deny_shell_operators=False,
        )
        result = validate_tool_args_against_sandbox(
            tool_name="exec",
            tool_args={"command": "git/some-subcommand"},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is None

    def test_command_not_in_allowlist_blocked(self, tmp_path):
        profile = SandboxProfile(
            allow_command_execution=True,
            allowed_command_prefixes=["git", "npm"],
            deny_shell_operators=False,
        )
        result = validate_tool_args_against_sandbox(
            tool_name="exec",
            tool_args={"command": "rm -rf /"},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is not None
        assert "not allowlisted" in result

    # -- Shell operators --

    @pytest.mark.parametrize(
        "operator",
        ["&&", "||", ";", "|", "`", "$(", ">", ">>", "<", "<<", "&"],
    )
    def test_shell_operators_blocked_when_deny_enabled(self, tmp_path, operator):
        profile = SandboxProfile(
            allow_command_execution=True,
            allowed_command_prefixes=["ls"],
            deny_shell_operators=True,
        )
        result = validate_tool_args_against_sandbox(
            tool_name="run",
            tool_args={"command": f"ls {operator} whoami"},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is not None
        assert "shell operator" in result.lower()

    def test_shell_operators_allowed_when_deny_disabled(self, tmp_path):
        profile = SandboxProfile(
            allow_command_execution=True,
            allowed_command_prefixes=["ls && whoami"],
            deny_shell_operators=False,
        )
        result = validate_tool_args_against_sandbox(
            tool_name="run",
            tool_args={"command": "ls && whoami"},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is None

    # -- denied_paths --

    def test_denied_paths_blocks_file_in_denied_dir(self, tmp_path):
        profile = SandboxProfile(denied_paths=["/etc"])
        result = validate_tool_args_against_sandbox(
            tool_name="read",
            tool_args={"file_path": "/etc/passwd"},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is not None
        assert "denied" in result.lower()

    # -- allowed_paths --

    def test_allowed_paths_allows_files_under_allowed_dirs(self, tmp_path):
        allowed = tmp_path / "workspace"
        allowed.mkdir()
        profile = SandboxProfile(allowed_paths=[str(allowed)])
        result = validate_tool_args_against_sandbox(
            tool_name="read",
            tool_args={"file_path": str(allowed / "data.txt")},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is None

    def test_allowed_paths_blocks_files_outside_allowed_dirs(self, tmp_path):
        allowed = tmp_path / "workspace"
        allowed.mkdir()
        profile = SandboxProfile(allowed_paths=[str(allowed)])
        result = validate_tool_args_against_sandbox(
            tool_name="read",
            tool_args={"file_path": "/usr/local/bin/something"},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is not None
        assert "not in allowlist" in result

    # -- Path-like key detection --

    @pytest.mark.parametrize(
        "key",
        ["path", "file", "dir", "cwd", "root", "file_path", "root_dir", "CWD", "FilePath"],
    )
    def test_path_like_keys_detected(self, tmp_path, key):
        denied_dir = tmp_path / "forbidden"
        denied_dir.mkdir()
        profile = SandboxProfile(denied_paths=[str(denied_dir)])
        result = validate_tool_args_against_sandbox(
            tool_name="tool",
            tool_args={key: str(denied_dir / "secret.txt")},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is not None
        assert "denied" in result.lower()

    def test_non_path_keys_ignored_for_path_validation(self, tmp_path):
        profile = SandboxProfile(denied_paths=["/etc"])
        result = validate_tool_args_against_sandbox(
            tool_name="tool",
            tool_args={"name": "/etc/passwd", "description": "/etc/shadow"},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is None

    def test_http_urls_in_path_keys_not_treated_as_file_paths(self, tmp_path):
        profile = SandboxProfile(
            allow_network=True,
            denied_paths=["/etc"],
        )
        result = validate_tool_args_against_sandbox(
            tool_name="tool",
            tool_args={"file_path": "https://example.com/etc/data"},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is None

    def test_returns_none_when_no_violations(self, tmp_path):
        profile = SandboxProfile(
            allow_network=True,
            allow_command_execution=True,
            deny_shell_operators=False,
        )
        result = validate_tool_args_against_sandbox(
            tool_name="safe_tool",
            tool_args={"data": "hello"},
            profile=profile,
            cwd=tmp_path,
        )
        assert result is None


# ---------------------------------------------------------------------------
# 3. build_registry_sandbox_policy
# ---------------------------------------------------------------------------
class TestBuildRegistrySandboxPolicy:
    """Tests for build_registry_sandbox_policy."""

    def test_returns_callable(self, tmp_path):
        profile = SandboxProfile()
        policy = build_registry_sandbox_policy(profile=profile, cwd=tmp_path)
        assert callable(policy)

    def test_callable_raises_tool_policy_error_on_violation(self, tmp_path):
        profile = SandboxProfile(allow_network=False)
        policy = build_registry_sandbox_policy(profile=profile, cwd=tmp_path)
        ctx = ToolContext()
        with pytest.raises(ToolPolicyError, match="Network access denied"):
            policy("webfetch", {}, ctx)

    def test_callable_passes_when_no_violation(self, tmp_path):
        profile = SandboxProfile(allow_network=True, deny_shell_operators=False)
        policy = build_registry_sandbox_policy(profile=profile, cwd=tmp_path)
        ctx = ToolContext()
        # Should not raise
        policy("webfetch", {"url": "https://example.com"}, ctx)


# ---------------------------------------------------------------------------
# 4. apply_tool_output_limits
# ---------------------------------------------------------------------------
class TestApplyToolOutputLimits:
    """Tests for apply_tool_output_limits."""

    def _make_result(self, output=None, error_message=None):
        return ToolResult(
            output=output,
            success=True,
            error_message=error_message,
            tool_name="test_tool",
            tool_call_id="call_1",
        )

    def test_none_profile_returns_result_unchanged(self):
        result = self._make_result(output="hello world")
        returned = apply_tool_output_limits(result, profile=None)
        assert returned is result

    def test_truncates_long_string_output(self):
        long_text = "a" * 50
        profile = SandboxProfile(max_output_chars=20)
        result = self._make_result(output=long_text)
        returned = apply_tool_output_limits(result, profile=profile)
        assert isinstance(returned.output, str)
        assert len(returned.output) < len(long_text)
        assert "truncated" in returned.output

    def test_truncates_long_error_message(self):
        long_error = "e" * 100
        profile = SandboxProfile(max_output_chars=30)
        result = self._make_result(error_message=long_error)
        returned = apply_tool_output_limits(result, profile=profile)
        assert isinstance(returned.error_message, str)
        assert "truncated" in returned.error_message

    def test_truncates_nested_dict_values(self):
        nested = {"key": "v" * 100}
        profile = SandboxProfile(max_output_chars=20)
        result = self._make_result(output=nested)
        returned = apply_tool_output_limits(result, profile=profile)
        assert isinstance(returned.output, dict)
        assert "truncated" in returned.output["key"]

    def test_truncates_nested_list_values(self):
        nested = ["z" * 100, "short"]
        profile = SandboxProfile(max_output_chars=20)
        result = self._make_result(output=nested)
        returned = apply_tool_output_limits(result, profile=profile)
        assert isinstance(returned.output, list)
        assert "truncated" in returned.output[0]
        assert returned.output[1] == "[truncated]"

    def test_short_output_passes_through_unchanged(self):
        profile = SandboxProfile(max_output_chars=1000)
        result = self._make_result(output="short")
        returned = apply_tool_output_limits(result, profile=profile)
        assert returned.output == "short"

    def test_none_error_message_preserved(self):
        profile = SandboxProfile(max_output_chars=10)
        result = self._make_result(output="ok", error_message=None)
        returned = apply_tool_output_limits(result, profile=profile)
        assert returned.error_message is None


# ---------------------------------------------------------------------------
# 5. resolve_sandbox_profile
# ---------------------------------------------------------------------------
class TestResolveSandboxProfile:
    """Tests for resolve_sandbox_profile."""

    class _MockProvider:
        """A simple mock that returns a pre-set profile or None."""

        def __init__(self, return_value):
            self._return_value = return_value

        def resolve(self, *, tool_name, tool_args, run_context):
            return self._return_value

    def test_returns_provider_result_when_available(self):
        provider_profile = SandboxProfile(profile_id="from_provider")
        provider = self._MockProvider(provider_profile)
        default = SandboxProfile(profile_id="default_fallback")
        result = resolve_sandbox_profile(
            tool_name="tool",
            tool_args={},
            run_context={},
            default_profile=default,
            provider=provider,
        )
        assert result is provider_profile
        assert result.profile_id == "from_provider"

    def test_falls_back_to_default_when_provider_returns_none(self):
        provider = self._MockProvider(None)
        default = SandboxProfile(profile_id="fallback")
        result = resolve_sandbox_profile(
            tool_name="tool",
            tool_args={},
            run_context={},
            default_profile=default,
            provider=provider,
        )
        assert result is default
        assert result.profile_id == "fallback"

    def test_returns_none_when_both_none(self):
        result = resolve_sandbox_profile(
            tool_name="tool",
            tool_args={},
            run_context={},
            default_profile=None,
            provider=None,
        )
        assert result is None

    def test_returns_default_when_no_provider(self):
        default = SandboxProfile(profile_id="only_default")
        result = resolve_sandbox_profile(
            tool_name="tool",
            tool_args={},
            run_context={},
            default_profile=default,
            provider=None,
        )
        assert result is default


# ---------------------------------------------------------------------------
# 6. _is_command_allowed (private helper)
# ---------------------------------------------------------------------------
class TestIsCommandAllowed:
    """Tests for the _is_command_allowed private helper."""

    def test_exact_match_returns_true(self):
        assert _is_command_allowed("git", ["git", "npm"]) is True

    def test_path_prefix_match_returns_true(self):
        assert _is_command_allowed("git/subcommand", ["git"]) is True

    def test_no_match_returns_false(self):
        assert _is_command_allowed("rm", ["git", "npm"]) is False

    def test_empty_allowlist_items_skipped(self):
        assert _is_command_allowed("git", ["", "  ", "git"]) is True

    def test_empty_allowlist(self):
        assert _is_command_allowed("git", []) is False

    def test_partial_name_no_match(self):
        # "gitconfig" should not match "git" (no path separator)
        assert _is_command_allowed("gitconfig", ["git"]) is False

    def test_command_with_whitespace_in_allowlist(self):
        # Leading/trailing spaces in allowlist items are stripped
        assert _is_command_allowed("npm", [" npm "]) is True


# ---------------------------------------------------------------------------
# 7. _truncate_text and _truncate_json_like (private helpers)
# ---------------------------------------------------------------------------
class TestTruncateText:
    """Tests for _truncate_text."""

    def test_text_within_limit_unchanged(self):
        assert _truncate_text("hello", max_chars=10) == "hello"

    def test_text_at_limit_unchanged(self):
        text = "a" * 10
        assert _truncate_text(text, max_chars=10) == text

    def test_text_over_limit_truncated_with_message(self):
        text = "a" * 50
        result = _truncate_text(text, max_chars=20)
        assert result.startswith("a" * 20)
        assert "truncated" in result
        assert "30 chars" in result


class TestTruncateJsonLike:
    """Tests for _truncate_json_like."""

    def test_string_within_limit(self):
        assert _truncate_json_like("short", max_chars=100) == "short"

    def test_string_over_limit(self):
        result = _truncate_json_like("a" * 50, max_chars=10)
        assert isinstance(result, str)
        assert "truncated" in result

    def test_list_items_truncated_recursively(self):
        data = ["a" * 50, "ok"]
        result = _truncate_json_like(data, max_chars=10)
        assert isinstance(result, list)
        assert len(result) == 2
        assert "truncated" in result[0]
        assert result[1] == "[truncated]"

    def test_dict_values_truncated_recursively(self):
        data = {"key1": "b" * 50, "key2": 42}
        result = _truncate_json_like(data, max_chars=10)
        assert isinstance(result, dict)
        assert "truncated" in result["key1"]
        assert result["..."] == "[truncated]"

    def test_nested_dict_list_structure(self):
        data = {"items": [{"name": "x" * 100}]}
        result = _truncate_json_like(data, max_chars=20)
        assert isinstance(result, dict)
        assert isinstance(result["items"], list)
        assert "truncated" in result["items"][0]["name"]

    def test_non_string_non_container_passthrough(self):
        assert _truncate_json_like(42, max_chars=5) == 42
        assert _truncate_json_like(3.14, max_chars=5) == 3.14
        assert _truncate_json_like(None, max_chars=5) is None
        assert _truncate_json_like(True, max_chars=5) is True


# ---------------------------------------------------------------------------
# Additional helpers: _looks_like_path_key, _iter_leaf_values, _extract_command_parts
# ---------------------------------------------------------------------------
class TestLooksLikePathKey:
    """Tests for _looks_like_path_key."""

    @pytest.mark.parametrize(
        "key",
        ["path", "file", "dir", "cwd", "root", "file_path", "root_dir", "CWD", "FilePath"],
    )
    def test_path_like_keys_detected(self, key):
        assert _looks_like_path_key(key) is True

    @pytest.mark.parametrize(
        "key",
        ["name", "description", "command", "url", "text", "data", "value"],
    )
    def test_non_path_keys_not_detected(self, key):
        assert _looks_like_path_key(key) is False

    def test_whitespace_stripped(self):
        assert _looks_like_path_key("  file  ") is True


class TestIterLeafValues:
    """Tests for _iter_leaf_values."""

    def test_flat_dict(self):
        result = _iter_leaf_values({"a": 1, "b": "hello"})
        keys = [k for k, _ in result]
        vals = [v for _, v in result]
        assert "a" in keys
        assert "b" in keys
        assert 1 in vals
        assert "hello" in vals

    def test_nested_dict(self):
        result = _iter_leaf_values({"outer": {"inner": "val"}})
        assert len(result) == 1
        assert result[0] == ("inner", "val")

    def test_list_in_dict(self):
        result = _iter_leaf_values({"items": [10, 20]})
        vals = [v for _, v in result]
        assert 10 in vals
        assert 20 in vals

    def test_empty_dict(self):
        assert _iter_leaf_values({}) == []

    def test_deeply_nested(self):
        data = {"a": {"b": {"c": "deep"}}}
        result = _iter_leaf_values(data)
        assert len(result) == 1
        assert result[0] == ("c", "deep")


class TestExtractCommandParts:
    """Tests for _extract_command_parts."""

    def test_command_only(self):
        result = _extract_command_parts({"command": "ls"})
        assert result == ["ls"]

    def test_command_with_args_list(self):
        result = _extract_command_parts({"command": "git", "args": ["status", "-s"]})
        assert result == ["git", "status", "-s"]

    def test_no_command_key(self):
        result = _extract_command_parts({"other": "value"})
        assert result == []

    def test_empty_command_string(self):
        result = _extract_command_parts({"command": "  "})
        assert result == []

    def test_non_string_command(self):
        result = _extract_command_parts({"command": 123})
        assert result == []

    def test_args_not_list_ignored(self):
        result = _extract_command_parts({"command": "echo", "args": "hello"})
        assert result == ["echo"]

    def test_command_whitespace_stripped(self):
        result = _extract_command_parts({"command": "  git  "})
        assert result == ["git"]
