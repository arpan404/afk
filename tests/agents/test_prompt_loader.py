from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from afk.agents import (
    Agent,
    ChatAgent,
    PromptAccessError,
    PromptResolutionError,
    PromptTemplateError,
)


def run_async(coro):
    return asyncio.run(coro)


def _write_prompt(root: Path, filename: str, content: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / filename
    path.write_text(content, encoding="utf-8")
    return path


def test_auto_prompt_loads_for_chat_agent(tmp_path: Path):
    root = tmp_path / "prompts"
    _write_prompt(root, "CHAT_AGENT.md", "You are chat.")

    agent = ChatAgent(model="gpt-4.1-mini", prompts_dir=root)
    out = run_async(agent.resolve_instructions({}))
    assert out == "You are chat."


def test_auto_prompt_loads_for_lowercase_chatagent_name(tmp_path: Path):
    root = tmp_path / "prompts"
    _write_prompt(root, "CHAT_AGENT.md", "lowercase name mapping works")

    agent = Agent(model="gpt-4.1-mini", name="chatagent", prompts_dir=root)
    out = run_async(agent.resolve_instructions({}))
    assert out == "lowercase name mapping works"


def test_explicit_instruction_file_loads(tmp_path: Path):
    root = tmp_path / "prompts"
    _write_prompt(root, "custom.md", "explicit prompt file")

    agent = Agent(
        model="gpt-4.1-mini",
        name="Anything",
        instruction_file="custom.md",
        prompts_dir=root,
    )
    out = run_async(agent.resolve_instructions({}))
    assert out == "explicit prompt file"


def test_inline_instructions_win_over_instruction_file(tmp_path: Path):
    root = tmp_path / "prompts"
    _write_prompt(root, "custom.md", "{{ missing_required_key }}")

    agent = Agent(
        model="gpt-4.1-mini",
        name="InlineWins",
        instructions="INLINE",
        instruction_file="custom.md",
        prompts_dir=root,
    )
    out = run_async(agent.resolve_instructions({}))
    assert out == "INLINE"


def test_prompts_dir_argument_overrides_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    env_root = tmp_path / "env_prompts"
    arg_root = tmp_path / "arg_prompts"
    _write_prompt(env_root, "DUAL_SOURCE.md", "from env")
    _write_prompt(arg_root, "DUAL_SOURCE.md", "from arg")
    monkeypatch.setenv("AFK_AGENT_PROMPTS_DIR", str(env_root))

    agent = Agent(
        model="gpt-4.1-mini",
        name="DualSource",
        prompts_dir=arg_root,
    )
    out = run_async(agent.resolve_instructions({}))
    assert out == "from arg"


def test_env_prompts_dir_used_when_agent_prompts_dir_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    root = tmp_path / "env_prompts"
    _write_prompt(root, "ENV_AGENT.md", "loaded from env")
    monkeypatch.setenv("AFK_AGENT_PROMPTS_DIR", str(root))

    agent = Agent(model="gpt-4.1-mini", name="EnvAgent")
    out = run_async(agent.resolve_instructions({}))
    assert out == "loaded from env"


def test_default_prompts_dir_used_when_no_arg_and_no_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    monkeypatch.delenv("AFK_AGENT_PROMPTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_prompt(tmp_path / ".agents" / "prompt", "DEFAULT_AGENT.md", "default path")

    agent = Agent(model="gpt-4.1-mini", name="DefaultAgent")
    out = run_async(agent.resolve_instructions({}))
    assert out == "default path"


def test_missing_auto_prompt_file_raises_resolution_error(tmp_path: Path):
    agent = Agent(
        model="gpt-4.1-mini", name="MissingPrompt", prompts_dir=tmp_path / "prompts"
    )
    with pytest.raises(PromptResolutionError):
        run_async(agent.resolve_instructions({}))


def test_prompt_path_escape_raises_access_error(tmp_path: Path):
    root = tmp_path / "prompts"
    root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")

    agent = Agent(
        model="gpt-4.1-mini",
        name="EscapeTest",
        instruction_file="../outside.md",
        prompts_dir=root,
    )
    with pytest.raises(PromptAccessError):
        run_async(agent.resolve_instructions({}))


def test_absolute_path_bypasses_security_check(tmp_path: Path):
    """Absolute paths should bypass security check but still require file to exist."""
    from afk.agents.prompts.store import resolve_prompts_dir

    root = tmp_path / "prompts"
    root.mkdir(parents=True, exist_ok=True)

    # Create absolute path to existing file
    valid_file = root / "valid.md"
    valid_file.write_text("valid content", encoding="utf-8")
    abs_path = valid_file.resolve()

    # Get prompt root for context
    prompt_root = resolve_prompts_dir(prompts_dir=None, cwd=tmp_path)

    # Import directly - this is a module-level function
    from afk.agents.prompts.store import resolve_prompt_file_path

    # Should work - absolute path bypasses security check
    resolved = resolve_prompt_file_path(
        prompt_root=prompt_root,
        instruction_file=abs_path,
        agent_name="test",
    )
    assert resolved == abs_path

    # Non-existent absolute path should fail with PromptResolutionError (not access error)
    non_existent = Path("/nonexistent/file.txt")
    with pytest.raises(PromptResolutionError):
        resolve_prompt_file_path(
            prompt_root=prompt_root,
            instruction_file=non_existent,
            agent_name="test",
        )


def test_prompt_template_renders_context_and_reserved_keys(tmp_path: Path):
    root = tmp_path / "prompts"
    _write_prompt(
        root,
        "TEMPLATE_AGENT.md",
        (
            "user={{ user_id }}; account={{ context.account_id }}; "
            "locale={{ ctx.locale }}; name={{ agent_name }}; class={{ agent_class }}"
        ),
    )

    agent = Agent(model="gpt-4.1-mini", name="TemplateAgent", prompts_dir=root)
    out = run_async(
        agent.resolve_instructions(
            {
                "user_id": "u1",
                "account_id": "acc-1",
                "locale": "en-US",
                "agent_name": "user-value-ignored",
            }
        )
    )
    assert (
        out == "user=u1; account=acc-1; locale=en-US; name=TemplateAgent; class=Agent"
    )


def test_prompt_template_missing_variable_raises(tmp_path: Path):
    root = tmp_path / "prompts"
    _write_prompt(root, "MISSING_KEY.md", "value={{ required_key }}")

    agent = Agent(model="gpt-4.1-mini", name="MissingKey", prompts_dir=root)
    with pytest.raises(PromptTemplateError):
        run_async(agent.resolve_instructions({}))


def test_prompt_store_deduplicates_same_content(tmp_path: Path):
    root = tmp_path / "prompts"
    _write_prompt(root, "shared.md", "same prompt body")

    first = Agent(
        model="gpt-4.1-mini",
        name="First",
        instruction_file="shared.md",
        prompts_dir=root,
    )
    second = Agent(
        model="gpt-4.1-mini",
        name="Second",
        instruction_file="shared.md",
        prompts_dir=root,
    )

    out1 = run_async(first.resolve_instructions({}))
    out2 = run_async(second.resolve_instructions({}))
    assert out1 == "same prompt body"
    assert out2 == "same prompt body"
    assert out1 is out2


def test_prompt_cache_invalidates_when_file_changes(tmp_path: Path):
    root = tmp_path / "prompts"
    path = _write_prompt(root, "CACHE_AGENT.md", "version-1")
    agent = Agent(model="gpt-4.1-mini", name="CacheAgent", prompts_dir=root)

    out1 = run_async(agent.resolve_instructions({}))
    assert out1 == "version-1"

    time.sleep(0.002)
    path.write_text("version-2", encoding="utf-8")

    out2 = run_async(agent.resolve_instructions({}))
    assert out2 == "version-2"
    assert out1 != out2


def test_instruction_roles_append_after_prompt_file(tmp_path: Path):
    root = tmp_path / "prompts"
    _write_prompt(root, "ROLE_AGENT.md", "base prompt")

    def role(_ctx, _state):
        return "role tail"

    agent = Agent(
        model="gpt-4.1-mini",
        name="RoleAgent",
        prompts_dir=root,
        instruction_roles=[role],
    )

    out = run_async(agent.resolve_instructions({}))
    assert out == "base prompt\n\nrole tail"
