"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Prompt loading and caching utilities for agent system instructions.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined, Template, TemplateError

from ...llms.types import JSONValue
from ..errors import PromptAccessError, PromptResolutionError, PromptTemplateError

PROMPTS_DIR_ENV = "AFK_AGENT_PROMPTS_DIR"
DEFAULT_PROMPTS_DIR = Path(".agents/prompt")
_RESERVED_RENDER_KEYS = {"context", "ctx", "agent_name", "agent_class"}


def derive_auto_prompt_filename(agent_name: str) -> str:
    """
    Convert agent name into deterministic prompt filename (`UPPER_SNAKE.md`).

    Examples:
        - `ChatAgent` -> `CHAT_AGENT.md`
        - `chatagent` -> `CHAT_AGENT.md`
    """
    value = (agent_name or "").strip()
    if not value:
        raise PromptResolutionError(
            "agent name must be non-empty for auto prompt loading"
        )

    # Split camel/pascal boundaries first.
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)

    # Special-case lowercase *agent suffixes such as `chatagent`.
    lowered = normalized.lower()
    if lowered.endswith("agent") and not lowered.endswith("_agent"):
        prefix = normalized[:-5]
        if prefix and prefix[-1].isalnum():
            normalized = f"{prefix}_agent"

    normalized = re.sub(r"[^A-Za-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise PromptResolutionError(
            f"unable to derive auto prompt filename from agent name '{agent_name}'"
        )

    return f"{normalized.upper()}.md"


def resolve_prompts_dir(
    *,
    prompts_dir: str | Path | None,
    cwd: Path,
) -> Path:
    """
    Resolve effective prompt root from constructor arg/env/default.
    """
    source: str | Path | None = prompts_dir
    if source is None:
        env_value = os.getenv(PROMPTS_DIR_ENV)
        if env_value is not None:
            env_value = env_value.strip()
            if not env_value:
                raise PromptResolutionError(
                    f"environment variable {PROMPTS_DIR_ENV} is set but empty"
                )
            source = env_value
        else:
            source = DEFAULT_PROMPTS_DIR

    if isinstance(source, str):
        source = source.strip()
        if not source:
            raise PromptResolutionError("prompts_dir must not be an empty string")

    root = Path(source)
    resolved = root if root.is_absolute() else (cwd / root)
    resolved = resolved.resolve()
    if resolved.exists() and not resolved.is_dir():
        raise PromptResolutionError(f"prompts_dir is not a directory: {resolved}")
    return resolved


def resolve_prompt_file_path(
    *,
    prompt_root: Path,
    instruction_file: str | Path | None,
    agent_name: str,
) -> Path:
    """
    Resolve the target prompt file path with root-constrained access checks.

    Absolute paths bypass the security check but still require the file to exist.
    Relative paths are resolved against prompt_root and checked for directory escape.
    """
    if instruction_file is not None:
        candidate = Path(instruction_file)
        source = "instruction_file"
    else:
        candidate = Path(derive_auto_prompt_filename(agent_name))
        source = "auto_prompt"

    # Absolute paths bypass security check but must exist
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not resolved.exists() or not resolved.is_file():
            raise PromptResolutionError(
                f"prompt file not found for {source} "
                f"(path='{resolved}')"
            )
        return resolved

    # Relative paths: resolve against root and check for escape
    target = prompt_root / candidate
    resolved = target.resolve()
    if not _is_under(resolved, prompt_root):
        raise PromptAccessError(
            f"{source} path escapes configured prompts root "
            f"(path='{resolved}', root='{prompt_root}')"
        )
    if not resolved.exists() or not resolved.is_file():
        raise PromptResolutionError(
            f"prompt file not found for {source} "
            f"(path='{resolved}', root='{prompt_root}')"
        )
    return resolved


def build_prompt_render_context(
    *,
    context: dict[str, JSONValue],
    agent_name: str,
    agent_class: str,
) -> dict[str, Any]:
    """
    Build Jinja render context for prompt templates.
    """
    payload: dict[str, Any] = {}
    for key, value in context.items():
        if isinstance(key, str) and key not in _RESERVED_RENDER_KEYS:
            payload[key] = value

    # Reserved keys always win over conflicting user keys.
    payload["context"] = context
    payload["ctx"] = context
    payload["agent_name"] = agent_name
    payload["agent_class"] = agent_class
    return payload


@dataclass(frozen=True, slots=True)
class _PromptFileCacheEntry:
    stat_signature: tuple[int, int, int]
    content_hash: str


class PromptStore:
    """
    Process-wide prompt store with file + content + compiled-template caching.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._file_cache: dict[Path, _PromptFileCacheEntry] = {}
        self._text_pool: dict[str, str] = {}
        self._template_cache: dict[str, Template] = {}
        self._jinja = Environment(undefined=StrictUndefined, autoescape=False)

    def intern_text(self, text: str) -> str:
        """
        Deduplicate text content by SHA-256 hash.
        """
        digest = _sha256_text(text)
        with self._lock:
            cached = self._text_pool.get(digest)
            if cached is not None:
                return cached
            self._text_pool[digest] = text
            return text

    def load_prompt_file(self, prompt_path: Path) -> str:
        """
        Read a prompt file with stat-based cache invalidation.
        """
        resolved = prompt_path.resolve()
        if not resolved.exists() or not resolved.is_file():
            raise PromptResolutionError(f"prompt file not found: {resolved}")

        stat = resolved.stat()
        signature = (stat.st_mtime_ns, stat.st_size, stat.st_ino)

        with self._lock:
            entry = self._file_cache.get(resolved)
            if entry is not None and entry.stat_signature == signature:
                cached = self._text_pool.get(entry.content_hash)
                if cached is not None:
                    return cached

        text = resolved.read_text(encoding="utf-8")
        interned = self.intern_text(text)
        digest = _sha256_text(interned)
        with self._lock:
            self._file_cache[resolved] = _PromptFileCacheEntry(
                stat_signature=signature,
                content_hash=digest,
            )
            self._text_pool[digest] = interned
        return interned

    def render_template(
        self,
        template_text: str,
        render_context: dict[str, Any],
        *,
        prompt_path: Path | None = None,
    ) -> str:
        """
        Render a Jinja template with strict undefined-variable behavior.
        """
        digest = _sha256_text(template_text)
        with self._lock:
            template = self._template_cache.get(digest)

        if template is None:
            try:
                compiled = self._jinja.from_string(template_text)
            except TemplateError as exc:
                path = f" path='{prompt_path}'" if prompt_path is not None else ""
                raise PromptTemplateError(
                    f"invalid prompt template syntax{path}: {exc}"
                ) from exc
            with self._lock:
                template = self._template_cache.setdefault(digest, compiled)

        try:
            rendered = template.render(render_context)
        except TemplateError as exc:
            path = f" path='{prompt_path}'" if prompt_path is not None else ""
            raise PromptTemplateError(
                f"failed to render prompt template{path}: {exc}"
            ) from exc
        return self.intern_text(rendered)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


_PROMPT_STORE: PromptStore | None = None
_PROMPT_STORE_LOCK = threading.Lock()


def get_prompt_store() -> PromptStore:
    """
    Return process-wide prompt store singleton.
    """
    global _PROMPT_STORE
    if _PROMPT_STORE is not None:
        return _PROMPT_STORE
    with _PROMPT_STORE_LOCK:
        if _PROMPT_STORE is None:
            _PROMPT_STORE = PromptStore()
    return _PROMPT_STORE


def reset_prompt_store() -> None:
    """
    Reset the process-wide prompt store singleton.

    Intended for test isolation — clears all cached prompts, templates,
    and interned text so each test starts from a clean state.
    """
    global _PROMPT_STORE
    with _PROMPT_STORE_LOCK:
        _PROMPT_STORE = None
