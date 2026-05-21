"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Module: factory.py.
"""

from __future__ import annotations

from typing import Any

from .errors import LLMConfigurationError


def _removed() -> None:
    """Raise v2 hard-break error for removed legacy factory entrypoints."""
    raise LLMConfigurationError(
        "Legacy factory APIs are removed in llms v2. "
        "Use create_llm_client(...) or LLMBuilder()."
    )


def register_llm_adapter(*args: Any, **kwargs: Any) -> None:
    """Removed legacy API shim."""
    _removed()


def available_llm_adapters(*args: Any, **kwargs: Any) -> list[str]:
    """Removed legacy API shim."""
    _removed()
    return []


def create_llm(*args: Any, **kwargs: Any) -> Any:
    """Removed legacy API shim."""
    _removed()


def create_llm_from_env(*args: Any, **kwargs: Any) -> Any:
    """Removed legacy API shim."""
    _removed()
