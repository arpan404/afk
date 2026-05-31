"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Type models and error hierarchy for the external MCP store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MCPServerRef:
    """
    External MCP server reference.

    Attributes:
        name: Stable local alias for the external MCP server.
        url: JSON-RPC endpoint URL (for example: http://localhost:8000/mcp).
        headers: Optional HTTP headers sent with each request.
        timeout_s: HTTP timeout in seconds for `tools/list` and `tools/call`.
        prefix_tools: Whether generated AFK tool names should be prefixed.
        tool_name_prefix: Optional explicit prefix override.
    """

    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 20.0
    prefix_tools: bool = True
    tool_name_prefix: str | None = None
    require_https: bool = True
    allow_private_networks: bool = False
    allowed_hostnames: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MCPRemoteTool:
    """Normalized remote MCP tool descriptor."""

    server_name: str
    name: str
    qualified_name: str
    description: str
    input_schema: dict[str, Any]


class MCPStoreError(RuntimeError):
    """Base MCP store error."""


class MCPServerResolutionError(MCPStoreError):
    """Raised when an MCP server reference cannot be resolved."""


class MCPRemoteProtocolError(MCPStoreError):
    """Raised when remote MCP response shape is invalid."""


class MCPRemoteCallError(MCPStoreError):
    """Raised when remote MCP server returns a call error."""
