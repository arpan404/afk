"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

MCP (Model Context Protocol) package for AFK.

Exposes tools from a ``ToolRegistry`` as an MCP server using FastAPI
with JSON-RPC 2.0 wire format.

Quick start::

    from afk.tools import ToolRegistry, tool
    from afk.mcp import MCPServer

    registry = ToolRegistry()

    @tool(name="greet", description="Greet someone")
    def greet(name: str) -> str:
        return f"Hello, {name}!"

    registry.register(greet)

    server = MCPServer(
        registry,
        config=MCPServerConfig(host="127.0.0.1"),
    )  # Secure-by-default local bind
    server.run()  # starts on http://127.0.0.1:8000
"""

from .server import MCPServer, MCPServerConfig, create_mcp_server
from .store import (
    MCPRemoteCallError,
    MCPRemoteProtocolError,
    MCPRemoteTool,
    MCPServerRef,
    MCPServerResolutionError,
    MCPStore,
    MCPStoreError,
    get_mcp_store,
    reset_mcp_store,
)

__all__ = [
    "MCPServer",
    "MCPServerConfig",
    "create_mcp_server",
    "MCPServerRef",
    "MCPRemoteTool",
    "MCPStore",
    "MCPStoreError",
    "MCPServerResolutionError",
    "MCPRemoteProtocolError",
    "MCPRemoteCallError",
    "get_mcp_store",
    "reset_mcp_store",
]
