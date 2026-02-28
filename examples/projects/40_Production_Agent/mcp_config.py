"""
---
name: Production Agent — MCP Configuration
description: MCP (Model Context Protocol) server references for connecting to external tool servers.
tags: [mcp, mcp-server, external-tools, integration]
---
---
MCPServerRef defines a connection to an external MCP tool server. MCP lets agents discover and
call tools hosted on remote servers using a standardized protocol. This enables tool sharing
across multiple agents and languages without bundling tool code directly.

Each MCPServerRef specifies:
  - name: Human-readable identifier for the server
  - url: The MCP server endpoint
  - headers: Optional auth headers (e.g., API keys)
  - timeout_s: Connection/call timeout
  - prefix_tools: If True, tool names are prefixed with tool_name_prefix
  - tool_name_prefix: Prefix string (defaults to server name)

The agent receives MCP tools alongside its local tools. When the agent calls an MCP tool,
the Runner routes the call to the MCP server transparently.

NOTE: The MCP servers below are examples. Replace URLs with real MCP server endpoints in
production. Comment out mcp_servers in agents.py if you don't have live MCP servers.
---
"""

from afk.mcp.store.types import MCPServerRef  # <- MCPServerRef defines a connection to an external MCP tool server.


# ===========================================================================
# MCP server references
# ===========================================================================
# Each entry configures a connection to an MCP-compatible tool server.
# The agent discovers available tools from the server at runtime.

notifications_server = MCPServerRef(  # <- Example: a notifications MCP server that provides send_email, send_slack, etc.
    name="notifications",  # <- Human-readable name for this server.
    url="http://localhost:8100/mcp",  # <- MCP server endpoint. Replace with your actual server URL.
    headers={"Authorization": "Bearer ${NOTIFICATIONS_API_KEY}"},  # <- Auth headers. Use environment variable references for secrets.
    timeout_s=10.0,  # <- Timeout for MCP calls. Set appropriately for network latency.
    prefix_tools=True,  # <- Prefix tool names with the server name to avoid collisions (e.g., "notifications_send_email").
    tool_name_prefix="notify",  # <- Custom prefix instead of using the server name.
)

calendar_server = MCPServerRef(  # <- Example: a calendar MCP server for scheduling and reminders.
    name="calendar",
    url="http://localhost:8101/mcp",
    headers={},
    timeout_s=15.0,
    prefix_tools=True,
    tool_name_prefix="cal",
)


# Collect all MCP servers for easy import
mcp_servers = [  # <- This list is imported by agents.py and passed to the coordinator agent.
    notifications_server,
    calendar_server,
]
