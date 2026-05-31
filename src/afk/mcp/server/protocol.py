"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Protocol-layer helpers for MCP JSON-RPC request handling.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from afk.tools import ToolContext, ToolRegistry

logger = logging.getLogger("afk.mcp")

MCP_PROTOCOL_VERSION = "2026-02-20"

# Standard JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNAUTHORIZED = -32001


def jsonrpc_response(id: Any, result: Any) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 success response."""
    return {"jsonrpc": "2.0", "id": id, "result": result}


def jsonrpc_error(
    id: Any,
    code: int,
    message: str,
    data: Any = None,
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error response."""
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": id, "error": error}


class MCPProtocolHandler:
    """
    Handles MCP methods and JSON-RPC envelope validation.

    This class keeps protocol and tool-execution behavior independent from
    HTTP routing concerns so it can be reused by different transports.
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        server_name: str,
        server_version: str,
        instructions: str | None = None,
        require_tools_auth: bool = True,
        insecure_development: bool = False,
        auth_token: str | None = None,
        token_header: str = "Authorization",
    ) -> None:
        self._registry = registry
        self._server_name = server_name
        self._server_version = server_version
        self._instructions = instructions
        self._require_tools_auth = bool(require_tools_auth)
        self._insecure_development = bool(insecure_development)
        self._auth_token = auth_token
        self._token_header = token_header

    async def handle_message(
        self,
        message: dict[str, Any],
        *,
        headers: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Route one JSON-RPC 2.0 message to the appropriate MCP method."""
        if not isinstance(message, dict):
            return jsonrpc_error(None, INVALID_REQUEST, "Invalid Request")

        jsonrpc = message.get("jsonrpc")
        if jsonrpc != "2.0":
            return jsonrpc_error(
                message.get("id"),
                INVALID_REQUEST,
                "Invalid JSON-RPC version",
            )

        method = message.get("method")
        params = message.get("params", {})
        msg_id = message.get("id")

        if not method:
            return jsonrpc_error(msg_id, INVALID_REQUEST, "Missing method")

        is_notification = msg_id is None

        try:
            if method == "initialize":
                result = self.handle_initialize(params)
            elif method == "tools/list":
                result = self.handle_tools_list(params)
            elif method == "tools/call":
                self._require_tools_auth_if_configured(method=method, headers=headers)
                result = await self.handle_tools_call(params)
            elif method == "ping" or method == "notifications/initialized":
                result = {}
            else:
                return jsonrpc_error(
                    msg_id,
                    METHOD_NOT_FOUND,
                    f"Method not found: {method}",
                )

            if is_notification:
                return None
            return jsonrpc_response(msg_id, result)
        except PermissionError as exc:
            return jsonrpc_error(msg_id, UNAUTHORIZED, str(exc))
        except Exception as exc:
            logger.exception("Error handling MCP method %s", method)
            return jsonrpc_error(msg_id, INTERNAL_ERROR, str(exc))

    def _require_tools_auth_if_configured(
        self,
        *,
        method: str,
        headers: dict[str, Any] | None,
    ) -> None:
        if self._insecure_development:
            return
        if not self._require_tools_auth:
            return
        if method != "tools/call":
            return

        if self._auth_token is None:
            raise PermissionError("MCP auth token is not configured")

        supplied = None
        if headers is not None:
            for key, value in headers.items():
                if key.lower() == self._token_header.lower():
                    supplied = str(value)
                    break
        if supplied is None:
            raise PermissionError("Missing MCP auth token")

        expected = f"Bearer {self._auth_token}".strip()
        if supplied.strip() == self._auth_token:
            return
        if supplied.strip() == expected:
            return
        raise PermissionError("Invalid MCP auth token")

    def handle_initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle ``initialize`` and return server capabilities."""
        _ = params
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {
                    "listChanged": False,
                },
            },
            "serverInfo": {
                "name": self._server_name,
                "version": self._server_version,
            },
            **({"instructions": self._instructions} if self._instructions else {}),
        }

    def handle_tools_list(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle ``tools/list`` and return MCP tool schemas."""
        _ = params
        tools = []
        for tool_obj in self._registry.list():
            spec = tool_obj.spec
            tools.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "inputSchema": {
                        "type": "object",
                        **(spec.parameters_schema or {}),
                    },
                }
            )
        return {"tools": tools}

    async def handle_tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        """Handle ``tools/call`` and return MCP content result."""
        tool_name = params.get("name")
        if not tool_name:
            raise ValueError("Missing 'name' in tools/call params")

        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("'arguments' must be an object")

        ctx = ToolContext(
            request_id=uuid.uuid4().hex,
            metadata={"source": "mcp", "tool_name": tool_name},
        )
        result = await self._registry.call(
            tool_name,
            arguments,
            ctx=ctx,
        )
        return {
            "content": self._result_content(
                result.output, result.success, result.error_message
            ),
            "isError": not result.success,
        }

    def _result_content(
        self,
        output: Any,
        success: bool,
        error_message: str | None,
    ) -> list[dict[str, Any]]:
        if not success:
            return [
                {
                    "type": "text",
                    "text": error_message or "Tool execution failed",
                }
            ]

        if isinstance(output, str):
            return [{"type": "text", "text": output}]
        if output is None:
            return [{"type": "text", "text": ""}]
        return [
            {
                "type": "text",
                "text": json.dumps(output, default=str),
            }
        ]
