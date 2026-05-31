"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Transport layer for remote MCP JSON-RPC calls.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from typing import Any

from afk.mcp.store.types import MCPRemoteCallError, MCPRemoteProtocolError, MCPServerRef
from afk.mcp.store.utils import _validate_remote_url


def _validate_server_for_call(server: MCPServerRef) -> None:
    _validate_remote_url(
        server.url,
        allow_private_networks=server.allow_private_networks,
        require_https=server.require_https,
        allowed_hostnames=server.allowed_hostnames,
    )


class MCPJsonRpcClient:
    """HTTP JSON-RPC client used by ``MCPStore``."""

    async def call(
        self,
        server: MCPServerRef,
        *,
        method: str,
        params: dict[str, Any],
        post: Callable[[MCPServerRef, bytes], bytes] | None = None,
    ) -> dict[str, Any]:
        request_body = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": method,
            "params": params,
        }
        payload = json.dumps(request_body).encode("utf-8")
        _validate_server_for_call(server)
        post_fn = post or self.http_post
        response_bytes = await asyncio.to_thread(post_fn, server, payload)
        try:
            decoded = json.loads(response_bytes.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            raise MCPRemoteProtocolError(
                f"Invalid JSON response from MCP server '{server.name}'"
            ) from e

        if not isinstance(decoded, dict):
            raise MCPRemoteProtocolError(
                f"Invalid JSON-RPC envelope from MCP server '{server.name}'"
            )

        err = decoded.get("error")
        if isinstance(err, dict):
            message = err.get("message") if isinstance(err.get("message"), str) else ""
            raise MCPRemoteCallError(
                f"MCP server '{server.name}' method '{method}' failed: {message or err}"
            )

        result = decoded.get("result")
        if not isinstance(result, dict):
            raise MCPRemoteProtocolError(
                f"Missing JSON-RPC result object from MCP server '{server.name}'"
            )
        return result

    def http_post(self, server: MCPServerRef, payload: bytes) -> bytes:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **server.headers,
        }
        req = urllib.request.Request(
            server.url,
            data=payload,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=server.timeout_s) as resp:  # noqa: S310
                return resp.read()
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                body = ""
            raise MCPRemoteCallError(
                f"HTTP {e.code} calling MCP server '{server.name}': {body or e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise MCPRemoteCallError(
                f"Network error calling MCP server '{server.name}': {e.reason}"
            ) from e
