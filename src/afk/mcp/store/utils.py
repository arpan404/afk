"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Utility helpers for MCP store reference parsing and schema normalization.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.parse
from typing import Any

from afk.mcp.store.types import (
    MCPRemoteProtocolError,
    MCPRemoteTool,
    MCPServerRef,
    MCPServerResolutionError,
)


def _sanitize_name(value: str) -> str:
    out = re.sub(r"[^a-zA-Z0-9_]+", "_", value)
    out = out.strip("_")
    return out or "mcp"


def _looks_private_or_local_host(host: str) -> bool:
    normalized = host.lower().strip()
    if normalized in {"localhost", "127.0.0.1", "::1", "[::1]"}:
        return True
    try:
        ip = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return _is_restricted_ip(ip)


def _is_restricted_ip(ip: ipaddress._BaseAddress) -> bool:
    return bool(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
    )


def _validate_resolved_ips(*, host: str, allow_private_networks: bool) -> None:
    try:
        addrs = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise MCPServerResolutionError(f"Unable to resolve MCP host '{host}': {exc}") from exc

    ips = {sockaddr[0] for *_rest, sockaddr in addrs if isinstance(sockaddr, tuple)}
    if not ips:
        raise MCPServerResolutionError(f"No resolvable addresses for MCP host '{host}'")

    if allow_private_networks:
        return

    for candidate in ips:
        try:
            ip = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if _is_restricted_ip(ip):
            raise MCPServerResolutionError(
                f"MCP host '{host}' resolves to restricted address '{candidate}'"
            )


def _validate_remote_url(
    url: str,
    *,
    parsed: urllib.parse.SplitResult | None = None,
    allow_private_networks: bool = False,
    require_https: bool = True,
    allowed_hostnames: tuple[str, ...] = (),
) -> str:
    parsed_url = parsed or urllib.parse.urlparse(url)

    if parsed_url.scheme not in {"http", "https"}:
        raise MCPServerResolutionError("MCP server URL scheme must be http or https")
    if not parsed_url.netloc:
        raise MCPServerResolutionError("MCP server URL must include network location")

    if require_https and parsed_url.scheme != "https" and not _looks_private_or_local_host(
        parsed_url.hostname or ""
    ):
        raise MCPServerResolutionError(
            "MCP server URL must use HTTPS for remote references"
        )

    if parsed_url.hostname is None:
        raise MCPServerResolutionError("MCP server URL must include a valid hostname")

    hostname = parsed_url.hostname.lower()
    allowed = tuple(name.lower() for name in allowed_hostnames if isinstance(name, str))
    if allowed and hostname not in allowed:
        raise MCPServerResolutionError(
            f"MCP server hostname '{hostname}' is not in allowed_hostnames"
        )

    if _looks_private_or_local_host(hostname) and not allow_private_networks:
        raise MCPServerResolutionError(
            "MCP server URL points to a restricted private/loopback host"
        )

    _validate_resolved_ips(host=hostname, allow_private_networks=allow_private_networks)
    return url


def _validate_http_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return _validate_remote_url(
        url,
        parsed=parsed,
        allow_private_networks=True,
        require_https=False,
        allowed_hostnames=(),
    )


def _qualified_tool_name(server: MCPServerRef, tool_name: str) -> str:
    if not server.prefix_tools:
        return tool_name
    prefix = server.tool_name_prefix or server.name
    safe_prefix = _sanitize_name(prefix)
    safe_tool = _sanitize_name(tool_name)
    return f"{safe_prefix}__{safe_tool}"


def _extract_mcp_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    chunks: list[str] = []
    for row in content:
        if not isinstance(row, dict):
            continue
        text = row.get("text")
        if isinstance(text, str):
            chunks.append(text)
    if not chunks:
        return None
    return "\n".join(chunks)


def resolve_server_ref(ref: str | dict[str, Any] | MCPServerRef) -> MCPServerRef:
    """Resolve a server reference from string/dict/dataclass form."""
    if isinstance(ref, MCPServerRef):
        _validate_remote_url(
            ref.url,
            allow_private_networks=ref.allow_private_networks,
            require_https=ref.require_https,
            allowed_hostnames=ref.allowed_hostnames,
        )
        return ref

    if isinstance(ref, str):
        parsed = ref.strip()
        if not parsed:
            raise MCPServerResolutionError("MCP server reference cannot be empty")

        if "=" in parsed:
            name, url = parsed.split("=", 1)
            normalized_url = _validate_http_url(url.strip())
            return MCPServerRef(name=name.strip(), url=normalized_url)

        if not parsed.startswith(("http://", "https://")):
            raise MCPServerResolutionError(
                "String MCP server refs must be 'name=url' or an http(s) URL"
            )
        normalized_url = _validate_http_url(parsed)
        url_parts = urllib.parse.urlparse(normalized_url)
        base = url_parts.netloc or "mcp_server"
        derived_name = _sanitize_name(base.replace(":", "_"))
        return MCPServerRef(name=derived_name, url=normalized_url)

    if isinstance(ref, dict):
        url = ref.get("url")
        if not isinstance(url, str) or not url.strip():
            raise MCPServerResolutionError(
                "MCP server dict ref requires non-empty 'url'"
            )
        normalized_url = _validate_http_url(url.strip())

        name_raw = ref.get("name")
        if isinstance(name_raw, str) and name_raw.strip():
            name = name_raw.strip()
        else:
            netloc = urllib.parse.urlparse(normalized_url).netloc or "mcp_server"
            name = _sanitize_name(netloc.replace(":", "_"))

        headers = ref.get("headers") or {}
        if not isinstance(headers, dict):
            raise MCPServerResolutionError("MCP server 'headers' must be a dict")

        timeout_s = ref.get("timeout_s", 20.0)
        if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
            raise MCPServerResolutionError("MCP server 'timeout_s' must be > 0")

        prefix_tools = bool(ref.get("prefix_tools", True))
        tool_name_prefix = ref.get("tool_name_prefix")
        if tool_name_prefix is not None and not isinstance(tool_name_prefix, str):
            raise MCPServerResolutionError(
                "MCP server 'tool_name_prefix' must be a string when set"
            )

        require_https = bool(ref.get("require_https", True))
        allow_private = bool(ref.get("allow_private_networks", False))
        allowed_hostnames = tuple(
            str(v).strip() for v in ref.get("allowed_hostnames", ()) or ()
            if str(v).strip()
        )

        return MCPServerRef(
            name=name,
            url=normalized_url,
            headers={str(k): str(v) for k, v in headers.items()},
            timeout_s=float(timeout_s),
            prefix_tools=prefix_tools,
            tool_name_prefix=tool_name_prefix.strip()
            if isinstance(tool_name_prefix, str) and tool_name_prefix.strip()
            else None,
            require_https=require_https,
            allow_private_networks=allow_private,
            allowed_hostnames=allowed_hostnames,
        )


def normalize_remote_tools(
    server: MCPServerRef,
    tools: Any,
) -> list[MCPRemoteTool]:
    """Validate and normalize a remote ``tools/list`` payload."""
    if not isinstance(tools, list):
        raise MCPServerResolutionError(
            f"Invalid tools/list response from '{server.name}': missing tools list"
        )

    normalized: list[MCPRemoteTool] = []
    for row in tools:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        description = row.get("description")
        schema = row.get("inputSchema")
        normalized.append(
            MCPRemoteTool(
                server_name=server.name,
                name=name,
                qualified_name=_qualified_tool_name(server, name),
                description=(description if isinstance(description, str) else name),
                input_schema=(
                    schema if isinstance(schema, dict) else {"type": "object"}
                ),
            )
        )
    return normalized


def normalize_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure a safe object schema shape for model-facing tool definitions."""
    if not isinstance(schema, dict):
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

    out = dict(schema)
    out["type"] = "object"

    properties_raw = out.get("properties")
    if not isinstance(properties_raw, dict):
        properties: dict[str, Any] = {}
    else:
        properties = {
            str(key): (value if isinstance(value, dict) else {})
            for key, value in properties_raw.items()
        }
    out["properties"] = properties

    required_raw = out.get("required")
    if isinstance(required_raw, list):
        required = [
            str(name)
            for name in required_raw
            if isinstance(name, str) and name in properties
        ]
    else:
        required = []
    out["required"] = required

    additional_properties = out.get("additionalProperties")
    if not isinstance(additional_properties, (bool, dict)):
        out["additionalProperties"] = False

    return out
