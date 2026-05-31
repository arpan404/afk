from __future__ import annotations

from pydantic import BaseModel
import pytest

from fastapi.testclient import TestClient

from afk.mcp import MCPServer
from afk.mcp.server.runtime import MCPServerConfig
from afk.tools import ToolRegistry, tool


class EchoArgs(BaseModel):
    value: str


def test_mcp_endpoint_returns_204_for_jsonrpc_notification():
    registry = ToolRegistry()
    server = MCPServer(
        registry,
        config=MCPServerConfig(insecure_development=True),
    )
    client = TestClient(server.app)

    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "method": "ping",
            "params": {},
        },
    )

    assert response.status_code == 204
    assert response.text == ""


@tool(args_model=EchoArgs, name="echo")
def echo_tool(value: str) -> str:
    return value


def test_tools_call_requires_token_by_default():
    registry = ToolRegistry()
    registry.register(echo_tool)
    server = MCPServer(
        registry,
        config=MCPServerConfig(
            require_tools_auth=True,
            mcp_tools_auth_token="token-123",
        ),
    )
    client = TestClient(server.app)

    response = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": "auth-missing",
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"value": "hello"}},
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["error"]["code"] == -32001


def test_tools_call_accepts_bearer_token():
    registry = ToolRegistry()
    registry.register(echo_tool)
    server = MCPServer(
        registry,
        config=MCPServerConfig(
            require_tools_auth=True,
            mcp_tools_auth_token="token-123",
        ),
    )
    client = TestClient(server.app)

    response = client.post(
        "/mcp",
        headers={"Authorization": "Bearer token-123"},
        json={
            "jsonrpc": "2.0",
            "id": "auth-ok",
            "method": "tools/call",
            "params": {"name": "echo", "arguments": {"value": "hello"}},
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["result"]["content"][0]["text"] == "hello"


def test_wildcard_cors_not_allowed_with_credentials():
    with pytest.raises(ValueError, match="wildcard origin '\\*'"):
        MCPServerConfig(cors_origins=["*"], allow_credentials=True)
