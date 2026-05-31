from __future__ import annotations

import asyncio
import types

import pytest

from afk.mcp import MCPStore
from afk.mcp.store import MCPServerResolutionError, normalize_json_schema
from afk.tools import ToolRegistry


def run_async(coro):
    return asyncio.run(coro)


def test_mcp_store_materializes_remote_tools_and_executes_call():
    store = MCPStore()
    seen_methods: list[str] = []

    async def fake_call(self, server, *, method: str, params: dict, post=None):
        _ = self
        _ = server
        _ = post
        seen_methods.append(method)
        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "add",
                        "description": "Add two integers",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    }
                ]
            }
        if method == "tools/call":
            assert params["name"] == "add"
            assert params["arguments"] == {"a": 1, "b": 2}
            return {
                "content": [{"type": "text", "text": "3"}],
                "isError": False,
            }
        raise AssertionError(f"Unexpected method: {method}")

    store._client.call = types.MethodType(fake_call, store._client)

    tools = run_async(store.tools_from_servers(["calc=https://fake.example/mcp"]))
    assert [tool.spec.name for tool in tools] == ["calc__add"]

    registry = ToolRegistry()
    registry.register_many(tools)
    result = run_async(registry.call("calc__add", {"a": 1, "b": 2}))

    assert result.success is True
    assert result.output == "3"
    assert seen_methods.count("tools/list") == 1
    assert seen_methods.count("tools/call") == 1


def test_mcp_store_list_tools_uses_cache_until_refresh():
    store = MCPStore()
    tool_list_calls = 0

    async def fake_call(self, server, *, method: str, params: dict, post=None):
        _ = self
        _ = server
        _ = params
        _ = post
        nonlocal tool_list_calls
        if method == "tools/list":
            tool_list_calls += 1
            return {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]}
        raise AssertionError(f"Unexpected method: {method}")

    store._client.call = types.MethodType(fake_call, store._client)

    ref = "svc=https://fake.example/mcp"
    first = run_async(store.list_tools(ref))
    second = run_async(store.list_tools(ref))
    refreshed = run_async(store.list_tools(ref, refresh=True))

    assert [tool.name for tool in first] == ["echo"]
    assert [tool.name for tool in second] == ["echo"]
    assert [tool.name for tool in refreshed] == ["echo"]
    assert tool_list_calls == 2


def test_mcp_remote_tool_error_surfaces_as_failed_tool_result():
    store = MCPStore()

    async def fake_call(self, server, *, method: str, params: dict, post=None):
        _ = self
        _ = server
        _ = params
        _ = post
        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "explode",
                        "description": "Always fails",
                        "inputSchema": {"type": "object"},
                    }
                ]
            }
        if method == "tools/call":
            return {
                "content": [{"type": "text", "text": "remote boom"}],
                "isError": True,
            }
        raise AssertionError(f"Unexpected method: {method}")

    store._client.call = types.MethodType(fake_call, store._client)

    tools = run_async(store.tools_from_servers(["svc=https://fake.example/mcp"]))
    registry = ToolRegistry()
    registry.register_many(tools)

    result = run_async(registry.call("svc__explode", {}))

    assert result.success is False
    assert result.error_message is not None
    assert "remote boom" in result.error_message


def test_normalize_json_schema_handles_non_dict_and_defaults():
    assert normalize_json_schema("bad") == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_normalize_json_schema_coerces_invalid_fields():
    normalized = normalize_json_schema(
        {
            "type": "array",
            "properties": {
                "ok": {"type": "string"},
                "bad": None,
            },
            "required": ["ok", "missing", 42],
            "additionalProperties": "invalid",
        }
    )

    assert normalized["type"] == "object"
    assert normalized["properties"] == {
        "ok": {"type": "string"},
        "bad": {},
    }
    assert normalized["required"] == ["ok"]
    assert normalized["additionalProperties"] is False


def test_resolve_server_rejects_non_http_scheme_for_name_url_string():
    store = MCPStore()
    with pytest.raises(MCPServerResolutionError, match="http or https"):
        store.resolve_server("bad=file:///etc/passwd")


def test_resolve_server_rejects_non_http_scheme_for_dict_url():
    store = MCPStore()
    with pytest.raises(MCPServerResolutionError, match="http or https"):
        store.resolve_server({"name": "bad", "url": "ftp://example.com/mcp"})


def test_store_rejects_public_http_mcp_urls():
    store = MCPStore()
    with pytest.raises(MCPServerResolutionError, match="must use HTTPS for remote"):
        run_async(store.list_tools("svc=http://example.com/mcp"))


def test_store_rejects_private_host_mcp_urls_by_default():
    store = MCPStore()
    with pytest.raises(MCPServerResolutionError, match="restricted private"):
        run_async(store.list_tools("svc=http://127.0.0.1:8000"))
