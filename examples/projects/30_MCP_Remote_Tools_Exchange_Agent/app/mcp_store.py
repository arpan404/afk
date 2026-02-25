"""Fake MCP store transport for local deterministic execution."""

import types

from afk.mcp import MCPStore


def build_fake_store() -> MCPStore:
    store = MCPStore()

    async def fake_call(self, server, *, method: str, params: dict, post=None):
        _ = self
        _ = server
        _ = post
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
            args = params.get("arguments", {})
            total = int(args.get("a", 0)) + int(args.get("b", 0))
            return {
                "content": [{"type": "text", "text": str(total)}],
                "isError": False,
            }
        raise RuntimeError(f"unexpected MCP method: {method}")

    store._client.call = types.MethodType(fake_call, store._client)
    return store
