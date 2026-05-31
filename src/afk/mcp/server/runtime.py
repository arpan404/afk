"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

MCP (Model Context Protocol) server built on FastAPI.

Exposes tools from a ``ToolRegistry`` as MCP-compatible endpoints,
following the JSON-RPC 2.0 wire format specified by the MCP standard.

Supports:
- ``initialize`` — server capability handshake
- ``tools/list`` — discover available tools
- ``tools/call`` — execute a tool
- SSE transport for streaming JSON-RPC responses
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

try:
    from fastapi import Request as FastAPIRequest
    from fastapi.responses import Response as FastAPIResponse
except Exception:  # pragma: no cover
    FastAPIRequest = Any  # type: ignore[assignment]
    FastAPIResponse = Any  # type: ignore[assignment]

from afk.mcp.server.protocol import (
    INVALID_REQUEST,
    PARSE_ERROR,
    MCPProtocolHandler,
    jsonrpc_error,
)
from afk.tools import ToolRegistry

# ---------------------------------------------------------------------------
# MCP server configuration
# ---------------------------------------------------------------------------


@dataclass
class MCPServerConfig:
    """
    Configuration for the MCP server.

    Attributes:
        name: Server name advertised during ``initialize``.
        version: Server version string.
        host: Bind host for the FastAPI server.
        port: Bind port for the FastAPI server.
        instructions: Optional instructions describing the server's purpose.
        cors_origins: List of allowed CORS origins.
        mcp_path: JSON-RPC endpoint path.
        sse_path: SSE endpoint path.
        health_path: Health endpoint path.
        enable_sse: Whether to expose SSE endpoint.
        enable_health: Whether to expose health endpoint.
        allow_batch_requests: Whether JSON-RPC batch requests are accepted.
    """

    name: str = "afk-mcp-server"
    version: str = "1.0.0"
    host: str = "127.0.0.1"
    port: int = 8000
    instructions: str | None = None
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = False
    mcp_path: str = "/mcp"
    sse_path: str = "/mcp/sse"
    health_path: str = "/health"
    enable_sse: bool = True
    enable_health: bool = True
    allow_batch_requests: bool = True
    require_tools_auth: bool = True
    insecure_development: bool = False
    mcp_tools_auth_token: str | None = None
    token_header: str = "Authorization"

    def __post_init__(self) -> None:
        if self.insecure_development:
            return
        if self.require_tools_auth and self.mcp_tools_auth_token is None:
            raise ValueError(
                "MCP server requires an auth token via mcp_tools_auth_token unless"
                " insecure_development=True"
            )
        if self.allow_credentials and "*" in self.cors_origins:
            raise ValueError(
                "CORS wildcard origin '*' cannot be used with credentials enabled"
            )


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------


class MCPServer:
    """
    MCP server that exposes ``ToolRegistry`` tools via FastAPI.

    Implements the Model Context Protocol over HTTP with JSON-RPC 2.0
    message format. Tools are automatically discovered from the registry.

    Usage::

        from afk.tools import ToolRegistry, tool
        from afk.mcp import MCPServer

        registry = ToolRegistry()

        @tool(name="greet", description="Greet someone")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        registry.register(greet)

        server = MCPServer(registry)
        server.run()  # starts FastAPI on port 8000

    Endpoints:
        ``POST /mcp`` — JSON-RPC 2.0 endpoint for ``initialize``, ``tools/list``, ``tools/call``
        ``GET /mcp/sse`` — SSE transport (Server-Sent Events)
        ``GET /health`` — Health check

    Args:
        registry: ``ToolRegistry`` containing tools to expose.
        config: Server configuration.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        config: MCPServerConfig | None = None,
        app: Any | None = None,
    ) -> None:
        self._registry = registry
        self._config = config or MCPServerConfig()
        self._protocol_handler = self._create_protocol_handler()
        self._app = app or self._create_app()
        if app is not None:
            self.mount(app)

    @classmethod
    def from_tools(
        cls,
        tools: Iterable[Any],
        *,
        config: MCPServerConfig | None = None,
        app: Any | None = None,
    ) -> MCPServer:
        """
        Build an MCP server from an iterable of AFK tools.

        This is a DX-focused convenience for simple setups where callers have
        tools but not an explicit ``ToolRegistry`` instance.
        """
        registry = ToolRegistry()
        registry.register_many(tools)
        return cls(registry, config=config, app=app)

    @property
    def app(self):
        """
        The FastAPI application instance.

        Use this to mount the MCP server into an existing app or for testing::

            from fastapi.testclient import TestClient
            client = TestClient(server.app)
        """
        return self._app

    @property
    def config(self) -> MCPServerConfig:
        """Server configuration."""
        return self._config

    def _create_protocol_handler(self) -> MCPProtocolHandler:
        return MCPProtocolHandler(
            registry=self._registry,
            server_name=self._config.name,
            server_version=self._config.version,
            instructions=self._config.instructions,
            require_tools_auth=self._config.require_tools_auth,
            insecure_development=self._config.insecure_development,
            auth_token=self._config.mcp_tools_auth_token,
            token_header=self._config.token_header,
        )

    def _create_router(self):
        """Build an APIRouter containing MCP routes."""
        try:
            from fastapi import APIRouter
            from fastapi.responses import JSONResponse, StreamingResponse
        except ImportError:
            raise ImportError(
                "FastAPI is required for MCPServer. "
                "Install it with: pip install fastapi uvicorn"
            )

        router = APIRouter()

        if self._config.enable_health:

            @router.get(self._config.health_path)
            async def health():
                return {
                    "status": "ok",
                    "server": self._config.name,
                    "version": self._config.version,
                    "tools_count": len(self._registry.names()),
                }

        @router.post(self._config.mcp_path)
        async def mcp_endpoint(request: FastAPIRequest):
            """Main JSON-RPC 2.0 endpoint for MCP."""
            try:
                body = await request.json()
            except Exception:
                return JSONResponse(
                    jsonrpc_error(None, PARSE_ERROR, "Parse error"),
                    status_code=200,
                )

            if isinstance(body, list):
                if not self._config.allow_batch_requests:
                    return JSONResponse(
                        jsonrpc_error(None, INVALID_REQUEST, "Batch requests disabled"),
                        status_code=200,
                    )
                responses = []
                for item in body:
                    resp = await self._protocol_handler.handle_message(
                        item,
                        headers=dict(request.headers),
                    )
                    if resp is not None:
                        responses.append(resp)
                return JSONResponse(responses, status_code=200)

            result = await self._protocol_handler.handle_message(
                body,
                headers=dict(request.headers),
            )
            if result is None:
                return FastAPIResponse(status_code=204)
            return JSONResponse(result, status_code=200)

        if self._config.enable_sse:

            @router.get(self._config.sse_path)
            async def sse_endpoint(request: FastAPIRequest):
                """SSE transport for MCP — sends JSON-RPC messages as events."""
                _ = request
                import asyncio

                session_id = uuid.uuid4().hex

                async def event_stream():
                    endpoint_data = json.dumps(
                        {
                            "endpoint": self._config.mcp_path,
                            "sessionId": session_id,
                        }
                    )
                    yield f"event: endpoint\ndata: {endpoint_data}\n\n"

                    try:
                        while True:
                            await asyncio.sleep(30)
                            yield ": heartbeat\n\n"
                    except asyncio.CancelledError:
                        pass

                return StreamingResponse(
                    event_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

        return router

    def _create_app(self):
        """Build the FastAPI application with MCP routes."""
        try:
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware
        except ImportError:
            raise ImportError(
                "FastAPI is required for MCPServer. "
                "Install it with: pip install fastapi uvicorn"
            )

        app = FastAPI(
            title=self._config.name,
            version=self._config.version,
            description="AFK MCP Server — Model Context Protocol",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=self._config.cors_origins,
            allow_credentials=self._config.allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.include_router(self._create_router())
        return app

    def mount(self, app: Any) -> Any:
        """
        Mount MCP routes into an existing FastAPI app.

        Returns the provided app for fluent usage.
        """
        app.include_router(self._create_router())
        return app

    def run(self, **kwargs: Any) -> None:
        """
        Start the MCP server using uvicorn.

        Args:
            **kwargs: Additional arguments passed to ``uvicorn.run()``.
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError(
                "uvicorn is required to run MCPServer. "
                "Install it with: pip install uvicorn"
            )

        uvicorn.run(
            self._app,
            host=kwargs.pop("host", self._config.host),
            port=kwargs.pop("port", self._config.port),
            **kwargs,
        )


def create_mcp_server(
    *,
    registry: ToolRegistry | None = None,
    tools: Iterable[Any] | None = None,
    config: MCPServerConfig | None = None,
    app: Any | None = None,
) -> MCPServer:
    """
    DX-first constructor for MCP servers.

    Callers can pass either an existing registry or a list of tools.
    """
    if registry is not None and tools is not None:
        raise ValueError("Pass either 'registry' or 'tools', not both")
    if registry is not None:
        return MCPServer(registry, config=config, app=app)
    return MCPServer.from_tools(tools or [], config=config, app=app)
