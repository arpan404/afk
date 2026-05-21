"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

FastAPI A2A service host for exposing AFK agents over protocol endpoints.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from starlette.requests import Request

from ...llms.types import JSONValue
from ..contracts import AgentCommunicationProtocol, AgentInvocationRequest
from .auth import (
    A2AAuthContext,
    A2AAuthError,
    A2AAuthProvider,
)


class A2AServiceHostError(RuntimeError):
    """Raised for invalid A2A service host setup."""


class A2AServiceHost:
    """Expose protocol operations via FastAPI endpoints with auth enforcement."""

    def __init__(
        self,
        *,
        protocol: AgentCommunicationProtocol,
        auth_provider: A2AAuthProvider,
        service_name: str = "afk-agent-service",
        production_mode: bool = True,
    ) -> None:
        if production_mode and getattr(auth_provider, "provider_id", "") == "allow_all":
            raise A2AServiceHostError(
                "Refusing to start A2A service in production mode with allow-all auth provider"
            )

        self.protocol = protocol
        self.auth_provider = auth_provider
        self.service_name = service_name
        self.production_mode = production_mode

    def create_app(self):
        """Create and return FastAPI app exposing A2A endpoints."""
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.responses import StreamingResponse
        except Exception as exc:  # pragma: no cover - optional runtime path
            raise A2AServiceHostError(
                "FastAPI is required to host A2A service endpoints"
            ) from exc

        app = FastAPI(
            title=self.service_name,
            docs_url=None if self.production_mode else "/docs",
            redoc_url=None if self.production_mode else "/redoc",
            openapi_url=None if self.production_mode else "/openapi.json",
        )

        def _auth_detail(default: str, exc: Exception | None = None) -> str:
            if self.production_mode:
                return default
            if exc is None:
                return default
            return str(exc)

        def _validation_detail(operation: str, exc: Exception) -> str:
            if self.production_mode:
                return f"Invalid {operation} payload"
            return f"Invalid {operation} payload: {exc}"

        async def _authorize(
            request: Request,
            *,
            action: str,
            resource: str,
            context: dict[str, JSONValue] | None = None,
        ) -> None:
            headers = {str(k): str(v) for k, v in request.headers.items()}
            auth_context = A2AAuthContext(
                headers=headers, peer_id=request.client.host if request.client else None
            )
            try:
                principal = await self.auth_provider.authenticate(auth_context)
            except A2AAuthError as exc:
                raise HTTPException(
                    status_code=401,
                    detail=_auth_detail("Authentication failed", exc),
                ) from exc

            decision = await self.auth_provider.authorize(
                principal,
                action=action,
                resource=resource,
                context=context,
            )
            if not decision.allowed:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "Forbidden"
                        if self.production_mode
                        else decision.reason or "Forbidden"
                    ),
                )

        @app.get("/.well-known/agent-card")
        async def agent_card() -> dict[str, Any]:
            return {
                "name": self.service_name,
                "protocol_id": getattr(self.protocol, "protocol_id", "unknown"),
                "capabilities": ["invoke", "invoke_stream", "get_task", "cancel_task"],
                "security": {
                    "provider": getattr(self.auth_provider, "provider_id", "unknown")
                },
            }

        @app.post("/a2a/invoke")
        async def invoke(payload: dict[str, Any], request: Request) -> dict[str, Any]:
            await _authorize(
                request, action="invoke", resource="a2a/invoke", context=payload
            )
            try:
                invocation = AgentInvocationRequest(**payload)
            except Exception as exc:
                raise HTTPException(
                    status_code=422, detail=_validation_detail("invoke", exc)
                ) from exc
            response = await self.protocol.invoke(invocation)
            return asdict(response)

        @app.post("/a2a/invoke/stream")
        async def invoke_stream(
            payload: dict[str, Any], request: Request
        ) -> dict[str, Any]:
            await _authorize(
                request,
                action="invoke_stream",
                resource="a2a/invoke/stream",
                context=payload,
            )
            try:
                invocation = AgentInvocationRequest(**payload)
            except Exception as exc:
                raise HTTPException(
                    status_code=422, detail=_validation_detail("invoke", exc)
                ) from exc

            async def _iter_events():
                async for event in self.protocol.invoke_stream(invocation):
                    yield json.dumps(asdict(event), ensure_ascii=True, default=str)
                    yield "\n"

            return StreamingResponse(
                _iter_events(),
                media_type="application/x-ndjson",
            )

        @app.get("/a2a/tasks/{task_id}")
        async def get_task(task_id: str, request: Request) -> dict[str, Any]:
            await _authorize(
                request,
                action="get_task",
                resource=f"a2a/tasks/{task_id}",
            )
            fn = getattr(self.protocol, "get_task", None)
            if fn is None:
                raise HTTPException(
                    status_code=501, detail="Protocol does not support get_task"
                )
            payload = fn(task_id)
            if hasattr(payload, "__await__"):
                payload = await payload
            if isinstance(payload, dict):
                return payload
            return {"task": payload}

        @app.post("/a2a/tasks/{task_id}/cancel")
        async def cancel_task(task_id: str, request: Request) -> dict[str, Any]:
            await _authorize(
                request,
                action="cancel_task",
                resource=f"a2a/tasks/{task_id}/cancel",
            )
            fn = getattr(self.protocol, "cancel_task", None)
            if fn is None:
                raise HTTPException(
                    status_code=501, detail="Protocol does not support cancel_task"
                )
            payload = fn(task_id)
            if hasattr(payload, "__await__"):
                payload = await payload
            if isinstance(payload, dict):
                return payload
            return {"task": payload}

        return app
