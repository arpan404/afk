from __future__ import annotations

import asyncio

import pytest

from afk.agents.a2a import (
    A2AAuthContext,
    A2AServiceHost,
    A2AServiceHostError,
    AllowAllA2AAuthProvider,
    APIKeyA2AAuthProvider,
    InternalA2AProtocol,
)
from afk.agents.contracts import AgentInvocationRequest, AgentInvocationResponse


def run_async(coro):
    return asyncio.run(coro)


async def _ok_dispatch(request: AgentInvocationRequest) -> AgentInvocationResponse:
    return AgentInvocationResponse(
        run_id=request.run_id,
        thread_id=request.thread_id,
        conversation_id=request.conversation_id,
        correlation_id=request.correlation_id,
        idempotency_key=request.idempotency_key,
        source_agent=request.target_agent,
        target_agent=request.source_agent,
        success=True,
        output={"ok": True},
    )


def test_service_host_rejects_allow_all_in_production_mode():
    with pytest.raises(A2AServiceHostError):
        A2AServiceHost(
            protocol=InternalA2AProtocol(dispatch=_ok_dispatch),
            auth_provider=AllowAllA2AAuthProvider(),
            production_mode=True,
        )


def test_api_key_auth_provider_enforces_role_authorization():
    provider = APIKeyA2AAuthProvider(
        key_to_subject={"k1": "svc-1"},
        key_to_roles={"k1": ("a2a:invoke",)},
    )

    async def scenario():
        principal = await provider.authenticate(
            A2AAuthContext(headers={"x-api-key": "k1"})
        )
        decision_ok = await provider.authorize(
            principal,
            action="invoke",
            resource="a2a/invoke",
        )
        decision_no = await provider.authorize(
            principal,
            action="cancel_task",
            resource="a2a/tasks/1/cancel",
        )
        return decision_ok, decision_no

    ok, denied = run_async(scenario())
    assert ok.allowed is True
    assert denied.allowed is False


def test_service_host_endpoints_authorize_and_invoke():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    provider = APIKeyA2AAuthProvider(
        key_to_subject={"prod-key": "svc-1"},
        key_to_roles={"prod-key": ("a2a:all",)},
    )
    host = A2AServiceHost(
        protocol=InternalA2AProtocol(dispatch=_ok_dispatch),
        auth_provider=provider,
        production_mode=True,
    )
    app = host.create_app()

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        card = client.get("/.well-known/agent-card")
        assert card.status_code == 200

        payload = {
            "run_id": "r1",
            "thread_id": "t1",
            "conversation_id": "c1",
            "correlation_id": "corr1",
            "idempotency_key": "idem1",
            "source_agent": "parent",
            "target_agent": "child",
            "payload": {"k": "v"},
            "metadata": {},
            "causation_id": "cause1",
            "timeout_s": 1.0,
        }

        unauthorized = client.post("/a2a/invoke", json=payload)
        assert unauthorized.status_code == 401

        authorized = client.post(
            "/a2a/invoke",
            json=payload,
            headers={"x-api-key": "prod-key"},
        )
        assert authorized.status_code == 200
        body = authorized.json()
        assert body["success"] is True


def test_service_host_production_hides_docs_and_sanitizes_errors():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    provider = APIKeyA2AAuthProvider(
        key_to_subject={"prod-key": "svc-1"},
        key_to_roles={"prod-key": ("a2a:all",)},
    )
    host = A2AServiceHost(
        protocol=InternalA2AProtocol(dispatch=_ok_dispatch),
        auth_provider=provider,
        production_mode=True,
    )
    app = host.create_app()

    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404

        unauthorized = client.post("/a2a/invoke", json={})
        assert unauthorized.status_code == 401
        assert unauthorized.json()["detail"] == "Authentication failed"

        invalid = client.post(
            "/a2a/invoke",
            json={},
            headers={"x-api-key": "prod-key"},
        )
        assert invalid.status_code == 422
        assert invalid.json()["detail"] == "Invalid invoke payload"


def test_service_host_stream_endpoint_returns_ndjson():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")

    provider = APIKeyA2AAuthProvider(
        key_to_subject={"prod-key": "svc-1"},
        key_to_roles={"prod-key": ("a2a:all",)},
    )
    host = A2AServiceHost(
        protocol=InternalA2AProtocol(dispatch=_ok_dispatch),
        auth_provider=provider,
        production_mode=True,
    )
    app = host.create_app()

    from fastapi.testclient import TestClient

    payload = {
        "run_id": "r1",
        "thread_id": "t1",
        "conversation_id": "c1",
        "correlation_id": "corr1",
        "idempotency_key": "idem1",
        "source_agent": "parent",
        "target_agent": "child",
        "payload": {"k": "v"},
        "metadata": {},
        "causation_id": "cause1",
        "timeout_s": 1.0,
    }

    with TestClient(app) as client:
        response = client.post(
            "/a2a/invoke/stream",
            json=payload,
            headers={"x-api-key": "prod-key"},
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        assert response.text.strip()
        assert all(line.startswith("{") for line in response.text.splitlines())
