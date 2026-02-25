"""A2A service host construction helpers."""

from afk.agents import (
    A2AServiceHost,
    APIKeyA2AAuthProvider,
    InternalA2AProtocol,
)
from afk.agents.contracts import AgentInvocationRequest, AgentInvocationResponse


async def dispatch(request: AgentInvocationRequest) -> AgentInvocationResponse:
    return AgentInvocationResponse(
        run_id=request.run_id,
        thread_id=request.thread_id,
        conversation_id=request.conversation_id,
        correlation_id=request.correlation_id,
        idempotency_key=request.idempotency_key,
        source_agent=request.target_agent,
        target_agent=request.source_agent,
        success=True,
        output={"status": "accepted", "assigned_team": "sre"},
    )


def build_service_host() -> A2AServiceHost:
    provider = APIKeyA2AAuthProvider(
        key_to_subject={"prod-key": "svc-control-plane"},
        key_to_roles={"prod-key": ("a2a:all",)},
    )
    return A2AServiceHost(
        protocol=InternalA2AProtocol(dispatch=dispatch),
        auth_provider=provider,
        production_mode=True,
    )
