"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Protocol-first messaging exports.

Hard break:
Legacy mailbox/bus APIs were removed. Use A2A protocol contracts.
"""

from ..agents.a2a import (
    A2AAuthContext,
    A2AAuthError,
    A2AAuthorizationDecision,
    A2AAuthorizationError,
    A2AAuthProvider,
    A2ADeliveryStore,
    A2APrincipal,
    A2AServiceHost,
    A2AServiceHostError,
    AllowAllA2AAuthProvider,
    APIKeyA2AAuthProvider,
    GoogleA2AAdapterError,
    GoogleA2AProtocolAdapter,
    InMemoryA2ADeliveryStore,
    InternalA2AEnvelope,
    InternalA2AProtocol,
    JWTA2AAuthProvider,
    RedisA2ADeliveryStore,
)
from ..agents.contracts import (
    AgentCommunicationProtocol,
    AgentDeadLetter,
    AgentInvocationRequest,
    AgentInvocationResponse,
    AgentProtocolEvent,
)

__all__ = [
    "AgentCommunicationProtocol",
    "AgentInvocationRequest",
    "AgentInvocationResponse",
    "AgentProtocolEvent",
    "AgentDeadLetter",
    "InternalA2AEnvelope",
    "InternalA2AProtocol",
    "GoogleA2AProtocolAdapter",
    "GoogleA2AAdapterError",
    "A2AServiceHost",
    "A2AServiceHostError",
    "A2AAuthContext",
    "A2APrincipal",
    "A2AAuthorizationDecision",
    "A2AAuthProvider",
    "A2AAuthError",
    "A2AAuthorizationError",
    "AllowAllA2AAuthProvider",
    "APIKeyA2AAuthProvider",
    "JWTA2AAuthProvider",
    "A2ADeliveryStore",
    "InMemoryA2ADeliveryStore",
    "RedisA2ADeliveryStore",
]


def __getattr__(name: str):
    legacy = {
        "AgentMessage",
        "AgentMailbox",
        "MessageBus",
        "MessagePriority",
        "InMemoryMessageBus",
        "RedisMessageBus",
    }
    if name in legacy:
        raise AttributeError(
            f"{name} was removed from afk.messaging. "
            "Use protocol-first A2A APIs from afk.messaging / afk.agents."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
