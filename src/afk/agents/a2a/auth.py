"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Authentication and authorization contracts for A2A communication.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ...llms.types import JSONValue


@dataclass(frozen=True, slots=True)
class A2AAuthContext:
    """Authentication context for inbound or outbound A2A requests."""

    headers: Mapping[str, str]
    peer_id: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class A2APrincipal:
    """Authenticated principal resolved from auth context."""

    subject: str
    principal_type: str = "service"
    roles: tuple[str, ...] = ()
    claims: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class A2AAuthorizationDecision:
    """Authorization result for a requested A2A action."""

    allowed: bool
    reason: str | None = None


class A2AAuthProvider(Protocol):
    """Provider contract for A2A authentication and authorization."""

    provider_id: str

    async def authenticate(self, context: A2AAuthContext) -> A2APrincipal:
        """Authenticate a principal from request context."""
        ...

    async def authorize(
        self,
        principal: A2APrincipal,
        *,
        action: str,
        resource: str,
        context: Mapping[str, JSONValue] | None = None,
    ) -> A2AAuthorizationDecision:
        """Authorize one action for a principal."""
        ...


class A2AAuthError(PermissionError):
    """Raised when authentication fails."""


class A2AAuthorizationError(PermissionError):
    """Raised when authorization fails."""


@dataclass(frozen=True, slots=True)
class _APIKeyRecord:
    salt: bytes
    digest: bytes
    subject: str
    roles: tuple[str, ...]


class AllowAllA2AAuthProvider:
    """Development-only provider that allows every request."""

    provider_id = "allow_all"

    async def authenticate(self, context: A2AAuthContext) -> A2APrincipal:
        _ = context
        return A2APrincipal(subject="anonymous", roles=("a2a:all",))

    async def authorize(
        self,
        principal: A2APrincipal,
        *,
        action: str,
        resource: str,
        context: Mapping[str, JSONValue] | None = None,
    ) -> A2AAuthorizationDecision:
        _ = principal
        _ = action
        _ = resource
        _ = context
        return A2AAuthorizationDecision(allowed=True)


class APIKeyA2AAuthProvider:
    """API key authentication provider with role-based authorization mapping."""

    provider_id = "api_key"

    def __init__(
        self,
        *,
        key_to_subject: Mapping[str, str],
        key_to_roles: Mapping[str, tuple[str, ...]] | None = None,
        header_name: str = "x-api-key",
    ) -> None:
        self._header_name = header_name.lower()
        self._records: tuple[_APIKeyRecord, ...]

        role_map = key_to_roles or {}
        records: list[_APIKeyRecord] = []
        for key, subject in key_to_subject.items():
            salt = secrets.token_bytes(16)
            digest = self._hash_key(key, salt)
            roles = tuple(role_map.get(key, ()))
            records.append(
                _APIKeyRecord(
                    salt=salt,
                    digest=digest,
                    subject=subject,
                    roles=roles,
                )
            )
        self._records = tuple(records)

    async def authenticate(self, context: A2AAuthContext) -> A2APrincipal:
        key = self._get_header(context.headers, self._header_name)
        if not key:
            raise A2AAuthError("Missing API key")

        for record in self._records:
            digest = self._hash_key(key, record.salt)
            if hmac.compare_digest(digest, record.digest):
                return A2APrincipal(
                    subject=record.subject,
                    principal_type="service",
                    roles=record.roles,
                )
        raise A2AAuthError("Invalid API key")

    async def authorize(
        self,
        principal: A2APrincipal,
        *,
        action: str,
        resource: str,
        context: Mapping[str, JSONValue] | None = None,
    ) -> A2AAuthorizationDecision:
        _ = resource
        _ = context
        required_role = f"a2a:{action}"
        if "a2a:all" in principal.roles or required_role in principal.roles:
            return A2AAuthorizationDecision(allowed=True)
        return A2AAuthorizationDecision(
            allowed=False,
            reason=f"Missing required role '{required_role}'",
        )

    def _get_header(self, headers: Mapping[str, str], target: str) -> str | None:
        for key, value in headers.items():
            if key.lower() == target:
                return value
        return None

    def _hash_key(self, key: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac(
            "sha256",
            key.encode("utf-8"),
            salt,
            120_000,
        )


class JWTA2AAuthProvider:
    """JWT auth provider with claim-driven role authorization."""

    provider_id = "jwt"

    def __init__(
        self,
        *,
        secret: str,
        algorithms: tuple[str, ...] = ("HS256",),
        audience: str | None = None,
        issuer: str | None = None,
        role_claim: str = "roles",
        subject_claim: str = "sub",
    ) -> None:
        self._secret = secret
        self._algorithms = algorithms
        self._audience = audience
        self._issuer = issuer
        self._role_claim = role_claim
        self._subject_claim = subject_claim

    async def authenticate(self, context: A2AAuthContext) -> A2APrincipal:
        token = self._extract_bearer_token(context.headers)
        if not token:
            raise A2AAuthError("Missing bearer token")

        try:
            import jwt  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise A2AAuthError(
                "JWT auth requires 'PyJWT'. Install with: pip install PyJWT"
            ) from exc

        try:
            claims = jwt.decode(
                token,
                self._secret,
                algorithms=list(self._algorithms),
                audience=self._audience,
                issuer=self._issuer,
                options={"verify_signature": True},
            )
        except Exception as exc:
            raise A2AAuthError("Invalid bearer token") from exc

        subject = claims.get(self._subject_claim)
        if not isinstance(subject, str) or not subject.strip():
            raise A2AAuthError("JWT subject claim is missing")

        raw_roles = claims.get(self._role_claim, [])
        roles: tuple[str, ...]
        if isinstance(raw_roles, list):
            roles = tuple(str(item) for item in raw_roles)
        elif isinstance(raw_roles, str):
            roles = (raw_roles,)
        else:
            roles = ()

        return A2APrincipal(
            subject=subject,
            principal_type="service",
            roles=roles,
            claims={str(key): value for key, value in claims.items()},
        )

    async def authorize(
        self,
        principal: A2APrincipal,
        *,
        action: str,
        resource: str,
        context: Mapping[str, JSONValue] | None = None,
    ) -> A2AAuthorizationDecision:
        _ = resource
        _ = context
        required_role = f"a2a:{action}"
        if "a2a:all" in principal.roles or required_role in principal.roles:
            return A2AAuthorizationDecision(allowed=True)
        return A2AAuthorizationDecision(
            allowed=False,
            reason=f"Missing required role '{required_role}'",
        )

    def _extract_bearer_token(self, headers: Mapping[str, str]) -> str | None:
        auth: str | None = None
        for key, value in headers.items():
            if key.lower() == "authorization":
                auth = value
                break
        if not auth:
            return None
        prefix = "Bearer "
        if not auth.startswith(prefix):
            return None
        return auth[len(prefix) :].strip()
