"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Policy audit logging for compliance and security monitoring.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from ..types import JSONValue, PolicyDecision, PolicyEvent


class AuditAction(str, Enum):
    """Actions captured in audit log."""

    POLICY_EVALUATED = "policy_evaluated"
    POLICY_ALLOWED = "policy_allowed"
    POLICY_DENIED = "policy_denied"
    POLICY_MODIFIED = "policy_modified"
    TOOL_EXECUTED = "tool_executed"
    TOOL_DENIED = "tool_denied"
    LLM_CALLED = "llm_called"
    SUBAGENT_INVOKED = "subagent_invoked"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"


class AuditLevel(str, Enum):
    """Audit log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """
    Single audit record for compliance logging.
    """

    id: str
    timestamp_ms: int
    timestamp_iso: str
    level: str
    action: str
    actor: str | None
    resource: str | None
    decision: str | None
    reason: str | None
    policy_id: str | None
    matched_rules: list[str]
    event_type: str | None
    tool_name: str | None
    run_id: str | None
    thread_id: str | None
    metadata: dict[str, JSONValue]
    risk_score: float = 0.0


@dataclass
class AuditConfig:
    """Configuration for audit logging."""

    enabled: bool = True
    min_level: str = "info"  # Capture info and above
    include_payloads: bool = True
    max_payload_size: int = 10_000
    retention_days: int = 90
    sink: str = "console"  # console, file, syslog, custom
    file_path: str | None = None
    syslog_host: str | None = None
    syslog_port: int = 514
    custom_sink: Callable[[list[AuditRecord]], Any] | None = None
    redact_patterns: list[str] = field(
        default_factory=lambda: [
            "api[_-]?key",
            "secret",
            "password",
            "token",
            "credential",
        ]
    )


class AuditSink:
    """Base class for audit sinks."""

    def __init__(self, config: AuditConfig) -> None:
        self._config = config

    async def write(self, record: AuditRecord) -> None:
        """Write a single record."""
        await self._write_batch([record])

    async def write_batch(self, records: list[AuditRecord]) -> None:
        """Write multiple records."""
        await self._write_batch(records)

    async def _write_batch(self, records: list[AuditRecord]) -> None:
        """Implement batch write."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close the sink."""
        pass


class ConsoleAuditSink(AuditSink):
    """Audit sink that writes to console."""

    async def _write_batch(self, records: list[AuditRecord]) -> None:
        for record in records:
            print(
                f"[{record.timestamp_iso}] {record.level.upper()}: "
                f"{record.action} - {record.decision} - {record.reason}"
            )


class FileAuditSink(AuditSink):
    """Audit sink that writes to file."""

    def __init__(self, config: AuditConfig) -> None:
        super().__init__(config)
        self._file: Path | None = None
        self._buffer: list[AuditRecord] = []
        self._lock = asyncio.Lock()
        self._flush_interval_s = 5.0

    async def _write_batch(self, records: list[AuditRecord]) -> None:
        if not self._config.file_path:
            return

        async with self._lock:
            self._buffer.extend(records)

            if len(self._buffer) >= 100:
                await self._flush()

    async def _flush(self) -> None:
        if not self._buffer or not self._config.file_path:
            return

        path = Path(self._config.file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a") as f:
            for record in self._buffer:
                line = json.dumps(
                    {
                        "id": record.id,
                        "timestamp_ms": record.timestamp_ms,
                        "timestamp_iso": record.timestamp_iso,
                        "level": record.level,
                        "action": record.action,
                        "actor": record.actor,
                        "resource": record.resource,
                        "decision": record.decision,
                        "reason": record.reason,
                        "policy_id": record.policy_id,
                        "matched_rules": record.matched_rules,
                        "event_type": record.event_type,
                        "tool_name": record.tool_name,
                        "run_id": record.run_id,
                        "thread_id": record.thread_id,
                        "metadata": self._redact(record.metadata),
                        "risk_score": record.risk_score,
                    },
                    default=str,
                )
                f.write(line + "\n")

        self._buffer.clear()

    def _redact(self, data: dict[str, JSONValue]) -> dict[str, JSONValue]:
        """Redact sensitive fields."""
        import re

        def _redact_value(key: str, value: Any) -> Any:
            key_lower = key.lower()
            for pattern in self._config.redact_patterns:
                if re.search(pattern, key_lower):
                    return "[REDACTED]"
            return value

        result = {}
        for key, value in data.items():
            result[key] = _redact_value(key, value)
        return result

    async def close(self) -> None:
        """Flush and close."""
        await self._flush()


class PolicyAuditLogger:
    """
    Audit logger for policy decisions.

    Captures all policy evaluations for compliance,
    security monitoring, and debugging.
    """

    def __init__(
        self,
        config: AuditConfig | None = None,
        sink: AuditSink | None = None,
    ) -> None:
        self._config = config or AuditConfig()
        self._sink = sink or ConsoleAuditSink(self._config)
        self._records: list[AuditRecord] = []
        self._lock = asyncio.Lock()

    @property
    def config(self) -> AuditConfig:
        """Get audit config."""
        return self._config

    async def log_policy_decision(
        self,
        event: PolicyEvent,
        decision: PolicyDecision,
        *,
        actor: str | None = None,
        run_id: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        """
        Log a policy decision.

        Args:
            event: Policy event that was evaluated.
            decision: Policy decision.
            actor: Actor that initiated the action.
            run_id: Run ID.
            thread_id: Thread ID.
        """
        if not self._config.enabled:
            return

        # Determine action based on decision
        action = {
            "allow": AuditAction.POLICY_ALLOWED,
            "deny": AuditAction.POLICY_DENIED,
            "modify": AuditAction.POLICY_MODIFIED,
        }.get(decision.action, AuditAction.POLICY_EVALUATED)

        # Determine level based on decision
        level = {
            "allow": AuditLevel.INFO,
            "deny": AuditLevel.WARNING,
            "modify": AuditLevel.INFO,
        }.get(decision.action, AuditLevel.INFO)

        if decision.action == "deny":
            risk_score = 0.8
        elif decision.action == "modify":
            risk_score = 0.3
        else:
            risk_score = 0.0

        record = AuditRecord(
            id=f"audit-{int(time.time() * 1000)}-{len(self._records)}",
            timestamp_ms=int(time.time() * 1000),
            timestamp_iso=datetime.utcnow().isoformat(),
            level=level.value,
            action=action.value,
            actor=actor,
            resource=event.tool_name,
            decision=decision.action,
            reason=decision.reason,
            policy_id=decision.policy_id,
            matched_rules=decision.matched_rules,
            event_type=event.event_type,
            tool_name=event.tool_name,
            run_id=run_id,
            thread_id=thread_id,
            metadata=self._build_metadata(event),
            risk_score=risk_score,
        )

        async with self._lock:
            self._records.append(record)
            if len(self._records) >= 10:
                await self._flush()

    async def log_tool_execution(
        self,
        tool_name: str,
        allowed: bool,
        reason: str | None = None,
        *,
        run_id: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        """Log tool execution attempt."""
        if not self._config.enabled:
            return

        action = AuditAction.TOOL_EXECUTED if allowed else AuditAction.TOOL_DENIED
        level = AuditLevel.INFO if allowed else AuditLevel.WARNING

        record = AuditRecord(
            id=f"audit-{int(time.time() * 1000)}-{len(self._records)}",
            timestamp_ms=int(time.time() * 1000),
            timestamp_iso=datetime.utcnow().isoformat(),
            level=level.value,
            action=action.value,
            actor=None,
            resource=tool_name,
            decision="allowed" if allowed else "denied",
            reason=reason,
            policy_id=None,
            matched_rules=[],
            event_type="tool_call",
            tool_name=tool_name,
            run_id=run_id,
            thread_id=thread_id,
            metadata={},
            risk_score=0.5 if not allowed else 0.0,
        )

        async with self._lock:
            self._records.append(record)

    async def log_approval(
        self,
        approved: bool,
        reason: str | None = None,
        *,
        run_id: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        """Log approval decision."""
        if not self._config.enabled:
            return

        action = AuditAction.APPROVAL_GRANTED if approved else AuditAction.APPROVAL_DENIED
        level = AuditLevel.INFO if approved else AuditLevel.WARNING

        record = AuditRecord(
            id=f"audit-{int(time.time() * 1000)}-{len(self._records)}",
            timestamp_ms=int(time.time() * 1000),
            timestamp_iso=datetime.utcnow().isoformat(),
            level=level.value,
            action=action.value,
            actor=None,
            resource=None,
            decision="approved" if approved else "denied",
            reason=reason,
            policy_id=None,
            matched_rules=[],
            event_type="approval",
            tool_name=None,
            run_id=run_id,
            thread_id=thread_id,
            metadata={},
            risk_score=0.3 if not approved else 0.0,
        )

        async with self._lock:
            self._records.append(record)

    def _build_metadata(self, event: PolicyEvent) -> dict[str, JSONValue]:
        """Build metadata from policy event."""
        metadata: dict[str, JSONValue] = {}

        if self._config.include_payloads:
            metadata["context"] = event.context
            metadata["event_metadata"] = event.metadata

        return metadata

    async def _flush(self) -> None:
        """Flush buffered records to sink."""
        if not self._records:
            return

        records = self._records.copy()
        self._records.clear()

        try:
            await self._sink.write_batch(records)
        except Exception:
            pass  # Never let audit failures break execution

    async def flush(self) -> None:
        """Explicitly flush all records."""
        await self._flush()

    async def close(self) -> None:
        """Close the audit logger."""
        await self.flush()
        await self._sink.close()


def create_policy_audit_logger(
    *,
    enabled: bool = True,
    file_path: str | None = None,
    min_level: str = "info",
) -> PolicyAuditLogger:
    """Create a policy audit logger."""
    config = AuditConfig(
        enabled=enabled,
        file_path=file_path,
        min_level=min_level,
    )

    if file_path:
        sink = FileAuditSink(config)
    else:
        sink = ConsoleAuditSink(config)

    return PolicyAuditLogger(config=config, sink=sink)
