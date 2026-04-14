"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Tests for policy audit logging.
"""

import pytest
from afk.agents.policy.audit import (
    AuditAction,
    AuditConfig,
    AuditLevel,
    AuditRecord,
    AuditSink,
    ConsoleAuditSink,
    FileAuditSink,
    PolicyAuditLogger,
    create_policy_audit_logger,
)
from afk.agents.types.policy import PolicyDecision, PolicyEvent


class TestAuditConfig:
    """Tests for AuditConfig."""

    def test_defaults(self) -> None:
        """Test default config."""
        config = AuditConfig()

        assert config.enabled is True
        assert config.min_level == "info"
        assert config.include_payloads is True
        assert config.max_payload_size == 10_000
        assert config.retention_days == 90
        assert config.sink == "console"

    def test_custom(self) -> None:
        """Test custom config."""
        config = AuditConfig(
            enabled=False,
            min_level="warning",
            include_payloads=False,
            file_path="/tmp/audit.log",
        )

        assert config.enabled is False
        assert config.min_level == "warning"
        assert config.include_payloads is False
        assert config.file_path == "/tmp/audit.log"

    def test_redact_patterns(self) -> None:
        """Test default redact patterns."""
        config = AuditConfig()

        assert "api[_-]?key" in config.redact_patterns
        assert "secret" in config.redact_patterns
        assert "password" in config.redact_patterns


class TestAuditRecord:
    """Tests for AuditRecord."""

    def test_creation(self) -> None:
        """Test creating audit record."""
        record = AuditRecord(
            id="audit-1",
            timestamp_ms=1000,
            timestamp_iso="2026-01-01T00:00:00",
            level="info",
            action="policy_evaluated",
            actor="system",
            resource="webfetch",
            decision="allow",
            reason="OK",
            policy_id="rule-1",
            matched_rules=["rule-1"],
            event_type="tool_call",
            tool_name="webfetch",
            run_id="run-1",
            thread_id="thread-1",
            metadata={"key": "value"},
            risk_score=0.0,
        )

        assert record.id == "audit-1"
        assert record.decision == "allow"
        assert record.risk_score == 0.0


class TestConsoleAuditSink:
    """Tests for ConsoleAuditSink."""

    @pytest.mark.asyncio
    async def test_write_batch(self) -> None:
        """Test writing batch to console."""
        config = AuditConfig()
        sink = ConsoleAuditSink(config)

        record = AuditRecord(
            id="audit-1",
            timestamp_ms=1000,
            timestamp_iso="2026-01-01T00:00:00",
            level="info",
            action="policy_evaluated",
            actor=None,
            resource=None,
            decision="allow",
            reason=None,
            policy_id=None,
            matched_rules=[],
            event_type=None,
            tool_name=None,
            run_id=None,
            thread_id=None,
            metadata={},
        )

        await sink._write_batch([record])


class TestFileAuditSink:
    """Tests for FileAuditSink."""

    @pytest.mark.asyncio
    async def test_redact(self) -> None:
        """Test redaction."""
        config = AuditConfig()
        sink = FileAuditSink(config)

        data = {
            "api_key": "secret123",
            "username": "admin",
            "password": "hidden",
            "action": "test",
        }

        result = sink._redact(data)

        assert result["api_key"] == "[REDACTED]"
        assert result["password"] == "[REDACTED]"
        assert result["username"] == "admin"
        assert result["action"] == "test"


class TestPolicyAuditLogger:
    """Tests for PolicyAuditLogger."""

    @pytest.mark.asyncio
    async def test_defaults(self) -> None:
        """Test default logger."""
        logger = PolicyAuditLogger()

        assert logger.config.enabled is True

    @pytest.mark.asyncio
    async def test_log_tool_execution_allowed(self) -> None:
        """Test logging tool execution."""
        config = AuditConfig(enabled=True)
        logger = PolicyAuditLogger(config=config)

        await logger.log_tool_execution(
            "webfetch",
            allowed=True,
            run_id="run-1",
            thread_id="thread-1",
        )

        await logger.flush()

    @pytest.mark.asyncio
    async def test_log_tool_execution_denied(self) -> None:
        """Test logging tool execution denied."""
        config = AuditConfig(enabled=True)
        logger = PolicyAuditLogger(config=config)

        await logger.log_tool_execution(
            "shell",
            allowed=False,
            reason="Shell denied by policy",
            run_id="run-1",
            thread_id="thread-1",
        )

        await logger.flush()

    @pytest.mark.asyncio
    async def test_log_approval_granted(self) -> None:
        """Test logging approval granted."""
        config = AuditConfig(enabled=True)
        logger = PolicyAuditLogger(config=config)

        await logger.log_approval(
            approved=True,
            run_id="run-1",
            thread_id="thread-1",
        )

        await logger.flush()

    @pytest.mark.asyncio
    async def test_log_approval_denied(self) -> None:
        """Test logging approval denied."""
        config = AuditConfig(enabled=True)
        logger = PolicyAuditLogger(config=config)

        await logger.log_approval(
            approved=False,
            reason="Not approved",
            run_id="run-1",
            thread_id="thread-1",
        )

        await logger.flush()

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test closing logger."""
        config = AuditConfig(enabled=True)
        logger = PolicyAuditLogger(config=config)

        await logger.close()


class TestCreatePolicyAuditLogger:
    """Tests for create_policy_audit_logger."""

    def test_console(self) -> None:
        """Test creating console logger."""
        logger = create_policy_audit_logger()

        assert logger.config.enabled is True
        assert logger.config.sink == "console"

    def test_file(self) -> None:
        """Test creating file logger."""
        logger = create_policy_audit_logger(
            file_path="/tmp/audit.log",
        )

        assert logger.config.file_path == "/tmp/audit.log"

    def test_disabled(self) -> None:
        """Test creating disabled logger."""
        logger = create_policy_audit_logger(enabled=False)

        assert logger.config.enabled is False
