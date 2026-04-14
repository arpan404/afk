"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Dead letter queue for failed tool calls and operations.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from ..llms.types import JSONValue


@dataclass(frozen=True, slots=True)
class DeadLetterEntry:
    """
    A failed operation entry in the dead letter queue.

    Attributes:
        id: Unique entry identifier.
        operation_type: Type of operation (tool_call, llm_call, subagent_call, etc.).
        operation_name: Name of the specific operation.
        payload: Operation input payload.
        error: Error message or reason for failure.
        attempt_count: Number of attempts made.
        last_attempt_ms: Timestamp of last attempt.
        next_retry_ms: When to retry (0 if no more retries).
        max_attempts: Maximum allowed attempts.
        created_ms: When entry was created.
        metadata: Additional operation metadata.
    """

    id: str
    operation_type: str
    operation_name: str
    payload: dict[str, JSONValue] = field(default_factory=dict)
    error: str = ""
    attempt_count: int = 0
    last_attempt_ms: int = 0
    next_retry_ms: int = 0
    max_attempts: int = 3
    created_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def can_retry(self) -> bool:
        """Check if entry can be retried."""
        return self.attempt_count < self.max_attempts

    def is_ready_for_retry(self) -> bool:
        """Check if entry is ready for next retry attempt."""
        if not self.can_retry():
            return False
        now = int(time.time() * 1000)
        return now >= self.next_retry_ms


@dataclass
class DeadLetterQueue:
    """
    Queue for storing failed operations that exceeded retry limits.

    Provides persistence and retry management for failed operations,
    with configurable retry policies and TTL.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_retry_delay_ms: int = 1000,
        max_retry_delay_ms: int = 60000,
        entry_ttl_ms: int = 86400000,
    ) -> None:
        """
        Initialize dead letter queue.

        Args:
            max_attempts: Maximum retry attempts before moving to DLQ.
            base_retry_delay_ms: Base delay between retries (exponential backoff).
            max_retry_delay_ms: Maximum retry delay cap.
            entry_ttl_ms: Time-to-live for DLQ entries in milliseconds.
        """
        self._entries: dict[str, DeadLetterEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()
        self._max_attempts = max_attempts
        self._base_retry_delay_ms = base_retry_delay_ms
        self._max_retry_delay_ms = max_retry_delay_ms
        self._entry_ttl_ms = entry_ttl_ms

    def _lock_for(self, entry_id: str) -> asyncio.Lock:
        if entry_id not in self._locks:
            self._locks[entry_id] = asyncio.Lock()
        return self._locks[entry_id]

    def _compute_backoff(self, attempt: int) -> int:
        """Compute exponential backoff delay."""
        delay = self._base_retry_delay_ms * (2 ** attempt)
        return min(delay, self._max_retry_delay_ms)

    def _new_id(self, operation_type: str, operation_name: str) -> str:
        """Generate unique entry ID."""
        ts = int(time.time() * 1000000)
        return f"dlq-{operation_type}-{operation_name}-{ts}"

    async def add_entry(
        self,
        operation_type: str,
        operation_name: str,
        payload: dict[str, JSONValue] | None = None,
        error: str = "",
        metadata: dict[str, JSONValue] | None = None,
    ) -> str:
        """
        Add a failed operation to the dead letter queue.

        Args:
            operation_type: Type of operation (tool_call, llm_call, etc.).
            operation_name: Name of the specific operation.
            payload: Operation input payload.
            error: Error message.
            metadata: Additional metadata.

        Returns:
            The generated entry ID.
        """
        entry_id = self._new_id(operation_type, operation_name)
        now_ms = int(time.time() * 1000)

        entry = DeadLetterEntry(
            id=entry_id,
            operation_type=operation_type,
            operation_name=operation_name,
            payload=payload or {},
            error=error,
            attempt_count=self._max_attempts,  # Already exhausted
            last_attempt_ms=now_ms,
            next_retry_ms=0,  # No more retries
            max_attempts=self._max_attempts,
            created_ms=now_ms,
            metadata=metadata or {},
        )

        async with self._lock:
            self._entries[entry_id] = entry

        return entry_id

    async def record_failure(
        self,
        operation_type: str,
        operation_name: str,
        payload: dict[str, JSONValue] | None = None,
        error: str = "",
        attempt: int = 1,
        metadata: dict[str, JSONValue] | None = None,
    ) -> str:
        """
        Record a failure and potentially add to DLQ if max attempts exceeded.

        Args:
            operation_type: Type of operation.
            operation_name: Name of the operation.
            payload: Operation payload.
            error: Error message.
            attempt: Current attempt number.
            metadata: Additional metadata.

        Returns:
            Entry ID (new or existing).
        """
        entry_id = self._new_id(operation_type, operation_name)
        now_ms = int(time.time() * 1000)

        if attempt >= self._max_attempts:
            # Max attempts exceeded, add to DLQ
            return await self.add_entry(
                operation_type=operation_type,
                operation_name=operation_name,
                payload=payload,
                error=error,
                metadata=metadata,
            )

        # Calculate next retry time
        next_retry = now_ms + self._compute_backoff(attempt)

        entry = DeadLetterEntry(
            id=entry_id,
            operation_type=operation_type,
            operation_name=operation_name,
            payload=payload or {},
            error=error,
            attempt_count=attempt,
            last_attempt_ms=now_ms,
            next_retry_ms=next_retry,
            max_attempts=self._max_attempts,
            created_ms=now_ms,
            metadata=metadata or {},
        )

        async with self._lock:
            self._entries[entry_id] = entry

        return entry_id

    async def get_entry(self, entry_id: str) -> DeadLetterEntry | None:
        """Get a specific DLQ entry by ID."""
        async with self._lock:
            return self._entries.get(entry_id)

    async def get_ready_retry(self) -> list[DeadLetterEntry]:
        """
        Get all entries that are ready for retry.

        Returns:
            List of entries ready for retry.
        """
        now_ms = int(time.time() * 1000)
        async with self._lock:
            return [
                entry
                for entry in self._entries.values()
                if entry.is_ready_for_retry() and (now_ms - entry.created_ms) < self._entry_ttl_ms
            ]

    async def remove_entry(self, entry_id: str) -> bool:
        """
        Remove an entry from the DLQ.

        Args:
            entry_id: Entry ID to remove.

        Returns:
            True if removed, False if not found.
        """
        async with self._lock:
            if entry_id in self._entries:
                del self._entries[entry_id]
                return True
            return False

    async def list_entries(
        self,
        operation_type: str | None = None,
        include_completed: bool = False,
    ) -> list[DeadLetterEntry]:
        """
        List DLQ entries with optional filtering.

        Args:
            operation_type: Filter by operation type.
            include_completed: Include entries past TTL.

        Returns:
            List of matching entries.
        """
        now_ms = int(time.time() * 1000)
        async with self._lock:
            entries = []
            for entry in self._entries.values():
                if operation_type and entry.operation_type != operation_type:
                    continue
                if not include_completed:
                    age_ms = now_ms - entry.created_ms
                    if age_ms > self._entry_ttl_ms:
                        continue
                entries.append(entry)
            return entries

    async def retry_entry(self, entry_id: str) -> DeadLetterEntry | None:
        """
        Mark an entry for retry by resetting its attempt count.

        Args:
            entry_id: Entry ID to retry.

        Returns:
            Updated entry or None if not found.
        """
        async with self._lock:
            entry = self._entries.get(entry_id)
            if not entry:
                return None

            # Reset for retry
            new_attempt = max(0, entry.attempt_count - 1)
            new_id = self._new_id(entry.operation_type, entry.operation_name)

            now_ms = int(time.time() * 1000)
            updated = DeadLetterEntry(
                id=new_id,
                operation_type=entry.operation_type,
                operation_name=entry.operation_name,
                payload=entry.payload,
                error=entry.error,
                attempt_count=new_attempt,
                last_attempt_ms=now_ms,
                next_retry_ms=now_ms + self._compute_backoff(new_attempt),
                max_attempts=entry.max_attempts,
                created_ms=entry.created_ms,
                metadata=entry.metadata,
            )

            # Remove old, add new
            del self._entries[entry_id]
            self._entries[new_id] = updated
            return updated

    async def clear_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        now_ms = int(time.time() * 1000)
        removed = 0
        async with self._lock:
            expired_ids = [
                entry_id
                for entry_id, entry in self._entries.items()
                if (now_ms - entry.created_ms) > self._entry_ttl_ms
            ]
            for entry_id in expired_ids:
                del self._entries[entry_id]
                removed += 1
            return removed


# Global DLQ instance
_global_dlq: DeadLetterQueue | None = None


def get_dead_letter_queue() -> DeadLetterQueue:
    """Get the global dead letter queue instance."""
    global _global_dlq
    if _global_dlq is None:
        _global_dlq = DeadLetterQueue()
    return _global_dlq