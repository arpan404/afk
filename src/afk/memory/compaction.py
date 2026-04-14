"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Memory compaction with importance-based consolidation and auto-compaction.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..agents.types.memory import (
    ConsolidationResult,
    Episode,
    ScoredMemoryEvent,
    compute_event_importance,
    should_compact_event,
)
from ..memory.types import MemoryEvent
from .types import JsonValue


@dataclass
class CompactionConfig:
    """Configuration for memory compaction."""

    enabled: bool = True
    trigger_threshold_bytes: int = 5_000_000  # 5MB
    target_size_bytes: int = 2_000_000  # 2MB
    pressure_threshold: float = 0.7  # 0-1, trigger at 70% pressure
    min_events_before_compaction: int = 50
    max_events_per_episode: int = 20
    episode_compression_ratio: float = 0.3  # Target 30% of original size
    compaction_interval_s: float = 60.0  # Check every 60s
    batch_size: int = 100
    exclude_tags: list[str] = field(default_factory=list)


SummarizerCallback = Callable[
    [list[ScoredMemoryEvent]],
    Awaitable[str],
]


@dataclass
class CompactionStats:
    """Statistics from compaction operations."""

    runs: int = 0
    events_compacted: int = 0
    episodes_created: int = 0
    bytes_freed: int = 0
    last_run_ms: int = 0


class MemoryCompactor:
    """
    Auto-compaction engine for memory stores.

    Continuously monitors memory pressure and compacts low-importance
    events while preserving high-value context.
    """

    def __init__(
        self,
        summarizer: SummarizerCallback | None = None,
        config: CompactionConfig | None = None,
    ) -> None:
        self._summarizer = summarizer
        self._config = config or CompactionConfig()
        self._stats = CompactionStats()
        self._running = False

    @property
    def config(self) -> CompactionConfig:
        """Get compaction config."""
        return self._config

    @property
    def stats(self) -> CompactionStats:
        """Get compaction stats."""
        return self._stats

    def compute_memory_pressure(
        self,
        current_size_bytes: int,
    ) -> float:
        """
        Compute current memory pressure level.

        Args:
            current_size_bytes: Current memory usage in bytes.

        Returns:
            Pressure level from 0.0 (empty) to 1.0 (full).
        """
        if self._config.trigger_threshold_bytes <= 0:
            return 0.0

        return min(1.0, current_size_bytes / self._config.trigger_threshold_bytes)

    async def should_compact(
        self,
        event_count: int,
        current_size_bytes: int,
    ) -> bool:
        """
        Determine if compaction should run.

        Args:
            event_count: Number of events in memory.
            current_size_bytes: Current memory usage.

        Returns:
            True if compaction should run.
        """
        if not self._config.enabled:
            return False

        if event_count < self._config.min_events_before_compaction:
            return False

        pressure = self.compute_memory_pressure(current_size_bytes)
        return pressure >= self._config.pressure_threshold

    async def score_events(
        self,
        events: list[MemoryEvent],
        *,
        thread_id: str,
    ) -> list[ScoredMemoryEvent]:
        """
        Score events by importance for compaction decisions.

        Args:
            events: Events to score.
            thread_id: Thread ID for context.

        Returns:
            Scored events sorted by importance (highest first).
        """
        scored = []

        for event in events:
            event_type = event.event_type or "message"
            payload = event.payload or {}

            # Extract signals for importance
            has_tool_failure = bool(payload.get("error")) or bool(payload.get("tool_failure"))
            has_user_correction = "correction" in str(payload.get("event_type", "")).lower()
            is_goal_milestone = payload.get("step_complete") or payload.get("goal_reached")
            has_error = bool(payload.get("error"))
            payload_size = len(str(payload))

            importance = compute_event_importance(
                event_type,
                has_tool_failure=has_tool_failure,
                has_user_correction=has_user_correction,
                is_goal_milestone=is_goal_milestone,
                payload_size=payload_size,
                has_error=has_error,
            )

            scored.append(
                ScoredMemoryEvent(
                    event_id=event.event_id,
                    thread_id=thread_id,
                    importance=importance,
                    event_type=event_type,
                    timestamp_ms=event.timestamp_ms,
                    summary=payload.get("summary", ""),
                    tags=event.tags,
                    should_compact=should_compact_event(
                        importance, self._config.pressure_threshold
                    ),
                    payload_preview=str(payload)[:200],
                )
            )

        # Sort by importance descending
        return sorted(scored, key=lambda e: e.importance, reverse=True)

    async def create_episodes(
        self,
        scored_events: list[ScoredMemoryEvent],
    ) -> list[Episode]:
        """
        Create episodes by grouping events.

        Args:
            scored_events: Scored events to group.

        Returns:
            Created episodes.
        """
        episodes: list[Episode] = []

        # Group into batches
        batch_size = self._config.max_events_per_episode
        for i in range(0, len(scored_events), batch_size):
            batch = scored_events[i : i + batch_size]
            if not batch:
                continue

            # Check if all should compact
            if not any(e.should_compact for e in batch):
                continue

            start_ms = min(e.timestamp_ms for e in batch)
            end_ms = max(e.timestamp_ms for e in batch)
            avg_importance = sum(e.importance for e in batch) / len(batch)

            # Generate summary
            if self._summarizer:
                summary = await self._summarizer(batch)
            else:
                # Simple text summary
                summary = f"Episode with {len(batch)} events"

            episode = Episode(
                id=f"episode-{int(time.time() * 1000)}-{i}",
                thread_id=batch[0].thread_id,
                start_ms=start_ms,
                end_ms=end_ms,
                summary=summary,
                event_count=len(batch),
                importance=avg_importance,
                tags=[t for e in batch for t in e.tags],
            )
            episodes.append(episode)

        return episodes

    async def compact(
        self,
        events: list[MemoryEvent],
        thread_id: str,
    ) -> ConsolidationResult:
        """
        Run compaction on events.

        Args:
            events: Events to compact.
            thread_id: Thread ID.

        Returns:
            ConsolidationResult with stats.
        """
        if not events:
            return ConsolidationResult()

        start_ms = time.time() * 1000

        # Score all events
        scored = await self.score_events(events, thread_id=thread_id)

        # Separate into compacted and retained
        to_compact = [e for e in scored if e.should_compact]
        to_retain = [e for e in scored if not e.should_compact]

        events_compacted = len(to_compact)

        # Create episodes
        episodes = await self.create_episodes(to_compact)

        # Estimate bytes saved
        original_size = sum(len(str(e.payload_preview)) for e in to_compact)
        episode_size = sum(len(e.summary) for e in episodes)
        bytes_freed = max(0, original_size - episode_size)

        self._stats.runs += 1
        self._stats.events_compacted += events_compacted
        self._stats.episodes_created += len(episodes)
        self._stats.bytes_freed += bytes_freed
        self._stats.last_run_ms = int(time.time() * 1000)

        consolidate_time_ms = (time.time() * 1000) - start_ms

        return ConsolidationResult(
            episodes_created=len(episodes),
            events_compacted=events_compacted,
            events_dropped=0,
            memory_saved_bytes=bytes_freed,
            consolidation_time_ms=consolidate_time_ms,
        )

    async def run_compaction_loop(
        self,
        get_memory_stats: Callable[[], Awaitable[tuple[int, int]]],
        get_events: Callable[[int], Awaitable[list[MemoryEvent]]],
        thread_id: str,
    ) -> ConsolidationResult | None:
        """
        Run the compaction loop.

        Args:
            get_memory_stats: Returns (event_count, size_bytes).
            get_events: Returns events up to limit.
            thread_id: Thread ID.

        Returns:
            ConsolidationResult if compaction ran, None otherwise.
        """
        self._running = True

        while self._running:
            try:
                event_count, size_bytes = await get_memory_stats()

                if await self.should_compact(event_count, size_bytes):
                    events = await get_events(self._config.batch_size * 10)
                    result = await self.compact(events, thread_id)
                    return result
                else:
                    await asyncio.sleep(self._config.compaction_interval_s)

            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(self._config.compaction_interval_s)

        return None

    def stop(self) -> None:
        """Stop the compaction loop."""
        self._running = False


class BackgroundCompactor(MemoryCompactor):
    """
    Background compactor that runs periodically in a separate task.
    """

    def __init__(
        self,
        summarizer: SummarizerCallback | None = None,
        config: CompactionConfig | None = None,
    ) -> None:
        super().__init__(summarizer, config)
        self._task: asyncio.Task | None = None

    async def start(
        self,
        get_memory_stats: Callable[[], Awaitable[tuple[int, int]]],
        get_events: Callable[[int], Awaitable[list[MemoryEvent]]],
        thread_id: str,
    ) -> None:
        """Start background compaction."""
        self._task = asyncio.create_task(
            self.run_compaction_loop(get_memory_stats, get_events, thread_id)
        )

    async def stop(self) -> None:
        """Stop background compaction."""
        self.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
