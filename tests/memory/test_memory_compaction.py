"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Tests for memory compaction.
"""

import pytest
from afk.memory import (
    CompactionConfig,
    CompactionStats,
    MemoryCompactor,
)
from afk.memory.types import MemoryEvent


class TestCompactionConfig:
    """Tests for CompactionConfig."""

    def test_defaults(self) -> None:
        """Test default config."""
        config = CompactionConfig()

        assert config.enabled is True
        assert config.trigger_threshold_bytes == 5_000_000
        assert config.target_size_bytes == 2_000_000
        assert config.pressure_threshold == 0.7

    def test_custom(self) -> None:
        """Test custom config."""
        config = CompactionConfig(
            enabled=True,
            trigger_threshold_bytes=1_000_000,
            target_size_bytes=500_000,
            pressure_threshold=0.5,
        )

        assert config.trigger_threshold_bytes == 1_000_000
        assert config.target_size_bytes == 500_000
        assert config.pressure_threshold == 0.5


class TestCompactionStats:
    """Tests for CompactionStats."""

    def test_defaults(self) -> None:
        """Test default stats."""
        stats = CompactionStats()

        assert stats.runs == 0
        assert stats.events_compacted == 0
        assert stats.episodes_created == 0
        assert stats.bytes_freed == 0

    def test_increment(self) -> None:
        """Test incrementing stats."""
        stats = CompactionStats()
        stats.runs += 1
        stats.events_compacted += 10
        stats.episodes_created += 2
        stats.bytes_freed += 1000

        assert stats.runs == 1
        assert stats.events_compacted == 10
        assert stats.episodes_created == 2
        assert stats.bytes_freed == 1000


class TestMemoryCompactor:
    """Tests for MemoryCompactor."""

    def test_defaults(self) -> None:
        """Test default compactor."""
        compactor = MemoryCompactor()

        assert compactor.config.enabled is True
        assert compactor.stats.runs == 0

    def test_custom_config(self) -> None:
        """Test custom config."""
        config = CompactionConfig(
            enabled=False,
            trigger_threshold_bytes=1000,
            target_size_bytes=500,
        )
        compactor = MemoryCompactor(config=config)

        assert compactor.config.enabled is False
        assert compactor.config.trigger_threshold_bytes == 1000

    def test_compute_memory_pressure(self) -> None:
        """Test memory pressure computation."""
        compactor = MemoryCompactor()

        pressure = compactor.compute_memory_pressure(0)
        assert pressure == 0.0

        pressure = compactor.compute_memory_pressure(2_500_000)
        assert pressure == 0.5

        pressure = compactor.compute_memory_pressure(5_000_000)
        assert pressure == 1.0

        pressure = compactor.compute_memory_pressure(10_000_000)
        assert pressure == 1.0

    @pytest.mark.asyncio
    async def test_should_compact_disabled(self) -> None:
        """Test compaction disabled."""
        config = CompactionConfig(enabled=False)
        compactor = MemoryCompactor(config=config)

        should = await compactor.should_compact(100, 1_000_000)
        assert should is False

    @pytest.mark.asyncio
    async def test_should_compact_min_events(self) -> None:
        """Test min events threshold."""
        compactor = MemoryCompactor(config=CompactionConfig(min_events_before_compaction=50))

        should = await compactor.should_compact(10, 10_000_000)
        assert should is False

    @pytest.mark.asyncio
    async def test_should_compact_pressure(self) -> None:
        """Test pressure threshold."""
        compactor = MemoryCompactor(
            config=CompactionConfig(
                pressure_threshold=0.7,
                min_events_before_compaction=10,
            )
        )

        should = await compactor.should_compact(50, 3_500_000)
        assert should is True

        should = await compactor.should_compact(50, 3_000_000)
        assert should is False

    @pytest.mark.asyncio
    async def test_compact_empty(self) -> None:
        """Test compacting empty events."""
        compactor = MemoryCompactor()

        result = await compactor.compact([], "thread-1")

        assert result.episodes_created == 0
        assert result.events_compacted == 0


class TestBackgroundCompactor:
    """Tests for BackgroundCompactor."""

    @pytest.mark.asyncio
    async def test_stop(self) -> None:
        """Test stopping background compactor."""
        from afk.memory import BackgroundCompactor

        compactor = BackgroundCompactor()
        compactor.stop()

        assert compactor._running is False
