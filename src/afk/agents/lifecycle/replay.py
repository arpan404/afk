"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Checkpoint replay from any checkpoint in event history.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ..errors import AgentCheckpointCorruptionError


@dataclass
class CheckpointInfo:
    """
    Information about a checkpoint for replay.

    Attributes:
        run_id: Run this checkpoint belongs to.
        step: Step number at checkpoint.
        phase: Checkpoint phase.
        timestamp_ms: When checkpoint was created.
        state: Agent state at checkpoint.
    """

    run_id: str
    step: int
    phase: str
    timestamp_ms: int
    state: str = "running"


@dataclass
class CheckpointReplayHandle:
    """
    Handle for replaying from a specific checkpoint.

    Provides methods to navigate and replay agent execution
    from any checkpoint in the event history.
    """

    run_id: str
    thread_id: str
    memory: Any  # MemoryStore
    _checkpoint_step: int = 0

    async def list_checkpoints(self) -> list[CheckpointInfo]:
        """
        List all checkpoints for this run.

        Returns:
            List of CheckpointInfo sorted by step number.
        """
        checkpoints = []
        step = 0
        while True:
            key = f"checkpoint:{self.run_id}:step:{step}"
            data = await self.memory.get_state(self.thread_id, key)
            if not data:
                break
            if isinstance(data, dict):
                checkpoints.append(
                    CheckpointInfo(
                        run_id=self.run_id,
                        step=int(data.get("step", step)),
                        phase=str(data.get("phase", "")),
                        timestamp_ms=int(data.get("timestamp_ms", 0)),
                        state=str(data.get("state", "running")),
                    )
                )
            step += 1

        # Also check latest checkpoint
        latest_key = f"checkpoint_latest:{self.run_id}"
        latest = await self.memory.get_state(self.thread_id, latest_key)
        if latest and isinstance(latest, dict):
            step_val = int(latest.get("step", 0))
            # Avoid duplicates
            if not any(c.step == step_val for c in checkpoints):
                checkpoints.append(
                    CheckpointInfo(
                        run_id=self.run_id,
                        step=step_val,
                        phase=str(latest.get("phase", "")),
                        timestamp_ms=int(latest.get("timestamp_ms", 0)),
                        state=str(latest.get("state", "running")),
                    )
                )

        return sorted(checkpoints, key=lambda c: c.step)

    async def get_checkpoint_at_step(self, step: int) -> dict[str, Any] | None:
        """
        Get checkpoint data at a specific step.

        Args:
            step: Step number to retrieve.

        Returns:
            Checkpoint data or None if not found.
        """
        key = f"checkpoint:{self.run_id}:step:{step}"
        data = await self.memory.get_state(self.thread_id, key)
        return data if isinstance(data, dict) else None

    async def replay_to_step(
        self,
        target_step: int,
        *,
        include_partial: bool = True,
    ) -> dict[str, Any]:
        """
        Get all data needed to replay to a specific step.

        Args:
            target_step: Step to replay to.
            include_partial: Include events up to but not including target.

        Returns:
            Dictionary with checkpoint data and events for replay.
        """
        # Get checkpoint at target step
        checkpoint = await self.get_checkpoint_at_step(target_step)
        if not checkpoint:
            raise AgentCheckpointCorruptionError(
                f"No checkpoint found at step {target_step} for run {self.run_id}"
            )

        # Get events up to target step
        events = await self.memory.get_events_since(
            self.thread_id,
            since_ms=0,  # Get all events
            limit=10000,
        )

        # Filter events to those up to and including target step
        relevant_events = []
        for event in events:
            event_step = event.payload.get("step", 0) if isinstance(event.payload, dict) else 0
            if include_partial:
                if event_step <= target_step:
                    relevant_events.append(event)
            else:
                if event_step < target_step:
                    relevant_events.append(event)

        return {
            "checkpoint": checkpoint,
            "events": relevant_events,
            "target_step": target_step,
            "replay_mode": "partial" if include_partial else "full",
        }

    async def get_replay_snapshot(self, step: int) -> dict[str, Any]:
        """
        Get a complete replay snapshot for a given step.

        This includes all runtime state needed to continue execution
        from exactly that step.

        Args:
            step: Step to get snapshot for.

        Returns:
            Complete replay snapshot.
        """
        replay_data = await self.replay_to_step(step)

        # Also load runtime state
        runtime_key = f"runtime:{self.run_id}:step:{step}"
        runtime_state = await self.memory.get_state(self.thread_id, runtime_key)

        return {
            **replay_data,
            "runtime_state": runtime_state if isinstance(runtime_state, dict) else {},
        }


class CheckpointReplayManager:
    """
    Manager for checkpoint replay operations.

    Provides unified interface for listing, selecting,
    and replaying from checkpoints.
    """

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    async def open_replay(
        self,
        run_id: str,
        thread_id: str,
    ) -> CheckpointReplayHandle:
        """
        Open a replay handle for a run.

        Args:
            run_id: Run identifier.
            thread_id: Thread identifier.

        Returns:
            CheckpointReplayHandle for this run.
        """
        return CheckpointReplayHandle(
            run_id=run_id,
            thread_id=thread_id,
            memory=self._memory,
        )

    async def get_latest_checkpoint(
        self,
        run_id: str,
        thread_id: str,
    ) -> CheckpointInfo | None:
        """
        Get the latest checkpoint for a run.

        Args:
            run_id: Run identifier.
            thread_id: Thread identifier.

        Returns:
            CheckpointInfo or None if no checkpoints exist.
        """
        latest_key = f"checkpoint_latest:{run_id}"
        data = await self._memory.get_state(thread_id, latest_key)
        if data and isinstance(data, dict):
            return CheckpointInfo(
                run_id=run_id,
                step=int(data.get("step", 0)),
                phase=str(data.get("phase", "")),
                timestamp_ms=int(data.get("timestamp_ms", 0)),
                state=str(data.get("state", "running")),
            )
        return None

    async def find_checkpoint_before(
        self,
        run_id: str,
        thread_id: str,
        timestamp_ms: int,
    ) -> CheckpointInfo | None:
        """
        Find the checkpoint just before a given timestamp.

        Args:
            run_id: Run identifier.
            thread_id: Thread identifier.
            timestamp_ms: Target timestamp.

        Returns:
            CheckpointInfo of the closest checkpoint before timestamp.
        """
        handle = await self.open_replay(run_id, thread_id)
        checkpoints = await handle.list_checkpoints()

        # Find the closest checkpoint before timestamp
        closest = None
        for ckpt in checkpoints:
            if ckpt.timestamp_ms <= timestamp_ms:
                if closest is None or ckpt.timestamp_ms > closest.timestamp_ms:
                    closest = ckpt

        return closest