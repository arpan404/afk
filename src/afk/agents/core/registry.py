"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Agent registry and discovery for runtime agent management.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from .base import BaseAgent


@dataclass
class AgentRegistration:
    """
    Registration entry for an agent in the registry.

    Attributes:
        agent: The agent instance.
        name: Agent name (unique identifier).
        description: Human-readable description.
        tags: Tags for categorization and discovery.
        created_at_ms: Registration timestamp.
        metadata: Additional metadata.
    """

    agent: BaseAgent
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    created_at_ms: int = field(default_factory=lambda: int(__import__("time").time() * 1000))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRegistry:
    """
    Thread-safe registry for agent instances.

    Provides registration, lookup, and discovery of agents at runtime.
    Supports tags for categorical lookup and metadata for extensibility.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentRegistration] = {}
        self._tags_index: dict[str, set[str]] = {}  # tag -> agent names
        self._lock = asyncio.Lock()

    async def register(
        self,
        agent: BaseAgent,
        name: str | None = None,
        description: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Register an agent with the registry.

        Args:
            agent: Agent instance to register.
            name: Optional name override (defaults to agent.name).
            description: Human-readable description.
            tags: Tags for categorization.
            metadata: Additional metadata.

        Returns:
            The registered agent name.

        Raises:
            ValueError: If agent has no name and no name provided.
        """
        agent_name = name or agent.name
        if not agent_name:
            raise ValueError("Agent must have a name or a name must be provided")

        async with self._lock:
            reg = AgentRegistration(
                agent=agent,
                name=agent_name,
                description=description,
                tags=tags or [],
                metadata=metadata or {},
            )
            self._agents[agent_name] = reg

            # Update tags index
            for tag in reg.tags:
                if tag not in self._tags_index:
                    self._tags_index[tag] = set()
                self._tags_index[tag].add(agent_name)

        return agent_name

    async def unregister(self, name: str) -> bool:
        """
        Remove an agent from the registry.

        Args:
            name: Agent name to remove.

        Returns:
            True if agent was removed, False if not found.
        """
        async with self._lock:
            if name not in self._agents:
                return False

            reg = self._agents.pop(name)

            # Remove from tags index
            for tag in reg.tags:
                if tag in self._tags_index:
                    self._tags_index[tag].discard(name)
                    if not self._tags_index[tag]:
                        del self._tags_index[tag]

            return True

    async def get(self, name: str) -> BaseAgent | None:
        """
        Get an agent by name.

        Args:
            name: Agent name.

        Returns:
            Agent instance or None if not found.
        """
        async with self._lock:
            reg = self._agents.get(name)
            return reg.agent if reg else None

    async def find_by_tag(self, tag: str) -> list[BaseAgent]:
        """
        Find all agents with a specific tag.

        Args:
            tag: Tag to search for.

        Returns:
            List of matching agent instances.
        """
        async with self._lock:
            names = self._tags_index.get(tag, set())
            return [self._agents[name].agent for name in names if name in self._agents]

    async def find_by_tags(self, tags: list[str], match_all: bool = False) -> list[BaseAgent]:
        """
        Find agents matching tag criteria.

        Args:
            tags: Tags to search for.
            match_all: If True, agents must have all tags. If False, any tag matches.

        Returns:
            List of matching agent instances.
        """
        async with self._lock:
            if match_all:
                matching_names = None
                for tag in tags:
                    tag_names = self._tags_index.get(tag, set())
                    if matching_names is None:
                        matching_names = tag_names
                    else:
                        matching_names = matching_names.intersection(tag_names)
                if matching_names is None:
                    return []
            else:
                matching_names: set[str] = set()
                for tag in tags:
                    matching_names.update(self._tags_index.get(tag, set()))

            return [
                self._agents[name].agent
                for name in matching_names
                if name in self._agents
            ]

    async def list_agents(self) -> list[str]:
        """
        List all registered agent names.

        Returns:
            List of agent names.
        """
        async with self._lock:
            return list(self._agents.keys())

    async def list_tags(self) -> list[str]:
        """
        List all known tags.

        Returns:
            List of tag names.
        """
        async with self._lock:
            return list(self._tags_index.keys())

    async def get_metadata(self, name: str) -> dict[str, Any] | None:
        """
        Get metadata for a registered agent.

        Args:
            name: Agent name.

        Returns:
            Metadata dict or None if not found.
        """
        async with self._lock:
            reg = self._agents.get(name)
            return dict(reg.metadata) if reg else None

    async def update_metadata(self, name: str, metadata: dict[str, Any]) -> bool:
        """
        Update metadata for a registered agent.

        Args:
            name: Agent name.
            metadata: Metadata to merge.

        Returns:
            True if updated, False if agent not found.
        """
        async with self._lock:
            reg = self._agents.get(name)
            if not reg:
                return False
            reg.metadata.update(metadata)
            return True


# Global registry instance
_global_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry