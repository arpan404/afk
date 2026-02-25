"""Scenario synthesis for dynamic data-rich demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

from .complexity_config import COMPLEXITY_LEVEL, EXAMPLE_NUMBER


@dataclass
class Scenario:
    """Input workload model used by complexity stages and analytics."""

    title: str
    records: list[dict[str, float | int | str]]
    constraints: dict[str, float]
    seed_label: str


def build_scenario(seed_text: str) -> Scenario:
    """Create deterministic but non-trivial scenario rows from user seed text."""
    base_volume = 80 + (COMPLEXITY_LEVEL * 20)
    rows: list[dict[str, float | int | str]] = []
    for idx in range(1, COMPLEXITY_LEVEL + 4):
        rows.append(
            {
                "segment": f"segment_{idx}",
                "volume": base_volume + (idx * 12),
                "cost": round(2.75 * idx * (1.0 + (COMPLEXITY_LEVEL * 0.08)), 2),
                "sla_hours": max(2.0, 24.0 - (idx * 1.15)),
            }
        )

    return Scenario(
        title=f"Example {EXAMPLE_NUMBER} workload",
        records=rows,
        constraints={
            "max_cost": round(base_volume * 0.9, 2),
            "min_service_level": max(0.65, 0.9 - (COMPLEXITY_LEVEL * 0.01)),
        },
        seed_label=seed_text.strip() or f"default_seed_{EXAMPLE_NUMBER}",
    )
