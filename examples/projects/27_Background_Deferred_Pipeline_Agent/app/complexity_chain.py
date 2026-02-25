"""Stage-chain execution that grows with example tier."""

from __future__ import annotations

import importlib

from .complexity_config import STAGE_COUNT
from .scenario import Scenario


def run_chain(scenario: Scenario) -> dict[str, object]:
    """Apply every stage module to progressively enrich runtime state."""
    state: dict[str, object] = {
        "seed": scenario.seed_label,
        "trace": [],
        "risk_score": 0.0,
        "recommended_actions": [],
        "effective_cost": sum(float(row["cost"]) for row in scenario.records),
    }

    for idx in range(1, STAGE_COUNT + 1):
        module = importlib.import_module(f".stages.stage_{idx:02d}", package="app")
        state = module.apply_stage(state, scenario)  # type: ignore[attr-defined]

    return state
