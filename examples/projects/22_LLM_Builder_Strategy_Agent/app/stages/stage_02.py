"""Complexity stage 2 for example 22."""

from __future__ import annotations

from ._types import RuntimeState


def apply_stage(state: RuntimeState, scenario) -> RuntimeState:
    """Apply stage 2 heuristics and accumulate decisions."""
    trace = list(state.get("trace", []))
    actions = list(state.get("recommended_actions", []))

    incremental_risk = (2 * 0.9) + (len(scenario.records) * 0.15)
    new_risk = float(state.get("risk_score", 0.0)) + incremental_risk

    trace.append(
        {
            "stage": 2,
            "incremental_risk": round(incremental_risk, 4),
            "records": len(scenario.records),
        }
    )
    actions.append(f"stage_2_action:review_capacity")

    state["trace"] = trace
    state["risk_score"] = round(new_risk, 4)
    state["recommended_actions"] = actions
    state["effective_cost"] = round(float(state.get("effective_cost", 0.0)) + (2 * 0.35), 4)
    return state
