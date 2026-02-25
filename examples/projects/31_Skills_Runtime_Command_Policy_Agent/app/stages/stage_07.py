"""Complexity stage 7 for example 31."""

from __future__ import annotations

from ._types import RuntimeState


def apply_stage(state: RuntimeState, scenario) -> RuntimeState:
    """Apply stage 7 heuristics and accumulate decisions."""
    trace = list(state.get("trace", []))
    actions = list(state.get("recommended_actions", []))

    incremental_risk = (7 * 0.9) + (len(scenario.records) * 0.15)
    new_risk = float(state.get("risk_score", 0.0)) + incremental_risk

    trace.append(
        {
            "stage": 7,
            "incremental_risk": round(incremental_risk, 4),
            "records": len(scenario.records),
        }
    )
    actions.append(f"stage_7_action:review_capacity")

    state["trace"] = trace
    state["risk_score"] = round(new_risk, 4)
    state["recommended_actions"] = actions
    state["effective_cost"] = round(float(state.get("effective_cost", 0.0)) + (7 * 0.35), 4)
    return state
