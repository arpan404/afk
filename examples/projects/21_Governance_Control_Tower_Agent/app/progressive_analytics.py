"""Multi-pass analytics pipeline to enforce progressive complexity growth."""

from __future__ import annotations

from .config import EXAMPLE_NUMBER


def _default_pass_count() -> int:
    return max(3, EXAMPLE_NUMBER + 1)


def run_progressive_passes(
    rows: list[dict[str, object]],
    *,
    pass_count: int | None = None,
) -> dict[str, object]:
    """Execute pass-by-pass enrichments where depth scales with example id."""
    passes = pass_count or _default_pass_count()
    state: dict[str, object] = {
        "passes": passes,
        "trace": [],
        "risk_score": 0.0,
        "effective_cost": sum(float(row.get("cost", 0.0)) for row in rows),
        "recommended_actions": [],
    }

    for step in range(1, passes + 1):
        step_risk = (step * 0.45) + (len(rows) * 0.21)
        state["risk_score"] = round(float(state["risk_score"]) + step_risk, 4)
        state["effective_cost"] = round(
            float(state["effective_cost"]) + (step * 0.11),
            4,
        )
        trace = list(state["trace"])
        trace.append(
            {
                "step": step,
                "step_risk": round(step_risk, 4),
                "rows": len(rows),
            }
        )
        state["trace"] = trace
        actions = list(state["recommended_actions"])
        actions.append(f"pass_{step}:optimize_operational_bottleneck")
        state["recommended_actions"] = actions

    return state


def complexity_snapshot(
    dataset_summary: dict[str, float | int],
    state: dict[str, object],
) -> dict[str, object]:
    """Compress pipeline state into printable analytics summary."""
    rows = int(dataset_summary.get("rows", 0))
    risk = float(state.get("risk_score", 0.0))
    passes = int(state.get("passes", 0))

    return {
        "dataset_rows": rows,
        "progressive_passes": passes,
        "risk_score": round(risk, 4),
        "risk_density": round(risk / max(1, rows), 4),
        "effective_cost": round(float(state.get("effective_cost", 0.0)), 4),
        "trace_entries": len(list(state.get("trace", []))),
        "action_count": len(list(state.get("recommended_actions", []))),
    }
