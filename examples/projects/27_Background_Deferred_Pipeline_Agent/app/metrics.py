"""Cross-cutting analytics computed from scenario, stages, and AFK result payload."""

from __future__ import annotations

from .complexity_config import LATENCY_WEIGHT, RISK_WEIGHT
from .scenario import Scenario


def compute_metrics(
    scenario: Scenario,
    chain_state: dict[str, object],
    feature_payload: dict[str, object],
) -> dict[str, float | int | str]:
    """Produce operational metrics that become richer in later examples."""
    risk_score = float(chain_state.get("risk_score", 0.0)) * RISK_WEIGHT
    cost = float(chain_state.get("effective_cost", 0.0))
    records = len(scenario.records)
    density = round((risk_score / max(1, records)), 4)
    latency_proxy = round((records * LATENCY_WEIGHT) + (risk_score * 0.05), 4)

    return {
        "scenario_title": scenario.title,
        "records": records,
        "risk_score": round(risk_score, 4),
        "risk_density": density,
        "effective_cost": round(cost, 4),
        "latency_proxy_ms": latency_proxy,
        "feature_kind": str(feature_payload.get("kind", "unknown")),
        "feature_status": str(feature_payload.get("status", "unknown")),
    }
