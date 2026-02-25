"""Render layered report output for each progressive example."""

from __future__ import annotations

from .complexity_config import COMPLEXITY_LEVEL, EXAMPLE_NUMBER, STAGE_COUNT


def build_report(
    *,
    feature_payload: dict[str, object],
    chain_state: dict[str, object],
    metrics: dict[str, float | int | str],
) -> str:
    """Compose a compact but information-dense report block."""
    trace = chain_state.get("trace", [])
    actions = chain_state.get("recommended_actions", [])

    lines = [
        f"example={EXAMPLE_NUMBER} level={COMPLEXITY_LEVEL} stages={STAGE_COUNT}",
        f"feature={metrics['feature_kind']} status={metrics['feature_status']}",
        f"risk_score={metrics['risk_score']} risk_density={metrics['risk_density']}",
        f"effective_cost={metrics['effective_cost']} latency_proxy_ms={metrics['latency_proxy_ms']}",
        f"feature_payload={feature_payload}",
        f"trace_steps={len(trace)} actions={len(actions)}",
    ]
    return "\n".join(lines)
