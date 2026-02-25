"""Governance analytics helpers for tier-5 examples."""


def governance_score(*, completion_rate_pct: float, avg_tokens_per_run: float, policy_gates: int) -> float:
    """Compute a synthetic governance score for comparison across runs."""
    efficiency_component = max(0.0, 100.0 - min(avg_tokens_per_run / 12.0, 45.0))
    control_component = min(policy_gates * 8.0, 24.0)
    return round((completion_rate_pct * 0.55) + (efficiency_component * 0.30) + control_component, 2)
