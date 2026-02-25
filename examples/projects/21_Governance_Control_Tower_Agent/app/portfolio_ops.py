"""Portfolio-level aggregation helpers."""


def summarize_portfolio_cost(costs: list[float | None]) -> dict[str, float]:
    """Summarize cost signals across portfolio runs."""
    cleaned = [value for value in costs if isinstance(value, (int, float))]
    if not cleaned:
        return {"total_cost_usd": 0.0, "avg_cost_usd": 0.0, "max_cost_usd": 0.0}
    total = float(sum(cleaned))
    return {
        "total_cost_usd": round(total, 4),
        "avg_cost_usd": round(total / len(cleaned), 4),
        "max_cost_usd": round(max(cleaned), 4),
    }
