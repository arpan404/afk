"""Dynamic dataset generation for progressive example complexity."""

from __future__ import annotations

from .config import DOMAIN, EXAMPLE_NUMBER, EXAMPLE_TITLE


def build_dynamic_dataset(seed_prompt: str) -> list[dict[str, object]]:
    """Build deterministic scenario rows whose size grows with example number."""
    prompt = seed_prompt.strip() or f"default-{EXAMPLE_NUMBER}"
    segment_count = 3 + max(0, EXAMPLE_NUMBER - 2)
    base_volume = 40 + (EXAMPLE_NUMBER * 7)

    rows: list[dict[str, object]] = []
    for idx in range(1, segment_count + 1):
        rows.append(
            {
                "segment": f"segment_{idx}",
                "domain": DOMAIN,
                "title": EXAMPLE_TITLE,
                "seed": prompt,
                "volume": base_volume + (idx * 5),
                "cost": round((idx * 1.4) + (EXAMPLE_NUMBER * 0.7), 4),
                "latency_ms": round(35 + (idx * 2.3) + (EXAMPLE_NUMBER * 1.1), 4),
                "quality_score": round(max(0.3, 0.97 - (idx * 0.015)), 4),
            }
        )
    return rows


def summarize_dataset(rows: list[dict[str, object]]) -> dict[str, float | int]:
    """Return baseline aggregates for dynamic dataset rows."""
    if not rows:
        return {
            "rows": 0,
            "total_volume": 0,
            "avg_cost": 0.0,
            "avg_latency_ms": 0.0,
        }

    total_volume = sum(int(row.get("volume", 0)) for row in rows)
    total_cost = sum(float(row.get("cost", 0.0)) for row in rows)
    total_latency = sum(float(row.get("latency_ms", 0.0)) for row in rows)
    count = len(rows)

    return {
        "rows": count,
        "total_volume": total_volume,
        "avg_cost": round(total_cost / count, 4),
        "avg_latency_ms": round(total_latency / count, 4),
    }
