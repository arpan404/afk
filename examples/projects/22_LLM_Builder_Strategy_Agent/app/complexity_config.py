"""Complexity configuration for progressive example tier 22."""

EXAMPLE_NUMBER = 22
COMPLEXITY_LEVEL = 1
STAGE_COUNT = 23

# Weighted scaling keeps later examples computationally and structurally richer.
RISK_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.2)
LATENCY_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.15)
