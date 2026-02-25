"""Complexity configuration for progressive example tier 29."""

EXAMPLE_NUMBER = 29
COMPLEXITY_LEVEL = 8
STAGE_COUNT = 30

# Weighted scaling keeps later examples computationally and structurally richer.
RISK_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.2)
LATENCY_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.15)
