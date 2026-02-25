"""Complexity configuration for progressive example tier 33."""

EXAMPLE_NUMBER = 33
COMPLEXITY_LEVEL = 12
STAGE_COUNT = 34

# Weighted scaling keeps later examples computationally and structurally richer.
RISK_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.2)
LATENCY_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.15)
