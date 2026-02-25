"""Complexity configuration for progressive example tier 24."""

EXAMPLE_NUMBER = 24
COMPLEXITY_LEVEL = 3
STAGE_COUNT = 25

# Weighted scaling keeps later examples computationally and structurally richer.
RISK_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.2)
LATENCY_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.15)
