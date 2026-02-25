"""Complexity configuration for progressive example tier 28."""

EXAMPLE_NUMBER = 28
COMPLEXITY_LEVEL = 7
STAGE_COUNT = 29

# Weighted scaling keeps later examples computationally and structurally richer.
RISK_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.2)
LATENCY_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.15)
