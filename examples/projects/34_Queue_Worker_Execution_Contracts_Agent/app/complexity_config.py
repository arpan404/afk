"""Complexity configuration for progressive example tier 34."""

EXAMPLE_NUMBER = 34
COMPLEXITY_LEVEL = 13
STAGE_COUNT = 35

# Weighted scaling keeps later examples computationally and structurally richer.
RISK_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.2)
LATENCY_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.15)
