"""Complexity configuration for progressive example tier 27."""

EXAMPLE_NUMBER = 27
COMPLEXITY_LEVEL = 6
STAGE_COUNT = 28

# Weighted scaling keeps later examples computationally and structurally richer.
RISK_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.2)
LATENCY_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.15)
