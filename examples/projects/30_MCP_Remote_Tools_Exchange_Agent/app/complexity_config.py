"""Complexity configuration for progressive example tier 30."""

EXAMPLE_NUMBER = 30
COMPLEXITY_LEVEL = 9
STAGE_COUNT = 31

# Weighted scaling keeps later examples computationally and structurally richer.
RISK_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.2)
LATENCY_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.15)
