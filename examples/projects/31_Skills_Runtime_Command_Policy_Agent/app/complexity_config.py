"""Complexity configuration for progressive example tier 31."""

EXAMPLE_NUMBER = 31
COMPLEXITY_LEVEL = 10
STAGE_COUNT = 32

# Weighted scaling keeps later examples computationally and structurally richer.
RISK_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.2)
LATENCY_WEIGHT = 1.0 + (COMPLEXITY_LEVEL * 0.15)
