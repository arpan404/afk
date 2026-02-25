"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 15
EXAMPLE_TITLE = "Supply Disruption Commander Agent"
EXAMPLE_SUMMARY = "Analyze disruption scenarios with multi-run orchestration and memory analytics."
DOMAIN = "supply chain risk"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 4

DEFAULT_PROMPT = (
    "Run supply chain risk analysis with concrete recommendations and operational next steps."
)
