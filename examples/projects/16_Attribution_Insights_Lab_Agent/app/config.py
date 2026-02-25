"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 16
EXAMPLE_TITLE = "Attribution Insights Lab Agent"
EXAMPLE_SUMMARY = "Evaluate attribution scenarios in batch with persistent memory and run summaries."
DOMAIN = "marketing attribution"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 4

DEFAULT_PROMPT = (
    "Run marketing attribution analysis with concrete recommendations and operational next steps."
)
