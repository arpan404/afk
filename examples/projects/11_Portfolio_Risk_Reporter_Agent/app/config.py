"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 11
EXAMPLE_TITLE = "Portfolio Risk Reporter Agent"
EXAMPLE_SUMMARY = "Produce portfolio risk summaries with governed analytics instrumentation."
DOMAIN = "portfolio risk management"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 3

DEFAULT_PROMPT = (
    "Run portfolio risk management analysis with concrete recommendations and operational next steps."
)
