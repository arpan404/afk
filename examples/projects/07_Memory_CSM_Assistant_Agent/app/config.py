"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 7
EXAMPLE_TITLE = "Memory CSM Assistant Agent"
EXAMPLE_SUMMARY = "Maintain customer success context across turns and summarize continuity metrics."
DOMAIN = "customer success management"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 2

DEFAULT_PROMPT = (
    "Run customer success management analysis with concrete recommendations and operational next steps."
)
