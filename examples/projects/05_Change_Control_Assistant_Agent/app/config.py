"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 5
EXAMPLE_TITLE = "Change Control Assistant Agent"
EXAMPLE_SUMMARY = "Model governed change workflows with policy-aware operational analysis."
DOMAIN = "change management"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 1

DEFAULT_PROMPT = (
    "Run change management analysis with concrete recommendations and operational next steps."
)
