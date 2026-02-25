"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 21
EXAMPLE_TITLE = "Governance Control Tower Agent"
EXAMPLE_SUMMARY = "Operate an enterprise governance control tower with top-tier portfolio analytics."
DOMAIN = "enterprise governance"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 5

DEFAULT_PROMPT = (
    "Run enterprise governance analysis with concrete recommendations and operational next steps."
)
