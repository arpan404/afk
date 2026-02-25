"""Scenario configuration for this example."""

EXAMPLE_NUMBER = 4
EXAMPLE_TITLE = "Incident Triage Desk Agent"
EXAMPLE_SUMMARY = "Coordinate incident triage and mitigation recommendations with measurable runtime analytics."
DOMAIN = "incident response"
MODEL = "ollama_chat/gpt-oss:20b"

# Tier controls architecture complexity.
# 1: sync + tools
# 2: + streaming + thread continuity
# 3: + policy + telemetry projection
# 4: + sqlite memory + batch orchestration
# 5: + governance + portfolio aggregation
TIER = 1

DEFAULT_PROMPT = (
    "Run incident response analysis with concrete recommendations and operational next steps."
)
