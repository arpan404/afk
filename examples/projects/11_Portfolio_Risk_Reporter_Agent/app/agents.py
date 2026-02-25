"""Agent and subagent construction."""

from afk.agents import Agent, FailSafeConfig

from .config import MODEL
from .tools import build_tools


def build_specialists(tier: int) -> list[Agent]:
    """Build specialist subagents for higher tiers."""
    specialists: list[Agent] = []

    if tier >= 2:
        specialists.append(
            Agent(
                name="diagnostics_specialist",
                model=MODEL,
                instructions="Focus on diagnostics, anomaly isolation, and validation checks.",
            )
        )
    if tier >= 3:
        specialists.append(
            Agent(
                name="operations_specialist",
                model=MODEL,
                instructions="Focus on operational mitigations, sequencing, and owner assignment.",
            )
        )
    if tier >= 5:
        specialists.append(
            Agent(
                name="governance_specialist",
                model=MODEL,
                instructions="Focus on controls, auditability, and executive-level risk statements.",
            )
        )

    return specialists


def build_primary_agent(*, tier: int, title: str, domain: str) -> Agent:
    """Build the primary orchestration agent for this example."""
    instructions = f"""
    You are {title}.
    Domain: {domain}.

    Required behavior:
    - Always gather evidence with tools before conclusions.
    - Provide practical recommendations with owners and timelines.
    - Include clear risk language and confidence levels.
    """

    if tier >= 4:
        instructions += "\n- Compare at least two scenario outcomes when relevant."
    if tier >= 5:
        instructions += "\n- Add governance and audit notes suitable for executives."

    return Agent(
        name="primary_orchestrator",
        model=MODEL,
        instructions=instructions,
        tools=build_tools(tier),
        subagents=build_specialists(tier),
        fail_safe=FailSafeConfig(
            max_steps=18 + (tier * 3),
            max_llm_calls=40,
            max_tool_calls=180,
            max_total_cost_usd=2.5 if tier >= 4 else 1.0,
            subagent_failure_policy="retry_then_degrade" if tier >= 3 else "continue",
        ),
    )
