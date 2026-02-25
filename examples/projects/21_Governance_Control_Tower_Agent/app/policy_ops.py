"""Policy configuration for tier-5 examples."""

from afk.agents import PolicyEngine, PolicyRule, PolicyRuleCondition


def build_policy_engine() -> PolicyEngine:
    """Build deterministic policy rules for governed operations."""
    return PolicyEngine(
        rules=[
            PolicyRule(
                rule_id="gate-action-plan",
                action="request_approval",
                priority=220,
                reason="Action-plan execution requires explicit approval.",
                condition=PolicyRuleCondition(
                    event_type="tool_before_execute",
                    tool_name="build_action_plan",
                ),
            ),
            PolicyRule(
                rule_id="gate-portfolio-aggregation",
                action="request_approval",
                priority=210,
                reason="Portfolio-level aggregation is governance-sensitive.",
                condition=PolicyRuleCondition(
                    event_type="tool_before_execute",
                    tool_name="aggregate_portfolio_risk",
                ),
            ),
        ]
    )
