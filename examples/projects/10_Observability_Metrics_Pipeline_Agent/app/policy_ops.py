"""Policy configuration for tier-3+ examples."""

from afk.agents import PolicyEngine, PolicyRule, PolicyRuleCondition


def build_policy_engine() -> PolicyEngine:
    """Build deterministic policy rules for mutating tools."""
    return PolicyEngine(
        rules=[
            PolicyRule(
                rule_id="gate-action-plan",
                action="request_approval",
                priority=200,
                reason="Action-plan execution requires explicit approval in governed workflows.",
                condition=PolicyRuleCondition(
                    event_type="tool_before_execute",
                    tool_name="build_action_plan",
                ),
            )
        ]
    )
