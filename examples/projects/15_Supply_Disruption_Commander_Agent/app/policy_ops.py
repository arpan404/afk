"""Policy configuration for tier-4 examples."""

from afk.agents import PolicyEngine, PolicyRule, PolicyRuleCondition


def build_policy_engine() -> PolicyEngine:
    """Build deterministic policy rules for action-plan gating."""
    return PolicyEngine(
        rules=[
            PolicyRule(
                rule_id="gate-action-plan",
                action="request_approval",
                priority=200,
                reason="Action-plan generation is gated for governed workflows.",
                condition=PolicyRuleCondition(
                    event_type="tool_before_execute",
                    tool_name="build_action_plan",
                ),
            )
        ]
    )
