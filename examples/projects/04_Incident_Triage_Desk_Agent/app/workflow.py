"""Tier 1 workflow: sync execution with run-level + progressive analytics."""

from afk.core import Runner

from .agents import build_primary_agent
from .analytics import digest_result, summarize_tool_latency
from .config import DEFAULT_PROMPT, DOMAIN, EXAMPLE_SUMMARY, EXAMPLE_TITLE, TIER
from .dynamic_dataset import build_dynamic_dataset, summarize_dataset
from .progressive_analytics import complexity_snapshot, run_progressive_passes


def run_example() -> None:
    """Run a single sync analysis turn and print layered analytics."""
    agent = build_primary_agent(tier=TIER, title=EXAMPLE_TITLE, domain=DOMAIN)
    runner = Runner()

    user_input = input("[] > ").strip() or DEFAULT_PROMPT
    dataset = build_dynamic_dataset(user_input)
    dataset_summary = summarize_dataset(dataset)
    progressive_state = run_progressive_passes(dataset)
    progressive_summary = complexity_snapshot(dataset_summary, progressive_state)

    result = runner.run_sync(agent, user_message=user_input)

    digest = digest_result(result)
    tool_latency = summarize_tool_latency(result)

    print(f"[{EXAMPLE_TITLE}] > {result.final_text}")
    print("\n--- Analytics ---")
    print(f"summary: {EXAMPLE_SUMMARY}")
    print(f"state: {digest.state}")
    print(f"total_tokens: {digest.total_tokens}")
    print(f"tool_calls: {digest.tool_calls}")
    print(f"subagent_calls: {digest.subagent_calls}")
    print(f"total_cost_usd: {digest.total_cost_usd}")
    print(f"tool_latency: {tool_latency}")
    print(f"progressive_summary: {progressive_summary}")
