"""Tier 3 workflow: policy/telemetry plus progressive analytics."""

import asyncio

from afk.core import Runner, RunnerConfig

from .agents import build_primary_agent
from .analytics import digest_result, summarize_policy_events
from .config import DEFAULT_PROMPT, DOMAIN, EXAMPLE_SUMMARY, EXAMPLE_TITLE, TIER
from .dynamic_dataset import build_dynamic_dataset, summarize_dataset
from .policy_ops import build_policy_engine
from .progressive_analytics import complexity_snapshot, run_progressive_passes
from .telemetry_ops import build_collector, project_metrics


async def _run() -> None:
    """Run governed execution and inspect policy/telemetry/progressive analytics."""
    agent = build_primary_agent(tier=TIER, title=EXAMPLE_TITLE, domain=DOMAIN)
    policy_engine = build_policy_engine()
    collector = build_collector()

    runner = Runner(
        policy_engine=policy_engine,
        telemetry=collector,
        config=RunnerConfig(interaction_mode="headless", approval_fallback="deny"),
    )

    prompt = input("[] > ").strip() or DEFAULT_PROMPT
    dataset = build_dynamic_dataset(prompt)
    dataset_summary = summarize_dataset(dataset)
    progressive_state = run_progressive_passes(dataset)
    progressive_summary = complexity_snapshot(dataset_summary, progressive_state)

    handle = await runner.run_handle(agent, user_message=prompt)

    policy_rows: list[dict] = []
    async for event in handle.events:
        if event.type == "policy_decision":
            policy_rows.append(dict(event.data))

    result = await handle.await_result()
    if result is None:
        raise RuntimeError("Run cancelled before terminal result")

    digest = digest_result(result)
    policy_counts = summarize_policy_events(policy_rows)
    collector_metrics, result_metrics = project_metrics(collector=collector, result=result)

    print(f"[{EXAMPLE_TITLE}] > {result.final_text}")
    print("\n--- Analytics ---")
    print(f"summary: {EXAMPLE_SUMMARY}")
    print(f"state: {digest.state}")
    print(f"tokens: {digest.total_tokens}")
    print(f"policy_counts: {policy_counts}")
    print(f"collector_metrics: {collector_metrics}")
    print(f"result_metrics: {result_metrics}")
    print(f"progressive_summary: {progressive_summary}")


def run_example() -> None:
    """Entry point for tier-3 async flow."""
    asyncio.run(_run())
