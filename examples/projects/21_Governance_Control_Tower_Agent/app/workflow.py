"""Tier 5 workflow: governance-grade batch + progressive portfolio analytics."""

import asyncio
from pathlib import Path

from afk.core import Runner, RunnerConfig
from afk.observability import RuntimeTelemetryCollector, project_run_metrics_from_result

from .agents import build_primary_agent
from .analytics import digest_result, summarize_batch, summarize_policy_events
from .batch_ops import build_batch_prompts
from .config import DEFAULT_PROMPT, DOMAIN, EXAMPLE_SUMMARY, EXAMPLE_TITLE, TIER
from .dynamic_dataset import build_dynamic_dataset, summarize_dataset
from .governance_ops import governance_score
from .memory_ops import build_sqlite_store
from .policy_ops import build_policy_engine
from .portfolio_ops import summarize_portfolio_cost
from .progressive_analytics import complexity_snapshot, run_progressive_passes


async def _run() -> None:
    """Run governed portfolio workflow with progressive analytics depth."""
    project_root = Path(__file__).resolve().parents[1]
    memory_store = build_sqlite_store(project_root)
    telemetry = RuntimeTelemetryCollector()

    runner = Runner(
        memory_store=memory_store,
        policy_engine=build_policy_engine(),
        telemetry=telemetry,
        config=RunnerConfig(interaction_mode="headless", approval_fallback="deny"),
    )
    agent = build_primary_agent(tier=TIER, title=EXAMPLE_TITLE, domain=DOMAIN)

    base_prompt = input("[] > ").strip() or DEFAULT_PROMPT
    prompts = build_batch_prompts(base_prompt)

    dataset = build_dynamic_dataset(base_prompt)
    dataset_summary = summarize_dataset(dataset)
    progressive_state = run_progressive_passes(dataset, pass_count=max(len(prompts) + 2, 5))
    progressive_summary = complexity_snapshot(dataset_summary, progressive_state)

    digests = []
    policy_events: list[dict] = []
    projected_metrics: list[dict] = []

    for index, prompt in enumerate(prompts, start=1):
        thread_id = f"{EXAMPLE_TITLE.lower().replace(' ', '-')}-portfolio-{index}"
        handle = await runner.run_handle(agent, user_message=prompt, thread_id=thread_id)

        async for event in handle.events:
            if event.type == "policy_decision":
                policy_events.append(dict(event.data))

        result = await handle.await_result()
        if result is None:
            continue

        digests.append(digest_result(result))
        projected_metrics.append(project_run_metrics_from_result(result).to_dict())

    batch_summary = summarize_batch(digests)
    policy_summary = summarize_policy_events(policy_events)
    cost_summary = summarize_portfolio_cost([row.total_cost_usd for row in digests])
    gov_score = governance_score(
        completion_rate_pct=float(batch_summary["completion_rate_pct"]),
        avg_tokens_per_run=float(batch_summary["avg_tokens_per_run"]),
        policy_gates=policy_summary["request_approval"] + policy_summary["deny"],
    )

    print(f"[{EXAMPLE_TITLE}] > Governance-grade workflow complete.")
    print("\n--- Analytics ---")
    print(f"summary: {EXAMPLE_SUMMARY}")
    print(f"batch_summary: {batch_summary}")
    print(f"policy_summary: {policy_summary}")
    print(f"cost_summary: {cost_summary}")
    print(f"governance_score: {gov_score}")
    print(f"projected_metrics: {projected_metrics}")
    print(f"progressive_summary: {progressive_summary}")


def run_example() -> None:
    """Entry point for tier-5 async flow."""
    asyncio.run(_run())
