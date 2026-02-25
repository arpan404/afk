"""Tier 2 workflow: streaming + thread continuity + progressive analytics."""

import asyncio

from afk.core import Runner

from .agents import build_primary_agent
from .analytics import digest_result, summarize_tool_latency
from .config import DEFAULT_PROMPT, DOMAIN, EXAMPLE_SUMMARY, EXAMPLE_TITLE, TIER
from .dynamic_dataset import build_dynamic_dataset, summarize_dataset
from .progressive_analytics import complexity_snapshot, run_progressive_passes
from .streaming_ops import stream_once


async def _run() -> None:
    """Run streaming + follow-up turn with progressive data processing."""
    agent = build_primary_agent(tier=TIER, title=EXAMPLE_TITLE, domain=DOMAIN)
    runner = Runner()

    first_prompt = input("[] > ").strip() or DEFAULT_PROMPT
    thread_id = f"{EXAMPLE_TITLE.lower().replace(' ', '-')}-thread-1"

    dataset = build_dynamic_dataset(first_prompt)
    dataset_summary = summarize_dataset(dataset)
    progressive_state = run_progressive_passes(dataset)
    progressive_summary = complexity_snapshot(dataset_summary, progressive_state)

    print("Assistant: ", end="")
    stream_counters, first_result = await stream_once(
        runner=runner,
        agent=agent,
        prompt=first_prompt,
        thread_id=thread_id,
    )

    follow_up = "Based on your last answer, provide a 3-step operational plan with owners."
    second_result = await runner.run(
        agent,
        user_message=follow_up,
        thread_id=thread_id,
    )

    first_digest = digest_result(first_result)
    second_digest = digest_result(second_result)

    print("\n\n--- Analytics ---")
    print(f"summary: {EXAMPLE_SUMMARY}")
    print(f"stream_counters: {stream_counters}")
    print(f"first_turn_tokens: {first_digest.total_tokens}")
    print(f"second_turn_tokens: {second_digest.total_tokens}")
    print(f"combined_tool_calls: {first_digest.tool_calls + second_digest.tool_calls}")
    print(f"second_turn_tool_latency: {summarize_tool_latency(second_result)}")
    print(f"progressive_summary: {progressive_summary}")


def run_example() -> None:
    """Entry point for tier-2 async flow."""
    asyncio.run(_run())
