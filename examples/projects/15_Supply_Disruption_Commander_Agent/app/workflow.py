"""Tier 4 workflow: sqlite memory + batch orchestration + progressive analytics."""

import asyncio
from pathlib import Path

from afk.core import Runner
from afk.memory import RetentionPolicy, StateRetentionPolicy

from .agents import build_primary_agent
from .analytics import digest_result, summarize_batch
from .batch_ops import build_batch_prompts
from .config import DEFAULT_PROMPT, DOMAIN, EXAMPLE_SUMMARY, EXAMPLE_TITLE, TIER
from .dynamic_dataset import build_dynamic_dataset, summarize_dataset
from .memory_ops import build_sqlite_store
from .progressive_analytics import complexity_snapshot, run_progressive_passes


async def _run() -> None:
    """Run a batch workflow with sqlite memory and progressive analytics."""
    project_root = Path(__file__).resolve().parents[1]
    memory_store = build_sqlite_store(project_root)

    runner = Runner(memory_store=memory_store)
    agent = build_primary_agent(tier=TIER, title=EXAMPLE_TITLE, domain=DOMAIN)

    base_prompt = input("[] > ").strip() or DEFAULT_PROMPT
    prompts = build_batch_prompts(base_prompt)

    dataset = build_dynamic_dataset(base_prompt)
    dataset_summary = summarize_dataset(dataset)
    progressive_state = run_progressive_passes(dataset, pass_count=max(len(prompts), 3))
    progressive_summary = complexity_snapshot(dataset_summary, progressive_state)

    thread_id = f"{EXAMPLE_TITLE.lower().replace(' ', '-')}-batch-thread"
    digests = []
    first_result = None

    for idx, prompt in enumerate(prompts, start=1):
        result = await runner.run(agent, user_message=prompt, thread_id=thread_id)
        if idx == 1:
            first_result = result
        digests.append(digest_result(result))

    if first_result is None:
        raise RuntimeError("No run results captured")

    resumed = await runner.resume(
        agent,
        run_id=first_result.run_id,
        thread_id=thread_id,
    )

    compaction = await runner.compact_thread(
        thread_id=thread_id,
        event_policy=RetentionPolicy(max_age_ms=86_400_000),
        state_policy=StateRetentionPolicy(max_entries=50),
    )

    batch_summary = summarize_batch(digests)

    print(f"[{EXAMPLE_TITLE}] > {resumed.final_text}")
    print("\n--- Analytics ---")
    print(f"summary: {EXAMPLE_SUMMARY}")
    print(f"batch_summary: {batch_summary}")
    print(f"resumed_state: {resumed.state}")
    print(f"compaction_events_removed: {compaction.events_removed}")
    print(f"compaction_states_removed: {compaction.states_removed}")
    print(f"progressive_summary: {progressive_summary}")


def run_example() -> None:
    """Entry point for tier-4 async flow."""
    asyncio.run(_run())
