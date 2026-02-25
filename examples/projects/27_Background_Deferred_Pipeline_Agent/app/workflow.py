"""Workflow entry for background/deferred tool orchestration."""

import asyncio

from afk.agents import Agent, FailSafeConfig
from afk.core import Runner, RunnerConfig

from .complexity_chain import run_chain
from .llm_stub import BackgroundAwareLLM
from .metrics import compute_metrics
from .quality import validate_feature_payload, validate_scenario
from .report_builder import build_report
from .scenario import build_scenario
from .tools import compile_report, draft_executive_summary


async def _run() -> None:
    scenario = build_scenario("background-deferred-report")
    validate_scenario(scenario)
    chain_state = run_chain(scenario)

    runner = Runner(
        config=RunnerConfig(
            background_tools_enabled=True,
            background_tool_poll_interval_s=0.01,
            background_tool_result_ttl_s=10.0,
            background_tool_interrupt_on_resolve=True,
            sanitize_tool_output=False,
        )
    )
    agent = Agent(
        model=BackgroundAwareLLM(),
        tools=[compile_report, draft_executive_summary],
        instructions="Coordinate report building with deferred tool completion.",
        fail_safe=FailSafeConfig(max_steps=20),
    )

    stream = await runner.run_stream(agent, user_message="build weekly report")
    seen_events: list[str] = []
    async for event in stream:
        seen_events.append(event.type)

    result = stream.result
    if result is None:
        feature_payload: dict[str, object] = {
            "kind": "background_tools",
            "status": "error",
            "event_count": len(seen_events),
        }
    else:
        feature_payload = {
            "kind": "background_tools",
            "status": "ok",
            "event_count": len(seen_events),
            "tool_executions": len(result.tool_executions),
        }

    validate_feature_payload(feature_payload)
    metrics = compute_metrics(scenario, chain_state, feature_payload)
    report = build_report(
        feature_payload=feature_payload,
        chain_state=chain_state,
        metrics=metrics,
    )

    print("[background] > report")
    print(report)


def run_example() -> None:
    asyncio.run(_run())
