"""Workflow entry for MCPStore and Agent mcp_servers runtime integration."""

import asyncio

import afk.core.runner.execution as runner_execution
from afk.agents import Agent
from afk.core import Runner
from afk.memory import InMemoryMemoryStore

from .complexity_chain import run_chain
from .llm_stub import MCPToolLLM
from .mcp_store import build_fake_store
from .metrics import compute_metrics
from .quality import validate_feature_payload, validate_scenario
from .report_builder import build_report
from .scenario import build_scenario


async def _run() -> None:
    scenario = build_scenario("mcp-remote-tools")
    validate_scenario(scenario)
    chain_state = run_chain(scenario)

    fake_store = build_fake_store()
    original_get_store = runner_execution.get_mcp_store

    try:
        runner_execution.get_mcp_store = lambda: fake_store
        agent = Agent(
            model=MCPToolLLM(),
            instructions="Use the remote calculator for arithmetic.",
            mcp_servers=["calc=https://fake.example/mcp"],
        )

        result = await Runner(memory_store=InMemoryMemoryStore()).run(
            agent,
            user_message="add 5 and 7",
        )
        feature_payload: dict[str, object] = {
            "kind": "mcp_store",
            "status": "ok",
            "tool_executions": len(result.tool_executions),
            "final_text": result.final_text,
        }
    except Exception as exc:  # noqa: BLE001
        feature_payload = {
            "kind": "mcp_store",
            "status": "error",
            "error": str(exc),
        }
    finally:
        runner_execution.get_mcp_store = original_get_store

    validate_feature_payload(feature_payload)
    metrics = compute_metrics(scenario, chain_state, feature_payload)
    report = build_report(
        feature_payload=feature_payload,
        chain_state=chain_state,
        metrics=metrics,
    )

    print("[mcp] > report")
    print(report)


def run_example() -> None:
    asyncio.run(_run())
