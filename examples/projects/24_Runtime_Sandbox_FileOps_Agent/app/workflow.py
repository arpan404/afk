"""Workflow entry for runtime tools plus sandbox governance."""

import asyncio
from pathlib import Path

from afk.tools import ToolRegistry, build_runtime_tools

from .complexity_chain import run_chain
from .metrics import compute_metrics
from .quality import validate_feature_payload, validate_scenario
from .report_builder import build_report
from .scenario import build_scenario
from .security import build_policy_and_middleware
from .workspace import ensure_workspace


async def _run() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runtime_root = ensure_workspace(project_root)

    scenario = build_scenario("sandbox-runtime-fileops")
    validate_scenario(scenario)
    chain_state = run_chain(scenario)

    profile, policy, middleware = build_policy_and_middleware(runtime_root)

    registry = ToolRegistry(policy=policy, middlewares=[middleware])
    registry.register_many(build_runtime_tools(root_dir=runtime_root))

    listed = await registry.call("list_directory", {"path": "reports"})
    read_ok = await registry.call(
        "read_file",
        {"path": "reports/q1_pipeline.txt", "max_chars": 1200},
    )
    blocked = await registry.call(
        "read_file",
        {"path": "../../../../etc/passwd", "max_chars": 200},
    )

    feature_payload: dict[str, object] = {
        "kind": "runtime_tools_sandbox",
        "status": "ok" if listed.success and read_ok.success else "error",
        "profile": profile.profile_id,
        "blocked_success": blocked.success,
        "blocked_error": blocked.error_message,
    }

    validate_feature_payload(feature_payload)
    metrics = compute_metrics(scenario, chain_state, feature_payload)
    report = build_report(
        feature_payload=feature_payload,
        chain_state=chain_state,
        metrics=metrics,
    )

    print("[sandbox] > report")
    print(report)


def run_example() -> None:
    asyncio.run(_run())
