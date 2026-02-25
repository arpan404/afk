"""Workflow entry for eval suite execution and reporting."""

import asyncio
from pathlib import Path

from afk.agents import Agent
from afk.core import Runner
from afk.evals import (
    EvalCase,
    EvalSuiteConfig,
    FinalTextContainsAssertion,
    StateCompletedAssertion,
    load_eval_cases_json,
    run_suite,
    write_suite_report_json,
)

from .complexity_chain import run_chain
from .llm_stub import EvalStaticLLM
from .metrics import compute_metrics
from .quality import validate_feature_payload, validate_scenario
from .report_builder import build_report
from .scenario import build_scenario


def _build_agent() -> Agent:
    return Agent(model=EvalStaticLLM(), instructions="Produce deterministic eval output.")


async def _run() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / "data" / "cases.json"

    scenario = build_scenario("eval-suite-quality-gates")
    validate_scenario(scenario)
    chain_state = run_chain(scenario)

    cases = load_eval_cases_json(
        dataset_path,
        agent_resolver=lambda name: _build_agent() if name == "ops_eval" else None,
    )
    cases.append(
        EvalCase(
            name="inline-fallback-case",
            agent=_build_agent(),
            user_message="Give me fallback SLA note",
            context={"service": "auth"},
            tags=("inline",),
        )
    )

    suite = run_suite(
        runner_factory=Runner,
        cases=cases,
        config=EvalSuiteConfig(
            execution_mode="sequential",
            assertions=(
                StateCompletedAssertion(),
                FinalTextContainsAssertion("ok:"),
            ),
        ),
    )

    report_path = project_root / "artifacts" / "eval_suite_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_suite_report_json(report_path, suite)

    feature_payload: dict[str, object] = {
        "kind": "eval_suite",
        "status": "ok" if suite.failed == 0 else "error",
        "total": suite.total,
        "passed": suite.passed,
        "failed": suite.failed,
        "report_path": str(report_path),
    }

    validate_feature_payload(feature_payload)
    metrics = compute_metrics(scenario, chain_state, feature_payload)
    report = build_report(
        feature_payload=feature_payload,
        chain_state=chain_state,
        metrics=metrics,
    )

    print("[evals] > report")
    print(report)


def run_example() -> None:
    asyncio.run(_run())
