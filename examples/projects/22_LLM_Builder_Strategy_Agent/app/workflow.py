"""Workflow entry for the LLMBuilder forecast strategy example."""

import asyncio

from .complexity_chain import run_chain
from .metrics import compute_metrics
from .quality import validate_feature_payload, validate_scenario
from .report_builder import build_report
from .scenario import build_scenario
from .llm_ops import forecast_opportunity


DEFAULT_PROMPT = (
    "Enterprise deal renewal: legal delay risk, but strong product adoption and CFO sponsorship. "
    "Produce confidence, risks, and next actions."
)


async def _run() -> None:
    prompt = input("[] > ").strip() or DEFAULT_PROMPT

    scenario = build_scenario(prompt)
    validate_scenario(scenario)
    chain_state = run_chain(scenario)

    try:
        forecast = await forecast_opportunity(prompt)
        feature_payload: dict[str, object] = {
            "kind": "llm_builder_forecast",
            "status": "ok",
            "confidence": forecast.confidence,
            "risk_flags": forecast.risk_flags,
            "next_actions": forecast.next_actions,
        }
    except Exception as exc:  # noqa: BLE001
        feature_payload = {
            "kind": "llm_builder_forecast",
            "status": "error",
            "error": str(exc),
        }

    validate_feature_payload(feature_payload)
    metrics = compute_metrics(scenario, chain_state, feature_payload)
    report = build_report(
        feature_payload=feature_payload,
        chain_state=chain_state,
        metrics=metrics,
    )

    print("[llm-builder] > report")
    print(report)


def run_example() -> None:
    asyncio.run(_run())
