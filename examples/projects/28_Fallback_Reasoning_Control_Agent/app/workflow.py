"""Workflow entry for fallback model chains and reasoning control overrides."""

import asyncio

from afk.agents import Agent, FailSafeConfig
from afk.core import Runner
from afk.llms import LLM

from .complexity_chain import run_chain
from .llm_stub import FailingLLM, ReasoningAwareLLM
from .metrics import compute_metrics
from .quality import validate_feature_payload, validate_scenario
from .report_builder import build_report
from .scenario import build_scenario


def build_resolver() -> dict[str, LLM]:
    return {
        "primary-model": FailingLLM(),
        "backup-model": ReasoningAwareLLM(),
    }


def resolve_model(model: str) -> LLM:
    resolver = build_resolver()
    if model not in resolver:
        raise ValueError(f"unknown model '{model}'")
    return resolver[model]


async def _run() -> None:
    scenario = build_scenario("fallback-reasoning-controls")
    validate_scenario(scenario)
    chain_state = run_chain(scenario)

    agent = Agent(
        model="primary-model",
        model_resolver=resolve_model,
        instructions="Explain reliability status.",
        fail_safe=FailSafeConfig(fallback_model_chain=["backup-model"]),
        reasoning_enabled=True,
        reasoning_effort="low",
        reasoning_max_tokens=192,
    )

    result = await Runner().run(
        agent,
        user_message="Summarize resilience posture.",
        context={
            "_afk": {
                "reasoning": {
                    "enabled": True,
                    "effort": "high",
                    "max_tokens": 512,
                }
            }
        },
    )

    feature_payload: dict[str, object] = {
        "kind": "fallback_reasoning",
        "status": "ok",
        "requested_model": result.requested_model,
        "normalized_model": result.normalized_model,
        "provider_adapter": result.provider_adapter,
    }

    validate_feature_payload(feature_payload)
    metrics = compute_metrics(scenario, chain_state, feature_payload)
    report = build_report(
        feature_payload=feature_payload,
        chain_state=chain_state,
        metrics=metrics,
    )

    print("[fallback] > report")
    print(report)


def run_example() -> None:
    asyncio.run(_run())
