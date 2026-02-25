"""Workflow entry for task queue and worker contract execution."""

import asyncio

from afk.queues import (
    JOB_DISPATCH_CONTRACT,
    InMemoryTaskQueue,
    TaskWorker,
    TaskWorkerConfig,
)

from .complexity_chain import run_chain
from .contracts import SLAEscalationContract
from .metrics import compute_metrics
from .quality import validate_feature_payload, validate_scenario
from .report_builder import build_report
from .scenario import build_scenario


async def aggregate_metrics(arguments, *, task_item):
    _ = task_item
    values = [int(v) for v in arguments.get("latency_ms", [])]
    if not values:
        return {"p95": 0, "count": 0}
    ordered = sorted(values)
    index = max(0, int(round(0.95 * (len(ordered) - 1))))
    return {
        "p95": ordered[index],
        "count": len(values),
    }


async def _run() -> None:
    scenario = build_scenario("queue-worker-contracts")
    validate_scenario(scenario)
    chain_state = run_chain(scenario)

    queue = InMemoryTaskQueue(retry_backoff_base_s=0.1)

    job_task = await queue.enqueue_contract(
        JOB_DISPATCH_CONTRACT,
        payload={
            "job_type": "aggregate_metrics",
            "arguments": {"latency_ms": [12, 21, 9, 44, 18, 33]},
        },
        agent_name=None,
    )

    custom_task = await queue.enqueue_contract(
        "sla.escalation.v1",
        payload={
            "severity": "critical",
            "owner": "payments-oncall",
        },
        agent_name=None,
    )

    worker = TaskWorker(
        queue,
        agents={},
        job_handlers={"aggregate_metrics": aggregate_metrics},
        execution_contracts={"sla.escalation.v1": SLAEscalationContract()},
        config=TaskWorkerConfig(poll_interval_s=0.05, max_concurrent_tasks=2),
    )

    await worker.start()
    await asyncio.sleep(0.25)
    await worker.shutdown()

    job_row = await queue.get(job_task.id)
    custom_row = await queue.get(custom_task.id)

    feature_payload: dict[str, object] = {
        "kind": "queue_worker_contracts",
        "status": "ok"
        if job_row is not None
        and custom_row is not None
        and job_row.status == "completed"
        and custom_row.status == "completed"
        else "error",
        "job_status": job_row.status if job_row else None,
        "custom_status": custom_row.status if custom_row else None,
    }

    validate_feature_payload(feature_payload)
    metrics = compute_metrics(scenario, chain_state, feature_payload)
    report = build_report(
        feature_payload=feature_payload,
        chain_state=chain_state,
        metrics=metrics,
    )

    print("[queue] > report")
    print(report)


def run_example() -> None:
    asyncio.run(_run())
