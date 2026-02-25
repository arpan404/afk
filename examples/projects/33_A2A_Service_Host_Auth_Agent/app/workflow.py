"""Workflow entry for authenticated A2A service host endpoint checks."""

from .complexity_chain import run_chain
from .metrics import compute_metrics
from .quality import validate_feature_payload, validate_scenario
from .report_builder import build_report
from .scenario import build_scenario
from .service import build_service_host


def _payload() -> dict[str, object]:
    return {
        "run_id": "run-33",
        "thread_id": "thread-33",
        "conversation_id": "conv-33",
        "correlation_id": "corr-33",
        "idempotency_key": "idem-33",
        "source_agent": "router",
        "target_agent": "incident-commander",
        "payload": {"severity": "high"},
        "metadata": {},
        "causation_id": "cause-33",
        "timeout_s": 2.0,
    }


def run_example() -> None:
    scenario = build_scenario("a2a-service-host-auth")
    validate_scenario(scenario)
    chain_state = run_chain(scenario)

    try:
        from fastapi.testclient import TestClient

        host = build_service_host()
        app = host.create_app()

        with TestClient(app) as client:
            unauthorized = client.post("/a2a/invoke", json=_payload())
            authorized = client.post(
                "/a2a/invoke",
                json=_payload(),
                headers={"x-api-key": "prod-key"},
            )

        feature_payload: dict[str, object] = {
            "kind": "a2a_service_host",
            "status": "ok" if authorized.status_code == 200 else "error",
            "unauthorized_status": unauthorized.status_code,
            "authorized_status": authorized.status_code,
        }
    except Exception as exc:  # noqa: BLE001
        feature_payload = {
            "kind": "a2a_service_host",
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

    print("[a2a-host] > report")
    print(report)
