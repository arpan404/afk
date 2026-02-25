"""Quality gates for scenario and feature payload consistency."""

from __future__ import annotations

from .scenario import Scenario


REQUIRED_FEATURE_KEYS = {"kind", "status"}


def validate_scenario(scenario: Scenario) -> None:
    """Fail early when generated scenario is structurally weak."""
    if not scenario.records:
        raise ValueError("scenario must include at least one record")
    if scenario.constraints.get("max_cost", 0.0) <= 0.0:
        raise ValueError("max_cost constraint must be positive")


def validate_feature_payload(payload: dict[str, object]) -> None:
    """Ensure every feature adapter returns comparable typed metadata."""
    missing = REQUIRED_FEATURE_KEYS.difference(payload)
    if missing:
        raise ValueError(f"feature payload missing keys: {sorted(missing)}")
