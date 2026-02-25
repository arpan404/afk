"""Typed models shared by tools and analytics."""

from dataclasses import dataclass

from pydantic import BaseModel, Field


class SnapshotArgs(BaseModel):
    entity: str = Field(description="Primary entity to analyze (service, account, symbol, or campaign).")


class SignalArgs(BaseModel):
    entity: str = Field(description="Entity for signal computation.")
    baseline: float = Field(ge=0.0, description="Baseline metric value.")
    current: float = Field(ge=0.0, description="Current metric value.")


class PlanArgs(BaseModel):
    entity: str = Field(description="Entity for plan generation.")
    risk_score: int = Field(ge=0, le=100, description="Risk score from previous analysis.")


class ScenarioArgs(BaseModel):
    entity: str = Field(description="Entity for scenario simulation.")
    shock_factor: float = Field(gt=0.0, description="Scenario multiplier.")


class PortfolioArgs(BaseModel):
    entities: list[str] = Field(min_length=1, description="Entity names included in a portfolio.")


@dataclass(slots=True)
class RunDigest:
    """Compact analytics digest for one run."""

    state: str
    total_tokens: int
    tool_calls: int
    subagent_calls: int
    total_cost_usd: float | None
