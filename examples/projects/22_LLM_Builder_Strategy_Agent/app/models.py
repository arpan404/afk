"""Structured output schemas for forecast analysis."""

from pydantic import BaseModel, Field


class OpportunityForecast(BaseModel):
    """Normalized forecast payload used by revenue operations."""

    confidence: float = Field(ge=0.0, le=1.0)
    risk_flags: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
