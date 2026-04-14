"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Graceful degradation result types for resilient agent execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import AgentState


@dataclass(frozen=True, slots=True)
class DegradedResult:
    """
    Result returned when agent execution degrades gracefully due to
    hitting limits, circuit breakers, or partial failures.

    Unlike a hard failure, DegradedResult provides partial output with
    an explicit confidence score, allowing callers to make informed
    decisions about whether to use the result.

    Attributes:
        confidence: Confidence score from 0.0 (completely degraded) to 1.0 (fully confident).
        partial_text: Text output produced before degradation.
        partial_structured: Partial structured output if available.
        degradation_reason: Human-readable explanation of what caused degradation.
        degradation_code: Machine-readable degradation code.
        degraded_steps: Number of steps completed before degradation.
        max_steps: Maximum steps that were allowed.
        fallback_used: Whether a fallback response was used.
        original_error: Original error message if degradation resulted from an error.
        recovery_hint: Suggested recovery action for callers.
        metadata: Additional provider-specific metadata.
    """

    confidence: float
    partial_text: str = ""
    partial_structured: dict[str, Any] | None = None
    degradation_reason: str = ""
    degradation_code: str = "unknown"
    degraded_steps: int = 0
    max_steps: int = 20
    fallback_used: bool = False
    original_error: str | None = None
    recovery_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_usable(self, min_confidence: float = 0.5) -> bool:
        """
        Determine if the degraded result is usable at a given confidence threshold.

        Args:
            min_confidence: Minimum acceptable confidence (default 0.5).

        Returns:
            True if confidence >= min_confidence and at least partial text exists.
        """
        return self.confidence >= min_confidence and bool(self.partial_text)

    def to_agent_state(self) -> AgentState:
        """Convert to AgentState, mapping confidence to appropriate state."""
        if self.confidence >= 0.8:
            return "completed"
        elif self.confidence >= 0.5:
            return "degraded"
        else:
            return "failed"


# Degradation codes for machine-readable detection
class DegradationCode:
    # Circuit breaker opened
    CIRCUIT_OPEN = "circuit_open"
    # Max retries exhausted
    MAX_RETRIES = "max_retries"
    # Max steps reached
    MAX_STEPS = "max_steps"
    # Budget exceeded
    BUDGET_EXCEEDED = "budget_exceeded"
    # Timeout
    TIMEOUT = "timeout"
    # LLM error with fallback
    LLM_ERROR_FALLBACK = "llm_error_fallback"
    # Tool failure causing degradation
    TOOL_FAILURE_DEGRADE = "tool_failure_degrade"
    # Subagent failure
    SUBAGENT_FAILURE = "subagent_failure"
    # Memory store unavailable
    MEMORY_UNAVAILABLE = "memory_unavailable"
    # Graceful cancellation
    GRACEFUL_CANCEL = "graceful_cancel"


def make_degraded_from_error(
    error: Exception,
    *,
    partial_text: str = "",
    degraded_steps: int = 0,
    max_steps: int = 20,
    fallback_used: bool = False,
) -> DegradedResult:
    """
    Factory to create a DegradedResult from an exception.

    Args:
        error: The exception that caused degradation.
        partial_text: Text produced before failure.
        degraded_steps: Steps completed.
        max_steps: Max steps allowed.
        fallback_used: Whether fallback was used.

    Returns:
        DegradedResult with error details mapped appropriately.
    """
    error_name = error.__class__.__name__
    degradation_code = DegradationCode.LLM_ERROR_FALLBACK

    if "Circuit" in error_name or "circuit" in str(error).lower():
        degradation_code = DegradationCode.CIRCUIT_OPEN
    elif "timeout" in str(error).lower():
        degradation_code = DegradationCode.TIMEOUT
    elif "budget" in str(error).lower():
        degradation_code = DegradationCode.BUDGET_EXCEEDED

    # Estimate confidence based on how far we got
    confidence = 0.9 * (degraded_steps / max_steps) if max_steps > 0 else 0.0

    return DegradedResult(
        confidence=confidence,
        partial_text=partial_text,
        degradation_reason=f"{error_name}: {str(error)[:200]}",
        degradation_code=degradation_code,
        degraded_steps=degraded_steps,
        max_steps=max_steps,
        fallback_used=fallback_used,
        original_error=str(error)[:500],
        recovery_hint=f"Consider retrying or using a different model/strategy for: {error_name}",
    )


def make_degraded_from_limit(
    limit_type: str,
    *,
    partial_text: str = "",
    degraded_steps: int = 0,
    max_steps: int = 20,
    limit_value: Any = None,
) -> DegradedResult:
    """
    Factory to create a DegradedResult from hitting a limit.

    Args:
        limit_type: Type of limit hit (max_steps, max_cost, timeout, etc.).
        partial_text: Text produced before hitting limit.
        degraded_steps: Steps completed.
        max_steps: Max steps allowed.
        limit_value: The limit value that was reached.

    Returns:
        DegradedResult describing the limit hit.
    """
    codes = {
        "max_steps": DegradationCode.MAX_STEPS,
        "max_cost": DegradationCode.BUDGET_EXCEEDED,
        "timeout": DegradationCode.TIMEOUT,
    }
    code = codes.get(limit_type, DegradationCode.UNKNOWN)
    confidence = 0.85 * (degraded_steps / max_steps) if max_steps > 0 else 0.0

    return DegradedResult(
        confidence=confidence,
        partial_text=partial_text,
        degradation_reason=f"{limit_type.capitalize()} reached: {limit_value}",
        degradation_code=code,
        degraded_steps=degraded_steps,
        max_steps=max_steps,
        fallback_used=False,
        recovery_hint=f"Increase {limit_type} or optimize agent strategy",
    )