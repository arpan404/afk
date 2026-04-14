"""
MIT License
Copyright (c) 2026 arpan404
See LICENSE file for full license text.

Formal agent contract specification and verification.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..llms.types import JSONValue


class Predicate(Protocol):
    """A callable that returns True if the predicate is satisfied."""

    def __call__(self, context: dict[str, Any]) -> bool: ...


@dataclass(frozen=True, slots=True)
class AgentContract:
    """
    Formal specification of an agent's behavior contract.

    AgentContract allows verification that an agent behaves according to
    a specified contract with preconditions, postconditions, and invariants.

    Attributes:
        name: Contract name for identification.
        description: Human-readable contract description.
        preconditions: Conditions that must be true before execution.
        postconditions: Conditions that must be true after execution.
        invariants: Conditions that must remain true during execution.
        max_execution_time_s: Maximum allowed execution time.
        allowed_tools: Set of tool names this agent is allowed to call.
        forbidden_tools: Set of tool names this agent must never call.
        max_cost_usd: Maximum allowed cost in USD.
        metadata: Additional contract metadata.
    """

    name: str
    description: str = ""
    preconditions: list[Callable[[dict[str, Any]], bool]] = field(default_factory=list)
    postconditions: list[Callable[[dict[str, Any]], bool]] = field(default_factory=list)
    invariants: list[Callable[[dict[str, Any]], bool]] = field(default_factory=list)
    max_execution_time_s: float | None = None
    allowed_tools: list[str] | None = None
    forbidden_tools: list[str] = field(default_factory=list)
    max_cost_usd: float | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContractViolation:
    """
    Record of a contract violation.

    Attributes:
        contract_name: Name of the violated contract.
        violation_type: Type of violation (precondition, postcondition, invariant).
        predicate_name: Name of the predicate that failed.
        message: Human-readable violation message.
        context_snapshot: Context at time of violation.
        timestamp_ms: When violation occurred.
    """

    contract_name: str
    violation_type: str
    predicate_name: str
    message: str
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    timestamp_ms: int = field(default_factory=lambda: int(__import__("time").time() * 1000))


@dataclass
class ContractVerifier:
    """
    Verifies agent behavior against an AgentContract.

    Provides both pre-execution validation and post-execution verification,
    with support for collecting violations for later analysis.
    """

    def __init__(self, contract: AgentContract) -> None:
        self._contract = contract
        self._violations: list[ContractViolation] = []

    def validate_preconditions(self, context: dict[str, Any]) -> list[ContractViolation]:
        """
        Check all preconditions against the given context.

        Args:
            context: Execution context to validate against.

        Returns:
            List of violations found (empty if all pass).
        """
        violations = []
        for pred in self._contract.preconditions:
            pred_name = getattr(pred, "__name__", repr(pred))
            try:
                if not pred(context):
                    violations.append(
                        ContractViolation(
                            contract_name=self._contract.name,
                            violation_type="precondition",
                            predicate_name=pred_name,
                            message=f"Precondition '{pred_name}' was not satisfied",
                            context_snapshot=dict(context),
                        )
                    )
            except Exception as e:
                violations.append(
                    ContractViolation(
                        contract_name=self._contract.name,
                        violation_type="precondition",
                        predicate_name=pred_name,
                        message=f"Precondition '{pred_name}' raised error: {e}",
                        context_snapshot=dict(context),
                    )
                )
        self._violations.extend(violations)
        return violations

    def validate_postconditions(
        self, context: dict[str, Any]
    ) -> list[ContractViolation]:
        """
        Check all postconditions against the given context.

        Args:
            context: Execution context to validate against.

        Returns:
            List of violations found (empty if all pass).
        """
        violations = []
        for pred in self._contract.postconditions:
            pred_name = getattr(pred, "__name__", repr(pred))
            try:
                if not pred(context):
                    violations.append(
                        ContractViolation(
                            contract_name=self._contract.name,
                            violation_type="postcondition",
                            predicate_name=pred_name,
                            message=f"Postcondition '{pred_name}' was not satisfied",
                            context_snapshot=dict(context),
                        )
                    )
            except Exception as e:
                violations.append(
                    ContractViolation(
                        contract_name=self._contract.name,
                        violation_type="postcondition",
                        predicate_name=pred_name,
                        message=f"Postcondition '{pred_name}' raised error: {e}",
                        context_snapshot=dict(context),
                    )
                )
        self._violations.extend(violations)
        return violations

    def validate_invariants(
        self, context: dict[str, Any]
    ) -> list[ContractViolation]:
        """
        Check all invariants against the given context.

        Args:
            context: Execution context to validate against.

        Returns:
            List of violations found (empty if all pass).
        """
        violations = []
        for pred in self._contract.invariants:
            pred_name = getattr(pred, "__name__", repr(pred))
            try:
                if not pred(context):
                    violations.append(
                        ContractViolation(
                            contract_name=self._contract.name,
                            violation_type="invariant",
                            predicate_name=pred_name,
                            message=f"Invariant '{pred_name}' was violated",
                            context_snapshot=dict(context),
                        )
                    )
            except Exception as e:
                violations.append(
                    ContractViolation(
                        contract_name=self._contract.name,
                        violation_type="invariant",
                        predicate_name=pred_name,
                        message=f"Invariant '{pred_name}' raised error: {e}",
                        context_snapshot=dict(context),
                    )
                )
        self._violations.extend(violations)
        return violations

    def validate_tool_allowed(self, tool_name: str) -> bool:
        """
        Check if a tool is allowed by this contract.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if tool is allowed.
        """
        if self._contract.forbidden_tools and tool_name in self._contract.forbidden_tools:
            return False
        if self._contract.allowed_tools is not None:
            return tool_name in self._contract.allowed_tools
        return True

    def get_violations(self) -> list[ContractViolation]:
        """Get all collected violations."""
        return list(self._violations)

    def clear_violations(self) -> None:
        """Clear collected violations."""
        self._violations.clear()


def contract_to_predicate(contract: AgentContract) -> Callable[[dict[str, Any]], bool]:
    """
    Convert an AgentContract into a single predicate function.

    This is useful for composing contracts together or using them
    in policy engines.

    Args:
        contract: The contract to convert.

    Returns:
        A predicate that returns True if all preconditions and invariants pass.
    """

    def predicate(context: dict[str, Any]) -> bool:
        verifier = ContractVerifier(contract)
        pre_violations = verifier.validate_preconditions(context)
        inv_violations = verifier.validate_invariants(context)
        return len(pre_violations) == 0 and len(inv_violations) == 0

    return predicate