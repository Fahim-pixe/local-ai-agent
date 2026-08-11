"""Runtime-owned tool routing pipeline.

The router is the only path from an approved model tool request to a registered
handler. It enforces the pre-execution controls in a fixed order and returns
structured evidence for persistence by the higher-level run loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from local_ai_agent.runtime.budget_manager import BudgetExceededError, BudgetManager, BudgetSnapshot
from local_ai_agent.runtime.loop_detector import LoopDetector
from local_ai_agent.runtime.permission_gate import PermissionGate
from local_ai_agent.runtime.retry_engine import RetryDecision, RetryEngine
from local_ai_agent.runtime.verification_engine import VerificationEngine
from local_ai_agent.schemas.contracts import RiskLevel, ToolResult, ToolStatus
from local_ai_agent.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class ToolRoutingOutcome:
    result: ToolResult
    retry: RetryDecision
    budget: BudgetSnapshot | None
    authorization_required: bool = False


class ToolRouter:
    """Execute registered tools only after all runtime controls approve the call."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        permission_gate: PermissionGate,
        budget_manager: BudgetManager,
        loop_detector: LoopDetector,
        verification_engine: VerificationEngine,
        retry_engine: RetryEngine,
    ) -> None:
        self._registry = registry
        self._permission_gate = permission_gate
        self._budget_manager = budget_manager
        self._loop_detector = loop_detector
        self._verification_engine = verification_engine
        self._retry_engine = retry_engine

    async def execute(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        authorization_granted: bool = False,
        attempts: int = 1,
        checkpoint_id: int | None = None,
    ) -> ToolRoutingOutcome:
        """Route one requested operation through the fixed execution-control pipeline."""
        try:
            definition = self._registry.get(tool_name)
        except KeyError:
            return self._outcome(
                ToolResult(
                    tool_name=tool_name,
                    status=ToolStatus.ERROR,
                    success=False,
                    error_code="UNKNOWN_TOOL",
                    error_message="The requested tool is not registered by runtime.",
                )
            )

        try:
            validated_arguments = definition.validate_arguments(arguments)
        except Exception as error:
            return self._outcome(
                ToolResult(
                    tool_name=tool_name,
                    status=ToolStatus.ERROR,
                    success=False,
                    error_code="INVALID_INPUT",
                    error_message=f"Tool arguments failed runtime validation: {type(error).__name__}.",
                ),
                risk=definition.risk,
                attempts=attempts,
            )

        permission = self._permission_gate.decide(
            risk=definition.risk, authorization_granted=authorization_granted
        )
        if not permission.allowed:
            return self._outcome(
                ToolResult(
                    tool_name=tool_name,
                    status=ToolStatus.PARTIAL,
                    success=False,
                    error_code=permission.error_code,
                    error_message=permission.reason,
                    metadata={"risk": definition.risk.value},
                ),
                risk=definition.risk,
                attempts=attempts,
                authorization_required=permission.authorization_required,
            )

        try:
            budget = self._budget_manager.reserve_tool_call(tool_name)
        except BudgetExceededError as error:
            return self._outcome(
                ToolResult(
                    tool_name=tool_name,
                    status=ToolStatus.PARTIAL,
                    success=False,
                    error_code=error.code,
                    error_message=str(error),
                ),
                risk=definition.risk,
                attempts=attempts,
            )

        if self._loop_detector.record_and_check(tool_name, validated_arguments):
            return self._outcome(
                ToolResult(
                    tool_name=tool_name,
                    status=ToolStatus.PARTIAL,
                    success=False,
                    error_code="LOOP_DETECTED",
                    error_message="Repeated identical tool call was blocked by runtime.",
                    metadata={
                        "arguments_hash": self._loop_detector.arguments_hash(validated_arguments)
                    },
                ),
                risk=definition.risk,
                attempts=attempts,
                budget=budget,
            )

        try:
            executed = await definition.handler(validated_arguments)
        except Exception as error:
            executed = ToolResult(
                tool_name=tool_name,
                status=ToolStatus.ERROR,
                success=False,
                error_code="TOOL_FAILURE",
                error_message=f"Registered tool handler raised {type(error).__name__}.",
                retryable=True,
            )
        verified = await self._verification_engine.verify(
            arguments=validated_arguments, result=executed, verifier=definition.verification
        )
        metadata = {**verified.metadata, "risk": definition.risk.value}
        verified = verified.model_copy(update={"metadata": metadata})
        return self._outcome(verified, risk=definition.risk, attempts=attempts, budget=budget)

    def _outcome(
        self,
        result: ToolResult,
        *,
        risk: RiskLevel | None = None,
        attempts: int = 1,
        budget: BudgetSnapshot | None = None,
        authorization_required: bool = False,
    ) -> ToolRoutingOutcome:
        retry = (
            self._retry_engine.decide(result=result, risk=risk, attempts=attempts)
            if risk is not None
            else RetryDecision(False, None, "No registered risk policy applies.")
        )
        return ToolRoutingOutcome(
            result=result,
            retry=retry,
            budget=budget,
            authorization_required=authorization_required,
        )
