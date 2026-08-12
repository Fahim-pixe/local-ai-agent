"""Run-bound tool execution with durable audit, cancellation, and authorization integration."""

from __future__ import annotations

from contextlib import suppress
from time import perf_counter
from typing import Any
from uuid import UUID

from local_ai_agent.db.repository import RunRepository
from local_ai_agent.runtime.lifecycle import (
    AuthorizationRequest,
    LifecycleError,
    RunLifecycleService,
)
from local_ai_agent.runtime.retry_engine import RetryDecision
from local_ai_agent.runtime.tool_router import ToolRouter, ToolRoutingOutcome
from local_ai_agent.schemas.contracts import AgentEvent, AgentState, ToolResult, ToolStatus
from local_ai_agent.tools.registry import ToolRegistry


class RunToolExecutor:
    """Bind runtime routing to one durable run ID and its lifecycle controls."""

    def __init__(
        self,
        *,
        run_id: UUID,
        registry: ToolRegistry,
        tool_router: ToolRouter,
        repository: RunRepository,
        lifecycle: RunLifecycleService,
    ) -> None:
        self._run_id = run_id
        self._registry = registry
        self._tool_router = tool_router
        self._repository = repository
        self._lifecycle = lifecycle

    async def execute(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        authorization_granted: bool = False,
        attempts: int = 1,
        checkpoint_id: int | None = None,
    ) -> ToolRoutingOutcome:
        if self._lifecycle.cancel_if_requested(self._run_id):
            return self._cancelled_outcome(tool_name)

        try:
            definition = self._registry.get(tool_name)
            risk = definition.risk.value
        except KeyError:
            definition = None
            risk = "UNKNOWN"
        tool_call_id = self._repository.record_tool_call(
            run_id=self._run_id, tool_name=tool_name, arguments=arguments, risk_level=risk
        )
        started = perf_counter()
        outcome = await self._tool_router.execute(
            tool_name=tool_name,
            arguments=arguments,
            authorization_granted=authorization_granted,
            attempts=attempts,
            checkpoint_id=checkpoint_id,
        )
        duration_ms = round((perf_counter() - started) * 1_000)
        self._repository.record_tool_result(
            tool_call_id=tool_call_id, result=outcome.result, duration_ms=duration_ms
        )
        self._repository.record_event(
            AgentEvent(
                run_id=self._run_id,
                type="tool.completed" if outcome.result.success else "tool.blocked_or_failed",
                state=AgentState.EXECUTE,
                message=f"Tool {tool_name} completed with status {outcome.result.status.value}.",
                data={
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "error_code": outcome.result.error_code,
                    "verified": outcome.result.verified,
                    "duration_ms": duration_ms,
                },
            )
        )
        if outcome.authorization_required and definition is not None:
            with suppress(LifecycleError):
                run = self._repository.get_run(self._run_id)
                if run is None:
                    raise LifecycleError("Cannot persist authorization metadata for a missing run.")
                self._lifecycle.require_authorization(
                    AuthorizationRequest(
                        run_id=self._run_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        risk=definition.risk.value,
                        checkpoint_id=checkpoint_id,
                        recovery_class=definition.recovery_class,
                        recovery_contract_version=definition.recovery_contract_version,
                        operation_key=self._repository.operation_key(
                            tool_name=tool_name,
                            arguments=arguments,
                            workspace_id=run.workspace_id,
                            recovery_contract_version=definition.recovery_contract_version,
                        ),
                        max_dispatch_attempts=1,
                    )
                )
        return outcome

    @staticmethod
    def _cancelled_outcome(tool_name: str) -> ToolRoutingOutcome:
        return ToolRoutingOutcome(
            result=ToolResult(
                tool_name=tool_name,
                status=ToolStatus.CANCELLED,
                success=False,
                error_code="CANCELLED",
                error_message="Run was cancelled before the tool could execute.",
            ),
            retry=RetryDecision(False, None, "Cancelled runs are not retried."),
            budget=None,
        )
