from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_ai_agent.runtime.budget_manager import BudgetExceededError, BudgetManager
from local_ai_agent.runtime.loop_detector import LoopDetector
from local_ai_agent.runtime.permission_gate import PermissionGate
from local_ai_agent.runtime.plan_tracker import PlanStepStatus, PlanTracker, PlanTrackingError
from local_ai_agent.runtime.retry_engine import RetryEngine
from local_ai_agent.runtime.tool_router import ToolRouter
from local_ai_agent.runtime.verification_engine import VerificationEngine
from local_ai_agent.schemas.contracts import (
    AgentPlan,
    PlanStep,
    RiskLevel,
    RunBudget,
    ToolResult,
    ToolStatus,
    VerificationResult,
)
from local_ai_agent.tools.registry import ToolDefinition, ToolRegistry


def make_plan(*steps: PlanStep) -> AgentPlan:
    return AgentPlan(
        goal="Validate runtime controls",
        steps=list(steps),
        success_criteria=["The plan progresses safely."],
        rollback_strategy="No writes occur in these tests.",
    )


def make_step(step_id: int, depends_on: list[int] | None = None) -> PlanStep:
    return PlanStep(
        id=step_id,
        action="filesystem.list_directory",
        description=f"Test step {step_id}",
        depends_on=depends_on or [],
        risk=RiskLevel.LOW,
    )


def make_router(
    definition: ToolDefinition,
    *,
    budget: RunBudget | None = None,
    loop_detector: LoopDetector | None = None,
) -> ToolRouter:
    registry = ToolRegistry()
    registry.register(definition)
    return ToolRouter(
        registry=registry,
        permission_gate=PermissionGate(),
        budget_manager=BudgetManager(
            budget or RunBudget(max_tool_calls=5, max_runtime_seconds=60, max_shell_executions=2)
        ),
        loop_detector=loop_detector or LoopDetector(),
        verification_engine=VerificationEngine(),
        retry_engine=RetryEngine(),
    )


def test_plan_tracker_progresses_only_after_dependencies_complete() -> None:
    tracker = PlanTracker(make_plan(make_step(1), make_step(2, [1])))

    first = tracker.activate_next()
    assert first is not None
    assert first.step.id == 1
    assert first.status is PlanStepStatus.ACTIVE
    tracker.mark_completed(1, "Directory listed")

    second = tracker.activate_next()
    assert second is not None
    assert second.step.id == 2
    tracker.mark_completed(2)
    assert tracker.is_complete is True


def test_plan_tracker_rejects_cyclic_dependencies() -> None:
    with pytest.raises(PlanTrackingError, match="must not contain a cycle"):
        PlanTracker(make_plan(make_step(1, [2]), make_step(2, [1])))


def test_budget_manager_reserves_calls_and_enforces_each_limit() -> None:
    current_time = [100.0]
    manager = BudgetManager(
        RunBudget(max_tool_calls=2, max_runtime_seconds=5, max_shell_executions=1),
        clock=lambda: current_time[0],
    )

    first = manager.reserve_tool_call("filesystem.list_directory")
    assert first.tool_calls == 1
    second = manager.reserve_tool_call("shell.execute")
    assert second.shell_executions == 1
    with pytest.raises(BudgetExceededError, match="tool-call"):
        manager.reserve_tool_call("filesystem.read_file")

    current_time[0] = 105.0
    with pytest.raises(BudgetExceededError, match="runtime"):
        manager.reserve_tool_call("filesystem.list_directory")


def test_permission_gate_requires_approval_for_high_and_critical_risk() -> None:
    gate = PermissionGate()
    assert gate.decide(risk=RiskLevel.LOW).allowed is True
    high = gate.decide(risk=RiskLevel.HIGH)
    assert high.allowed is False
    assert high.authorization_required is True
    assert gate.decide(risk=RiskLevel.CRITICAL, authorization_granted=True).allowed is True


def test_retry_engine_applies_error_and_risk_limits() -> None:
    retryable = ToolResult(
        tool_name="filesystem.list_directory",
        status=ToolStatus.ERROR,
        success=False,
        error_code="TRANSIENT",
        error_message="Temporary failure",
        retryable=True,
    )
    engine = RetryEngine()

    low = engine.decide(result=retryable, risk=RiskLevel.LOW, attempts=1)
    assert low.should_retry is True
    assert low.delay_seconds == 2.0
    assert engine.decide(result=retryable, risk=RiskLevel.HIGH, attempts=1).should_retry is False
    assert (
        engine.decide(result=retryable, risk=RiskLevel.CRITICAL, attempts=0).should_retry is False
    )


def test_router_returns_verified_success_with_budget_evidence() -> None:
    async def handler(arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(
            tool_name="filesystem.list_directory",
            status=ToolStatus.SUCCESS,
            success=True,
            data={"path": arguments["path"]},
        )

    async def verifier(arguments: dict[str, Any], result: ToolResult) -> VerificationResult:
        return VerificationResult(
            verified=result.data == {"path": arguments["path"]},
            strategy="result-echo",
            evidence={"path": arguments["path"]},
        )

    router = make_router(
        ToolDefinition(
            name="filesystem.list_directory",
            description="List a directory",
            input_schema={"type": "object"},
            risk=RiskLevel.LOW,
            handler=handler,
            verification=verifier,
            arguments_validator=lambda arguments: {"path": str(arguments["path"])},
        )
    )
    outcome = asyncio.run(
        router.execute(tool_name="filesystem.list_directory", arguments={"path": "src"})
    )

    assert outcome.result.success is True
    assert outcome.result.verified is True
    assert outcome.result.verification is not None
    assert outcome.result.verification.strategy == "result-echo"
    assert outcome.budget is not None
    assert outcome.budget.tool_calls == 1
    assert outcome.retry.should_retry is False


def test_router_blocks_unauthorized_risk_before_budget_or_handler() -> None:
    calls = [0]

    async def handler(_: dict[str, Any]) -> ToolResult:
        calls[0] += 1
        raise AssertionError("The handler must not execute without authorization.")

    router = make_router(
        ToolDefinition(
            name="filesystem.delete_file",
            description="Delete a file",
            input_schema={"type": "object"},
            risk=RiskLevel.HIGH,
            handler=handler,
        )
    )
    outcome = asyncio.run(router.execute(tool_name="filesystem.delete_file", arguments={}))

    assert outcome.result.error_code == "AUTHORIZATION_REQUIRED"
    assert outcome.authorization_required is True
    assert outcome.budget is None
    assert calls == [0]


def test_router_blocks_invalid_arguments_before_budget() -> None:
    async def handler(_: dict[str, Any]) -> ToolResult:
        raise AssertionError("The handler must not receive invalid arguments.")

    router = make_router(
        ToolDefinition(
            name="filesystem.read_file",
            description="Read a file",
            input_schema={"type": "object"},
            risk=RiskLevel.LOW,
            handler=handler,
            arguments_validator=lambda arguments: {"path": arguments["required"]},
        )
    )
    outcome = asyncio.run(router.execute(tool_name="filesystem.read_file", arguments={}))

    assert outcome.result.error_code == "INVALID_INPUT"
    assert outcome.budget is None


def test_router_blocks_repeated_identical_tool_call() -> None:
    async def handler(_: dict[str, Any]) -> ToolResult:
        return ToolResult(
            tool_name="filesystem.list_directory", status=ToolStatus.SUCCESS, success=True
        )

    router = make_router(
        ToolDefinition(
            name="filesystem.list_directory",
            description="List a directory",
            input_schema={"type": "object"},
            risk=RiskLevel.LOW,
            handler=handler,
        ),
        loop_detector=LoopDetector(repeat_threshold=2),
    )
    first = asyncio.run(
        router.execute(tool_name="filesystem.list_directory", arguments={"path": "src"})
    )
    second = asyncio.run(
        router.execute(tool_name="filesystem.list_directory", arguments={"path": "src"})
    )

    assert first.result.error_code is None
    assert second.result.error_code == "LOOP_DETECTED"
    assert second.budget is not None
    assert second.budget.tool_calls == 2


def test_router_returns_retry_decision_after_retryable_handler_failure() -> None:
    async def handler(_: dict[str, Any]) -> ToolResult:
        return ToolResult(
            tool_name="filesystem.list_directory",
            status=ToolStatus.ERROR,
            success=False,
            error_code="TRANSIENT",
            error_message="Temporary filesystem lock",
            retryable=True,
        )

    router = make_router(
        ToolDefinition(
            name="filesystem.list_directory",
            description="List a directory",
            input_schema={"type": "object"},
            risk=RiskLevel.LOW,
            handler=handler,
        )
    )
    outcome = asyncio.run(
        router.execute(tool_name="filesystem.list_directory", arguments={"path": "src"})
    )

    assert outcome.result.error_code == "TRANSIENT"
    assert outcome.result.verified is False
    assert outcome.retry.should_retry is True
    assert outcome.retry.delay_seconds == 2.0
