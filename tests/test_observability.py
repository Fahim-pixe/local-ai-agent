"""Tests for local, durable, privacy-safe operational observability."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from local_ai_agent.api.app import create_app
from local_ai_agent.config import Settings, ensure_workspace, load_settings
from local_ai_agent.db.repository import RunRepository
from local_ai_agent.schemas.contracts import (
    AgentEvent,
    AgentRun,
    AgentState,
    RunBudget,
    ToolResult,
    ToolStatus,
)


def configured_settings(tmp_path: Path) -> Settings:
    settings = replace(
        load_settings(),
        workspace_root=tmp_path / "workspace",
        sqlite_path=tmp_path / "workspace" / ".agent" / "agent.db",
        agent_api_token="metrics-token",
    )
    ensure_workspace(settings)
    return settings


def create_run(repository: RunRepository, state: AgentState) -> AgentRun:
    return repository.create_run(
        AgentRun(
            objective=f"Create a {state.value} run for aggregate metrics.",
            workspace_id=f"metrics-{state.value.lower()}",
            state=state,
            budget=RunBudget(max_tool_calls=10, max_runtime_seconds=60, max_shell_executions=1),
        )
    )


def record_tool_result(
    repository: RunRepository,
    run: AgentRun,
    *,
    success: bool,
    verified: bool,
    error_code: str | None = None,
) -> None:
    tool_call_id = repository.record_tool_call(
        run_id=run.id,
        tool_name="filesystem.list_directory",
        arguments={"path": "."},
        risk_level="LOW",
    )
    repository.record_tool_result(
        tool_call_id=tool_call_id,
        result=ToolResult(
            tool_name="filesystem.list_directory",
            status=ToolStatus.SUCCESS if success else ToolStatus.PARTIAL,
            success=success,
            verified=verified,
            error_code=error_code,
        ),
    )


def test_operational_metrics_are_durable_aggregates_and_require_a_token(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    complete_run = create_run(repository, AgentState.COMPLETE)
    failed_run = create_run(repository, AgentState.FAILED)

    record_tool_result(repository, complete_run, success=True, verified=True)
    record_tool_result(
        repository, failed_run, success=False, verified=False, error_code="LOOP_DETECTED"
    )
    record_tool_result(
        repository, failed_run, success=False, verified=False, error_code="MAX_TOOL_CALLS"
    )

    executed = repository.create_pending_action(
        run_id=complete_run.id,
        tool_name="filesystem.write_file",
        arguments={"path": "note.txt", "content": "safe"},
        risk_level="HIGH",
        checkpoint_id=None,
    )
    assert repository.approve_pending_action(complete_run.id) is not None
    claimed = repository.claim_approved_action(
        complete_run.id,
        now=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert claimed is not None
    repository.finish_pending_action(executed.id, succeeded=True)

    rejected = repository.create_pending_action(
        run_id=failed_run.id,
        tool_name="filesystem.delete_file",
        arguments={"path": "old.txt"},
        risk_level="HIGH",
        checkpoint_id=None,
    )
    repository.reject_pending_action(failed_run.id)
    assert repository.get_action(rejected.id).status == "REJECTED"  # type: ignore[union-attr]

    recovered = repository.create_pending_action(
        run_id=failed_run.id,
        tool_name="filesystem.write_file",
        arguments={"path": "retry.txt", "content": "never retry"},
        risk_level="HIGH",
        checkpoint_id=None,
    )
    assert repository.approve_pending_action(failed_run.id) is not None
    assert repository.claim_approved_action(
        failed_run.id,
        now=datetime(2025, 1, 1, tzinfo=UTC),
    )
    recovered_actions = repository.recover_stale_executing_actions(
        now=datetime(2025, 1, 2, tzinfo=UTC),
        lease_seconds=1,
    )
    assert [action.id for action in recovered_actions] == [recovered.id]

    repository.record_event(
        AgentEvent(
            run_id=complete_run.id,
            type="continuation.replayed",
            state=AgentState.COMPLETE,
            message="Durable conversation replayed.",
        )
    )

    with TestClient(create_app(settings)) as client:
        denied = client.get("/metrics/operational")
        assert denied.status_code == 401
        response = client.get(
            "/metrics/operational",
            headers={"Authorization": "Bearer metrics-token"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["runs_total"] == 2
    assert payload["runs_by_state"] == {"COMPLETE": 1, "FAILED": 1}
    assert payload["tool_calls_total"] == 3
    assert payload["tool_results_total"] == 3
    assert payload["verified_tool_successes"] == 1
    assert payload["loop_stops"] == 1
    assert payload["budget_stops"] == 1
    assert payload["authorization_requests"] == 3
    assert payload["authorization_approved"] == 2
    assert payload["authorization_denied"] == 1
    assert payload["authorization_executed"] == 1
    assert payload["action_recoveries"] == 1
    assert payload["continuations_replayed"] == 1
    assert "objective" not in str(payload)
    assert "arguments" not in str(payload)
