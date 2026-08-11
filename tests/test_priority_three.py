from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from local_ai_agent.config import ensure_workspace, load_settings
from local_ai_agent.db.repository import RunRepository
from local_ai_agent.runtime.lifecycle import (
    AuthorizationRequest,
    RunLifecycleService,
    WorkspaceBusyError,
)
from local_ai_agent.runtime.transaction_manager import TransactionError, TransactionManager
from local_ai_agent.schemas.contracts import AgentRun, AgentState, RunBudget
from local_ai_agent.security.command_policy import CommandPolicy, CommandPolicyError
from local_ai_agent.security.output_scrubber import SecretScrubber
from local_ai_agent.tools.mutation import MutationTools


def configured_settings(tmp_path: Path):
    settings = replace(
        load_settings(),
        workspace_root=tmp_path / "workspace",
        sqlite_path=tmp_path / "workspace" / ".agent" / "agent.db",
    )
    ensure_workspace(settings)
    return settings


def create_run(repository: RunRepository, workspace_id: str = "default") -> AgentRun:
    return repository.create_run(
        AgentRun(
            objective="Priority 3 test run",
            workspace_id=workspace_id,
            budget=RunBudget(max_tool_calls=10, max_runtime_seconds=60, max_shell_executions=2),
        )
    )


def transition_to_execute(service: RunLifecycleService, run: AgentRun) -> None:
    service.transition(run.id, AgentState.VALIDATE, "Inputs validated.")
    service.transition(run.id, AgentState.PLAN, "Plan accepted.")
    service.transition(run.id, AgentState.EXECUTE, "Execution started.")


def test_lifecycle_persists_events_cancellation_and_releases_workspace_lock(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    service = RunLifecycleService(repository)
    run = service.register_run(
        AgentRun(
            objective="Lifecycle test",
            workspace_id="shared-project",
            budget=RunBudget(max_tool_calls=5, max_runtime_seconds=60, max_shell_executions=0),
        )
    )

    service.transition(run.id, AgentState.VALIDATE, "Inputs validated.")
    service.request_cancellation(run.id)
    assert repository.cancellation_requested(run.id) is True
    assert service.cancel_if_requested(run.id) is True
    assert repository.get_run(run.id).state is AgentState.CANCELLED
    assert [event.type for event in repository.list_events(run.id)] == [
        "run.created",
        "run.validate",
        "run.cancel_requested",
        "run.cancelled",
    ]

    next_run = service.register_run(
        AgentRun(
            objective="Next lifecycle test",
            workspace_id="shared-project",
            budget=RunBudget(max_tool_calls=5, max_runtime_seconds=60, max_shell_executions=0),
        )
    )
    assert next_run.id != run.id


def test_lifecycle_blocks_concurrent_workspace_owner_and_persists_authorization(
    tmp_path: Path,
) -> None:
    settings = configured_settings(tmp_path)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    service = RunLifecycleService(repository)
    first = service.register_run(
        AgentRun(
            objective="First run",
            workspace_id="shared-project",
            budget=RunBudget(max_tool_calls=5, max_runtime_seconds=60, max_shell_executions=0),
        )
    )
    blocked = AgentRun(
        objective="Blocked run",
        workspace_id="shared-project",
        budget=RunBudget(max_tool_calls=5, max_runtime_seconds=60, max_shell_executions=0),
    )
    with pytest.raises(WorkspaceBusyError):
        service.register_run(blocked)
    assert repository.get_run(blocked.id).state is AgentState.FAILED

    transition_to_execute(service, first)
    waiting = service.require_authorization(
        AuthorizationRequest(
            run_id=first.id,
            tool_name="filesystem.delete_file",
            arguments={"path": "old.txt"},
            risk="HIGH",
        )
    )
    assert waiting.state is AgentState.AUTHORIZATION_REQUIRED
    pending = service.pending_authorization(first.id)
    assert pending is not None
    assert {key: pending[key] for key in ("tool_name", "arguments", "risk")} == {
        "tool_name": "filesystem.delete_file",
        "arguments": {"path": "old.txt"},
        "risk": "HIGH",
    }
    assert pending["action_id"]
    assert pending["checkpoint_id"] is None
    resumed = service.resolve_authorization(first.id, approved=True)
    assert resumed.state is AgentState.EXECUTE
    assert service.pending_authorization(first.id) is None


def test_transactional_mutation_tools_create_backups_and_verify_changes(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    run = create_run(repository)
    target = settings.workspace_project_path / "note.txt"
    target.write_text("before", encoding="utf-8")
    tools = MutationTools(settings=settings, run_id=run.id, repository=repository)

    write_result = asyncio.run(tools.write_file({"path": "note.txt", "content": "after"}))
    write_verification = asyncio.run(tools.verify_write_file({"path": "note.txt"}, write_result))

    assert write_result.success is True
    assert write_verification.verified is True
    assert target.read_text(encoding="utf-8") == "after"
    backup_id = write_result.metadata["backup_id"]
    with repository.connect() as connection:
        row = connection.execute(
            "SELECT backup_path FROM file_backups WHERE id = ?", (backup_id,)
        ).fetchone()
    assert Path(row["backup_path"]).read_text(encoding="utf-8") == "before"

    delete_result = asyncio.run(tools.delete_file({"path": "note.txt"}))
    delete_verification = asyncio.run(tools.verify_delete_file({"path": "note.txt"}, delete_result))
    assert delete_result.success is True
    assert delete_verification.verified is True
    assert not target.exists()


def test_transaction_manager_rolls_back_a_manual_snapshot(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    manager = TransactionManager(settings)
    target = settings.workspace_project_path / "rollback.txt"
    target.write_text("stable", encoding="utf-8")
    run_id = AgentRun(
        objective="Rollback",
        workspace_id="default",
        budget=RunBudget(max_tool_calls=1, max_runtime_seconds=1, max_shell_executions=0),
    ).id

    committed = manager.write_text(run_id=run_id, candidate="rollback.txt", content="changed")
    manager.rollback(committed.snapshot)

    assert target.read_text(encoding="utf-8") == "stable"


def test_command_policy_and_scrubber_block_unsafe_output() -> None:
    policy = CommandPolicy.from_allowlist(("echo", "python3"))
    assert policy.validate("echo safe") == "echo safe"
    with pytest.raises(CommandPolicyError):
        policy.validate("curl https://example.com")
    with pytest.raises(CommandPolicyError):
        policy.validate("echo safe && echo unsafe")

    scrubber = SecretScrubber({"AGENT_API_TOKEN": "super-secret-token"})
    assert scrubber.scrub({"stdout": "token=super-secret-token"}) == {"stdout": "token=[REDACTED]"}


def test_secure_run_runtime_persists_authorization_gated_mutation_and_cancellation(
    tmp_path: Path,
) -> None:
    from local_ai_agent.runtime.secure_run_runtime import build_secure_run_runtime

    settings = configured_settings(tmp_path)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    lifecycle = RunLifecycleService(repository)
    run = lifecycle.register_run(
        AgentRun(
            objective="Secure mutation",
            workspace_id="priority-three",
            budget=RunBudget(max_tool_calls=10, max_runtime_seconds=60, max_shell_executions=2),
        )
    )
    transition_to_execute(lifecycle, run)
    target = settings.workspace_project_path / "obsolete.txt"
    target.write_text("remove me", encoding="utf-8")
    runtime = build_secure_run_runtime(
        settings=settings, run_id=run.id, repository=repository, lifecycle=lifecycle
    )

    blocked = asyncio.run(
        runtime.executor.execute(
            tool_name="filesystem.delete_file", arguments={"path": "obsolete.txt"}
        )
    )
    assert blocked.authorization_required is True
    assert blocked.result.error_code == "AUTHORIZATION_REQUIRED"
    assert target.exists()
    assert repository.get_pending_authorization(run.id)["tool_name"] == "filesystem.delete_file"
    assert repository.get_run(run.id).state is AgentState.AUTHORIZATION_REQUIRED

    lifecycle.resolve_authorization(run.id, approved=True)
    executed = asyncio.run(
        runtime.executor.execute(
            tool_name="filesystem.delete_file",
            arguments={"path": "obsolete.txt"},
            authorization_granted=True,
        )
    )
    assert executed.result.success is True
    assert executed.result.verified is True
    assert not target.exists()
    with repository.connect() as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM tool_calls WHERE run_id = ?", (str(run.id),)
            ).fetchone()[0]
            == 2
        )
        assert connection.execute("SELECT COUNT(*) FROM tool_results").fetchone()[0] == 2

    lifecycle.request_cancellation(run.id)
    cancelled = asyncio.run(
        runtime.executor.execute(
            tool_name="filesystem.write_file", arguments={"path": "new.txt", "content": "no"}
        )
    )
    assert cancelled.result.status.value == "cancelled"
    assert repository.get_run(run.id).state is AgentState.CANCELLED


def test_transaction_manager_restores_existing_file_after_verification_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = configured_settings(tmp_path)
    manager = TransactionManager(settings)
    target = settings.workspace_project_path / "protected.txt"
    target.write_text("original", encoding="utf-8")
    run_id = AgentRun(
        objective="Rollback failure",
        workspace_id="default",
        budget=RunBudget(max_tool_calls=1, max_runtime_seconds=1, max_shell_executions=0),
    ).id
    original_hash = manager._hash_file(target)
    monkeypatch.setattr(manager, "_hash_file", lambda _: "incorrect-hash")

    with pytest.raises(TransactionError, match="rollback was attempted"):
        manager.write_text(run_id=run_id, candidate="protected.txt", content="new content")

    assert target.read_text(encoding="utf-8") == "original"
    assert manager._hash_file(target) == "incorrect-hash"
    assert original_hash != "incorrect-hash"


def test_sandboxed_shell_tool_enforces_command_policy_and_scrubs_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_ai_agent.schemas.contracts import ToolResult, ToolStatus

    settings = configured_settings(tmp_path)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    run = create_run(repository)
    monkeypatch.setenv("AGENT_API_TOKEN", "integration-secret")
    tools = MutationTools(settings=settings, run_id=run.id, repository=repository)
    observed: dict[str, str] = {}

    def fake_execute(*, tool_name: str, command: str, workspace_path: Path) -> ToolResult:
        observed.update(
            {"tool_name": tool_name, "command": command, "workspace": str(workspace_path)}
        )
        return ToolResult(
            tool_name=tool_name,
            status=ToolStatus.SUCCESS,
            success=True,
            data={"exit_code": 0, "stdout": "token=integration-secret", "stderr": ""},
        )

    monkeypatch.setattr(tools._sandbox, "execute", fake_execute)
    success = asyncio.run(tools.execute_shell({"command": "echo safe"}))
    denied = asyncio.run(tools.execute_shell({"command": "curl https://example.com"}))

    assert observed == {
        "tool_name": "shell.execute",
        "command": "echo safe",
        "workspace": str(settings.workspace_project_path),
    }
    assert success.success is True
    assert success.data["stdout"] == "token=[REDACTED]"
    assert denied.success is False
    assert denied.error_code == "POLICY_BLOCK"


def test_api_exposes_durable_lifecycle_controls(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from local_ai_agent.api.app import create_app

    settings = configured_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post(
            "/runs",
            json={"objective": "API lifecycle", "workspace_id": "api-workspace"},
        )
        assert created.status_code == 202
        run_id = created.json()["id"]
        duplicate = client.post(
            "/runs",
            json={"objective": "Blocked", "workspace_id": "api-workspace"},
        )
        assert duplicate.status_code == 409

        listed = client.get("/runs")
        assert listed.status_code == 200
        assert any(run["id"] == run_id for run in listed.json())
        cancel = client.post(f"/runs/{run_id}/cancel")
        assert cancel.status_code == 202
        assert app.state.repository.cancellation_requested(UUID(run_id)) is True
        replies = client.post(f"/runs/{run_id}/reply", json={"message": "stop after current step"})
        assert replies.status_code == 202
        event_types = [event.type for event in app.state.repository.list_events(UUID(run_id))]
        assert "run.created" in event_types
        assert "run.cancel_requested" in event_types
        assert "run.reply_received" in event_types

        authorized_run = app.state.lifecycle.register_run(
            AgentRun(
                objective="API approval",
                workspace_id="approval-workspace",
                budget=RunBudget(max_tool_calls=5, max_runtime_seconds=60, max_shell_executions=0),
            )
        )
        transition_to_execute(app.state.lifecycle, authorized_run)
        app.state.lifecycle.require_authorization(
            AuthorizationRequest(
                run_id=authorized_run.id,
                tool_name="filesystem.delete_file",
                arguments={"path": "x.txt"},
                risk="HIGH",
            )
        )
        pending = client.get(f"/runs/{authorized_run.id}/pending-authorization")
        assert pending.json()["pending"] is True
        approved = client.post(f"/runs/{authorized_run.id}/authorize", json={"approved": True})
        assert approved.status_code == 202
        assert approved.json()["state"] == "EXECUTE"


def test_secure_react_loop_pauses_before_unapproved_high_risk_tool(tmp_path: Path) -> None:
    from local_ai_agent.runtime.secure_run_runtime import build_secure_run_runtime

    class NativeClient:
        async def chat(self, **_: object) -> dict[str, object]:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "filesystem.delete_file",
                                "arguments": {"path": "keep.txt"},
                            }
                        }
                    ],
                }
            }

    settings = configured_settings(tmp_path)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    lifecycle = RunLifecycleService(repository)
    run = lifecycle.register_run(
        AgentRun(
            objective="Request deletion",
            workspace_id="react-authorization",
            budget=RunBudget(max_tool_calls=5, max_runtime_seconds=60, max_shell_executions=0),
        )
    )
    transition_to_execute(lifecycle, run)
    target = settings.workspace_project_path / "keep.txt"
    target.write_text("retain", encoding="utf-8")
    runtime = build_secure_run_runtime(
        settings=settings,
        run_id=run.id,
        repository=repository,
        lifecycle=lifecycle,
        client=NativeClient(),
    )

    result = asyncio.run(
        runtime.react_loop.run(objective="Remove keep.txt", system_prompt="Use tools.")
    )

    assert result.state is AgentState.AUTHORIZATION_REQUIRED
    assert target.read_text(encoding="utf-8") == "retain"
    assert lifecycle.pending_authorization(run.id)["tool_name"] == "filesystem.delete_file"
    assert repository.get_run(run.id).state is AgentState.AUTHORIZATION_REQUIRED
