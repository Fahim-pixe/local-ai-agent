from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID

import pytest

from local_ai_agent.config import ensure_workspace, load_settings
from local_ai_agent.db.repository import RunRepository
from local_ai_agent.runtime.continuation import ContinuationError
from local_ai_agent.runtime.lifecycle import RunLifecycleService
from local_ai_agent.runtime.secure_run_runtime import build_secure_run_runtime
from local_ai_agent.schemas.contracts import AgentRun, AgentState, RunBudget


def configured_settings(tmp_path: Path):
    settings = replace(
        load_settings(),
        workspace_root=tmp_path / "workspace",
        sqlite_path=tmp_path / "workspace" / ".agent" / "agent.db",
        agent_api_token=None,
    )
    ensure_workspace(settings)
    return settings


def transition_to_execute(service: RunLifecycleService, run: AgentRun) -> None:
    service.transition(run.id, AgentState.VALIDATE, "Validated.")
    service.transition(run.id, AgentState.PLAN, "Planned.")
    service.transition(run.id, AgentState.EXECUTE, "Executing.")


class SequencedNativeClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, **_: object) -> dict[str, object]:
        self.calls += 1
        if self.calls == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "filesystem.delete_file",
                                "arguments": {"path": "obsolete.txt"},
                            }
                        }
                    ],
                }
            }
        return {"message": {"role": "assistant", "content": "Deletion completed and verified."}}


def build_checkpointed_runtime(tmp_path: Path):
    settings = configured_settings(tmp_path)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    lifecycle = RunLifecycleService(repository)
    run = lifecycle.register_run(
        AgentRun(
            objective="Remove the obsolete file.",
            workspace_id="continuation-workspace",
            budget=RunBudget(max_tool_calls=5, max_runtime_seconds=60, max_shell_executions=0),
        )
    )
    transition_to_execute(lifecycle, run)
    target = settings.workspace_project_path / "obsolete.txt"
    target.write_text("remove me", encoding="utf-8")
    client = SequencedNativeClient()
    runtime = build_secure_run_runtime(
        settings=settings,
        run_id=run.id,
        repository=repository,
        lifecycle=lifecycle,
        client=client,
    )
    return repository, lifecycle, run, target, client, runtime


def test_checkpointed_authorization_replays_approved_action_exactly_once(tmp_path: Path) -> None:
    repository, lifecycle, run, target, client, runtime = build_checkpointed_runtime(tmp_path)

    paused = asyncio.run(runtime.run_with_context(system_prompt="Use tools."))
    pending = repository.get_pending_action(run.id)

    assert paused.state is AgentState.AUTHORIZATION_REQUIRED
    assert target.exists()
    assert pending is not None
    assert pending.status == "PENDING"
    assert pending.checkpoint_id is not None
    checkpoint = repository.get_react_checkpoint(pending.checkpoint_id)
    assert checkpoint is not None
    assert checkpoint.phase == "assistant-response"
    assert checkpoint.messages[-1]["role"] == "assistant"

    lifecycle.resolve_authorization(run.id, approved=True)
    resumed = asyncio.run(runtime.continuation.resume_approved_action(system_prompt="Use tools."))

    assert resumed.action_outcome.result.success is True
    assert resumed.action_outcome.result.verified is True
    assert resumed.react_result.state is AgentState.COMPLETE
    assert resumed.react_result.final_response == "Deletion completed and verified."
    assert not target.exists()
    with repository.connect() as connection:
        action_status = connection.execute(
            "SELECT status FROM pending_actions WHERE id = ?", (str(resumed.action.id),)
        ).fetchone()["status"]
        tool_calls = connection.execute(
            "SELECT COUNT(*) FROM tool_calls WHERE run_id = ?", (str(run.id),)
        ).fetchone()[0]
    assert action_status == "EXECUTED"
    assert tool_calls == 2
    assert client.calls == 2
    with pytest.raises(ContinuationError, match="No approved pending action"):
        asyncio.run(runtime.continuation.resume_approved_action(system_prompt="Use tools."))
    assert client.calls == 2


def test_cancellation_prevents_claiming_an_already_approved_action(tmp_path: Path) -> None:
    repository, lifecycle, run, target, client, runtime = build_checkpointed_runtime(tmp_path)
    paused = asyncio.run(runtime.run_with_context(system_prompt="Use tools."))
    assert paused.state is AgentState.AUTHORIZATION_REQUIRED
    lifecycle.resolve_authorization(run.id, approved=True)
    lifecycle.request_cancellation(run.id)

    with pytest.raises(ContinuationError, match="cancelled before an approved action"):
        asyncio.run(runtime.continuation.resume_approved_action(system_prompt="Use tools."))

    assert target.exists()
    assert repository.get_run(run.id).state is AgentState.CANCELLED
    assert repository.get_pending_action(run.id).status == "APPROVED"
    assert client.calls == 1


def test_api_continuation_executes_only_an_approved_checkpointed_action(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from local_ai_agent.api.app import create_app

    settings = configured_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post(
            "/runs",
            json={"objective": "Remove obsolete file.", "workspace_id": "api-continuation"},
        )
        assert created.status_code == 202
        run_id = created.json()["id"]
        run = app.state.repository.get_run(UUID(run_id))
        transition_to_execute(app.state.lifecycle, run)
        target = settings.workspace_project_path / "obsolete.txt"
        target.write_text("remove me", encoding="utf-8")
        native_client = SequencedNativeClient()
        runtime = build_secure_run_runtime(
            settings=settings,
            run_id=run.id,
            repository=app.state.repository,
            lifecycle=app.state.lifecycle,
            client=native_client,
        )
        app.state.runtime_builder = lambda **_: runtime
        paused = asyncio.run(runtime.run_with_context(system_prompt="Use tools."))
        assert paused.state is AgentState.AUTHORIZATION_REQUIRED

        approved = client.post(f"/runs/{run_id}/authorize", json={"approved": True})
        assert approved.status_code == 202
        continued = client.post(f"/runs/{run_id}/continue")

    assert continued.status_code == 202
    assert continued.json()["action_verified"] is True
    assert continued.json()["react_state"] == "COMPLETE"
    assert not target.exists()


def test_api_continuation_rejects_runs_without_an_approved_action(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from local_ai_agent.api.app import create_app

    settings = configured_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post(
            "/runs",
            json={"objective": "No continuation yet.", "workspace_id": "api-no-action"},
        )
        run_id = created.json()["id"]
        resumed = client.post(f"/runs/{run_id}/continue")

    assert resumed.status_code == 409
    assert "No approved pending action" in resumed.json()["detail"]


def test_stale_executing_action_is_failed_once_with_recovery_audit(tmp_path: Path) -> None:
    repository, lifecycle, run, target, _, runtime = build_checkpointed_runtime(tmp_path)
    paused = asyncio.run(runtime.run_with_context(system_prompt="Use tools."))
    assert paused.state is AgentState.AUTHORIZATION_REQUIRED
    lifecycle.resolve_authorization(run.id, approved=True)
    claimed_at = datetime(2026, 8, 12, tzinfo=UTC)
    claimed = repository.claim_approved_action(
        run.id, worker_id="worker-a", lease_seconds=10, now=claimed_at
    )
    assert claimed is not None
    assert claimed.status == "EXECUTING"
    assert claimed.worker_id == "worker-a"
    assert claimed.lease_expires_at == claimed_at + timedelta(seconds=10)

    recovered = lifecycle.recover_stale_executing_actions(
        now=claimed_at + timedelta(seconds=11), reason="WORKER_CRASH_RECOVERY"
    )

    assert [action.id for action in recovered] == [claimed.id]
    assert target.exists()
    with repository.connect() as connection:
        row = connection.execute(
            "SELECT status, recovery_reason FROM pending_actions WHERE id = ?", (str(claimed.id),)
        ).fetchone()
    assert row["status"] == "FAILED"
    assert row["recovery_reason"] == "WORKER_CRASH_RECOVERY"
    assert "continuation.action_recovered" in [
        event.type for event in repository.list_events(run.id)
    ]
    with pytest.raises(ContinuationError, match="No approved pending action"):
        asyncio.run(runtime.continuation.resume_approved_action(system_prompt="Use tools."))


def test_active_worker_lease_is_not_recovered_before_expiry(tmp_path: Path) -> None:
    repository, lifecycle, run, _, _, runtime = build_checkpointed_runtime(tmp_path)
    paused = asyncio.run(runtime.run_with_context(system_prompt="Use tools."))
    assert paused.state is AgentState.AUTHORIZATION_REQUIRED
    lifecycle.resolve_authorization(run.id, approved=True)
    claimed_at = datetime(2026, 8, 12, tzinfo=UTC)
    claimed = repository.claim_approved_action(
        run.id, worker_id="worker-a", lease_seconds=10, now=claimed_at
    )
    assert claimed is not None

    recovered = lifecycle.recover_stale_executing_actions(
        now=claimed_at + timedelta(seconds=9), reason="WORKER_CRASH_RECOVERY"
    )

    assert recovered == []
    current = repository.get_pending_action(run.id)
    assert current is not None
    assert current.status == "EXECUTING"
    assert current.worker_id == "worker-a"


def test_api_resume_token_validates_before_continuing_an_approved_action(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from local_ai_agent.api.app import create_app

    settings = configured_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post(
            "/runs",
            json={"objective": "Remove obsolete file.", "workspace_id": "api-resume-token"},
        )
        assert created.status_code == 202
        payload = created.json()
        run = app.state.repository.get_run(UUID(payload["id"]))
        assert run is not None
        transition_to_execute(app.state.lifecycle, run)
        target = settings.workspace_project_path / "obsolete.txt"
        target.write_text("remove me", encoding="utf-8")
        native_client = SequencedNativeClient()
        runtime = build_secure_run_runtime(
            settings=settings,
            run_id=run.id,
            repository=app.state.repository,
            lifecycle=app.state.lifecycle,
            client=native_client,
        )
        app.state.runtime_builder = lambda **_: runtime
        paused = asyncio.run(runtime.run_with_context(system_prompt="Use tools."))
        assert paused.state is AgentState.AUTHORIZATION_REQUIRED
        approved = client.post(f"/runs/{run.id}/authorize", json={"approved": True})
        assert approved.status_code == 202

        rejected = client.post(
            f"/runs/{run.id}/resume", json={"resume_token": "00000000-0000-0000-0000-000000000000"}
        )
        resumed = client.post(
            f"/runs/{run.id}/resume", json={"resume_token": payload["resume_token"]}
        )

    assert rejected.status_code == 403
    assert resumed.status_code == 202
    assert resumed.json()["action_verified"] is True
    assert resumed.json()["react_state"] == "COMPLETE"
    assert not target.exists()


def test_api_exposes_durable_action_history(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from local_ai_agent.api.app import create_app

    settings = configured_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post(
            "/runs",
            json={"objective": "Inspect action history.", "workspace_id": "api-action-history"},
        )
        assert created.status_code == 202
        actions = client.get(f"/runs/{created.json()['id']}/actions")

    assert actions.status_code == 200
    assert actions.json() == {"actions": []}
