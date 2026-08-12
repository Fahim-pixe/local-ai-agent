"""Contract tests for local multi-process dispatch primitives."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier
from uuid import UUID

from local_ai_agent.config import ensure_workspace, load_settings
from local_ai_agent.db.repository import RunRepository
from local_ai_agent.runtime.lifecycle import RunLifecycleService
from local_ai_agent.schemas.contracts import (
    AgentRun,
    RecoveryClass,
    RiskLevel,
    RunBudget,
    ToolResult,
)
from local_ai_agent.tools.registry import ToolDefinition


def configured_repository(tmp_path: Path) -> tuple[RunRepository, UUID]:
    settings = replace(
        load_settings(),
        workspace_root=tmp_path / "workspace",
        sqlite_path=tmp_path / "workspace" / ".agent" / "agent.db",
        agent_api_token=None,
    )
    ensure_workspace(settings)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    lifecycle = RunLifecycleService(repository)
    run = lifecycle.register_run(
        AgentRun(
            objective="Dispatch one safe action.",
            workspace_id="dispatch-workspace",
            budget=RunBudget(max_tool_calls=5, max_runtime_seconds=60, max_shell_executions=0),
        )
    )
    return repository, run.id


def approved_action(repository: RunRepository, run_id: UUID) -> UUID:
    action = repository.create_pending_action(
        run_id=run_id,
        tool_name="filesystem.list_directory",
        arguments={"path": "."},
        risk_level="LOW",
        checkpoint_id=None,
        recovery_class=RecoveryClass.NEVER_RECLAIM,
        recovery_contract_version=1,
        operation_key="a" * 64,
        max_dispatch_attempts=1,
    )
    repository.approve_pending_action(run_id)
    return action.id


async def _handler(_: dict[str, object]) -> ToolResult:
    raise AssertionError("The dispatch metadata test must not execute a tool.")


def test_tool_definition_defaults_to_never_reclaim() -> None:
    definition = ToolDefinition(
        name="test.tool",
        description="Test-only runtime contract.",
        input_schema={"type": "object"},
        risk=RiskLevel.LOW,
        handler=_handler,
    )

    assert definition.recovery_class is RecoveryClass.NEVER_RECLAIM
    assert definition.recovery_contract_version == 1


def test_register_worker_and_atomic_claim_persist_attempt_evidence(tmp_path: Path) -> None:
    repository, run_id = configured_repository(tmp_path)
    action_id = approved_action(repository, run_id)
    now = datetime(2026, 8, 12, tzinfo=UTC)
    repository.register_worker(
        worker_id="worker-a",
        hostname="host-a",
        process_id=101,
        capabilities=("filesystem.list_directory",),
        now=now,
    )

    claim = repository.claim_next_dispatchable_action(
        worker_id="worker-a",
        capabilities=("filesystem.list_directory",),
        lease_seconds=30,
        now=now,
    )

    assert claim is not None
    assert claim.action.id == action_id
    assert claim.action.status == "EXECUTING"
    assert claim.action.dispatch_attempt == 1
    assert claim.action.worker_id == "worker-a"
    assert claim.action.recovery_class is RecoveryClass.NEVER_RECLAIM
    attempts = repository.list_action_attempts(action_id)
    assert [(attempt.attempt, attempt.status, attempt.worker_id) for attempt in attempts] == [
        (1, "CLAIMED", "worker-a")
    ]


def test_inactive_or_unknown_worker_cannot_claim_approved_action(tmp_path: Path) -> None:
    repository, run_id = configured_repository(tmp_path)
    approved_action(repository, run_id)
    now = datetime(2026, 8, 12, tzinfo=UTC)
    assert (
        repository.claim_next_dispatchable_action(
            worker_id="missing",
            capabilities=("filesystem.list_directory",),
            lease_seconds=30,
            now=now,
        )
        is None
    )
    repository.register_worker(
        worker_id="worker-a",
        hostname="host-a",
        process_id=101,
        capabilities=("filesystem.list_directory",),
        now=now,
    )
    assert repository.drain_worker("worker-a", now=now)
    assert (
        repository.claim_next_dispatchable_action(
            worker_id="worker-a",
            capabilities=("filesystem.list_directory",),
            lease_seconds=30,
            now=now,
        )
        is None
    )


def test_two_workers_claim_one_action_exactly_once(tmp_path: Path) -> None:
    repository, run_id = configured_repository(tmp_path)
    action_id = approved_action(repository, run_id)
    now = datetime(2026, 8, 12, tzinfo=UTC)
    worker_ids = ("worker-a", "worker-b")
    for index, worker_id in enumerate(worker_ids, start=1):
        repository.register_worker(
            worker_id=worker_id,
            hostname=f"host-{index}",
            process_id=index,
            capabilities=("filesystem.list_directory",),
            now=now,
        )
    barrier = Barrier(2)

    def claim(worker_id: str):
        barrier.wait()
        local = RunRepository(repository.database_path)
        return local.claim_next_dispatchable_action(
            worker_id=worker_id,
            capabilities=("filesystem.list_directory",),
            lease_seconds=30,
            now=now,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(claim, worker_ids))

    winners = [claim for claim in claims if claim is not None]
    assert len(winners) == 1
    assert winners[0].action.id == action_id
    attempts = repository.list_action_attempts(action_id)
    assert len(attempts) == 1
    assert attempts[0].status == "CLAIMED"


def test_operation_key_is_canonical_and_contract_versioned() -> None:
    first = RunRepository.operation_key(
        tool_name="filesystem.list_directory",
        arguments={"a": 1, "b": 2},
        workspace_id="workspace-a",
        recovery_contract_version=1,
    )
    reordered = RunRepository.operation_key(
        tool_name="filesystem.list_directory",
        arguments={"b": 2, "a": 1},
        workspace_id="workspace-a",
        recovery_contract_version=1,
    )
    changed = RunRepository.operation_key(
        tool_name="filesystem.list_directory",
        arguments={"a": 1, "b": 2},
        workspace_id="workspace-a",
        recovery_contract_version=2,
    )

    assert first == reordered
    assert first != changed


def test_worker_and_action_attempt_api_observability(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from local_ai_agent.api.app import create_app

    repository, run_id = configured_repository(tmp_path)
    action_id = approved_action(repository, run_id)
    now = datetime(2026, 8, 12, tzinfo=UTC)
    repository.register_worker(
        worker_id="worker-api",
        hostname="host-api",
        process_id=202,
        capabilities=("filesystem.list_directory",),
        now=now,
    )
    claim = repository.claim_next_dispatchable_action(
        worker_id="worker-api",
        capabilities=("filesystem.list_directory",),
        lease_seconds=30,
        now=now,
    )
    assert claim is not None

    settings = replace(
        load_settings(),
        workspace_root=tmp_path / "workspace",
        sqlite_path=tmp_path / "workspace" / ".agent" / "agent.db",
        agent_api_token=None,
    )
    app = create_app(settings)
    with TestClient(app) as client:
        workers = client.get("/workers")
        attempts = client.get(f"/actions/{action_id}/attempts")
        drained = client.post("/workers/worker-api/drain")

    assert workers.status_code == 200
    assert workers.json()["workers"][0]["worker_id"] == "worker-api"
    assert attempts.status_code == 200
    assert attempts.json()["attempts"][0]["status"] == "CLAIMED"
    assert drained.status_code == 200
    assert drained.json() == {"worker_id": "worker-api", "state": "DRAINING"}


def test_dispatch_worker_executes_only_its_atomic_claim_and_fails_closed(tmp_path: Path) -> None:
    import asyncio

    from local_ai_agent.runtime.secure_run_runtime import build_secure_run_runtime
    from local_ai_agent.runtime.worker_dispatch import LocalDispatchWorker

    settings = replace(
        load_settings(),
        workspace_root=tmp_path / "workspace",
        sqlite_path=tmp_path / "workspace" / ".agent" / "agent.db",
        agent_api_token=None,
        dispatch_enabled=True,
    )
    ensure_workspace(settings)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    lifecycle = RunLifecycleService(repository)
    run = lifecycle.register_run(
        AgentRun(
            objective="Dispatch a checkpointless action safely.",
            workspace_id="dispatch-worker",
            budget=RunBudget(max_tool_calls=5, max_runtime_seconds=60, max_shell_executions=0),
        )
    )
    action = repository.create_pending_action(
        run_id=run.id,
        tool_name="filesystem.list_directory",
        arguments={"path": "."},
        risk_level="LOW",
        checkpoint_id=None,
    )
    repository.approve_pending_action(run.id)
    worker = LocalDispatchWorker(
        settings=settings,
        repository=repository,
        lifecycle=lifecycle,
        runtime_builder=build_secure_run_runtime,
        worker_id="worker-runtime",
        capabilities=("filesystem.list_directory",),
    )

    asyncio.run(worker.start())
    assert asyncio.run(worker.run_once()) is True

    stored = repository.get_action(action.id)
    assert stored is not None
    assert stored.status == "FAILED"
    assert stored.worker_id == "worker-runtime"
    assert stored.dispatch_attempt == 1
    assert [attempt.status for attempt in repository.list_action_attempts(action.id)] == [
        "CLAIMED",
        "FAILED",
    ]


def test_enabled_dispatch_supervisor_launches_bounded_worker_processes(
    tmp_path: Path, monkeypatch
) -> None:
    import asyncio

    from local_ai_agent.runtime.worker_dispatch import LocalDispatchPool

    settings = replace(
        load_settings(),
        workspace_root=tmp_path / "workspace",
        sqlite_path=tmp_path / "workspace" / ".agent" / "agent.db",
        dispatch_enabled=True,
        dispatch_max_workers=2,
    )
    ensure_workspace(settings)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    lifecycle = RunLifecycleService(repository)
    launches: list[list[str]] = []

    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self):
            return None if not self.terminated else 0

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout=None) -> int:
            del timeout
            return 0

    def fake_popen(command, **_):
        launches.append(command)
        return FakeProcess()

    monkeypatch.setattr("local_ai_agent.runtime.worker_dispatch.subprocess.Popen", fake_popen)
    pool = LocalDispatchPool(
        settings=settings,
        repository=repository,
        lifecycle=lifecycle,
        runtime_builder=lambda **_: None,
    )

    asyncio.run(pool.start())
    assert launches == [
        [__import__("sys").executable, "-m", "local_ai_agent.runtime.worker_main"],
        [__import__("sys").executable, "-m", "local_ai_agent.runtime.worker_main"],
    ]
    asyncio.run(pool.stop())
