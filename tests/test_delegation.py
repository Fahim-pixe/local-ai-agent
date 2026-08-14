"""Contract tests for bounded coordinator-to-specialist delegation."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from local_ai_agent.api.app import create_app
from local_ai_agent.config import ensure_workspace, load_settings
from local_ai_agent.db.repository import RunRepository
from local_ai_agent.runtime.delegation import (
    DelegationAuthority,
    DelegationCoordinator,
    DelegationError,
)
from local_ai_agent.schemas.contracts import (
    AgentPlan,
    AgentRun,
    DelegationUnit,
    PlanStep,
    RiskLevel,
    RunBudget,
    SpecialistEvidence,
)


def delegation_environment(tmp_path: Path) -> tuple[RunRepository, UUID]:
    settings = replace(
        load_settings(),
        workspace_root=tmp_path / "workspace",
        sqlite_path=tmp_path / "workspace" / ".agent" / "agent.db",
        agent_api_token=None,
    )
    ensure_workspace(settings)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    run = repository.create_run(
        AgentRun(
            objective="Coordinate two bounded research units.",
            workspace_id="delegation-workspace",
            budget=RunBudget(max_tool_calls=6, max_runtime_seconds=60, max_shell_executions=0),
        )
    )
    return repository, run.id


def plan() -> AgentPlan:
    return AgentPlan(
        goal="Collect verified local evidence in order.",
        steps=[
            PlanStep(
                id=1,
                action="List project files",
                description="Collect a compact project inventory.",
                risk=RiskLevel.LOW,
            ),
            PlanStep(
                id=2,
                action="Read README",
                description="Collect verified README facts.",
                depends_on=[1],
                risk=RiskLevel.LOW,
            ),
        ],
        success_criteria=["Evidence for both steps is verified."],
        rollback_strategy="Stop after the first failed unit; do not broaden authority.",
    )


def units() -> list[DelegationUnit]:
    return [
        DelegationUnit(
            id="inventory",
            plan_step_id=1,
            objective="List only the project root entries.",
            allowed_tool_names=("filesystem.list_directory",),
            max_tool_calls=1,
            max_model_turns=2,
        ),
        DelegationUnit(
            id="readme",
            plan_step_id=2,
            objective="Read README and return only verified implementation facts.",
            allowed_tool_names=("filesystem.read_file",),
            max_tool_calls=1,
            max_model_turns=2,
            depends_on=("inventory",),
        ),
    ]


def authority() -> DelegationAuthority:
    return DelegationAuthority(
        allowed_tool_names=("filesystem.list_directory", "filesystem.read_file"),
        max_tool_calls=2,
        max_model_turns=3,
        max_retries=0,
        max_units=2,
    )


def test_authority_uses_versioned_configuration_caps() -> None:
    settings = replace(
        load_settings(),
        delegation_max_units=4,
        delegation_max_unit_tool_calls=2,
        delegation_max_unit_model_turns=3,
        delegation_max_retries=0,
    )

    configured = DelegationAuthority.from_settings(
        settings,
        allowed_tool_names=("filesystem.list_directory",),
    )

    assert configured.max_units == 4
    assert configured.max_tool_calls == 2
    assert configured.max_model_turns == 3
    assert configured.max_retries == 0
    assert configured.allowed_tool_names == ("filesystem.list_directory",)


def test_coordinator_persists_plan_and_executes_only_dependency_ready_units(tmp_path: Path) -> None:
    repository, run_id = delegation_environment(tmp_path)
    coordinator = DelegationCoordinator(
        run_id=run_id,
        repository=repository,
        plan=plan(),
        units=units(),
        authority=authority(),
    )

    async def specialist(unit: DelegationUnit, received: DelegationAuthority) -> SpecialistEvidence:
        assert received == authority()
        return SpecialistEvidence(
            summary=f"Verified summary for {unit.id}.",
            verified=True,
            verification_strategy="independent-tool-verifier",
            evidence={"unit": unit.id, "verified": True},
        )

    stored = repository.get_run(run_id)
    assert stored is not None
    assert stored.plan == plan()
    first = asyncio.run(coordinator.execute_next(specialist))
    second = asyncio.run(coordinator.execute_next(specialist))
    assert first.unit.id == "inventory"
    assert second.unit.id == "readme"
    assert asyncio.run(coordinator.execute_next(specialist)) is None
    snapshots = coordinator.snapshot()
    assert [snapshot.status for snapshot in snapshots] == ["COMPLETED", "COMPLETED"]
    assert all(snapshot.evidence and snapshot.evidence.verified for snapshot in snapshots)


def test_coordinator_rejects_specialist_authority_widening_and_unverified_evidence(
    tmp_path: Path,
) -> None:
    repository, run_id = delegation_environment(tmp_path)
    overbroad = units()
    overbroad[0] = overbroad[0].model_copy(
        update={"allowed_tool_names": ("filesystem.list_directory", "shell.execute")}
    )
    with pytest.raises(DelegationError, match="outside coordinator authority"):
        DelegationCoordinator(
            run_id=run_id,
            repository=repository,
            plan=plan(),
            units=overbroad,
            authority=authority(),
        )

    missing_prerequisite = units()
    missing_prerequisite[1] = missing_prerequisite[1].model_copy(update={"depends_on": ()})
    with pytest.raises(DelegationError, match="must depend on the unit for plan step 1"):
        DelegationCoordinator(
            run_id=run_id,
            repository=repository,
            plan=plan(),
            units=missing_prerequisite,
            authority=authority(),
        )
    with pytest.raises(DelegationError, match="must not allow autonomous retries"):
        DelegationCoordinator(
            run_id=run_id,
            repository=repository,
            plan=plan(),
            units=units(),
            authority=replace(authority(), max_retries=1),
        )

    coordinator = DelegationCoordinator(
        run_id=run_id,
        repository=repository,
        plan=plan(),
        units=units(),
        authority=authority(),
    )

    async def unverified(_: DelegationUnit, __: DelegationAuthority) -> SpecialistEvidence:
        return SpecialistEvidence(
            summary="The specialist claims the work is complete.",
            verified=False,
            verification_strategy="none",
            evidence={},
        )

    with pytest.raises(DelegationError, match="unverified evidence"):
        asyncio.run(coordinator.execute_next(unverified))
    assert coordinator.snapshot()[0].status == "FAILED"


def test_delegation_snapshot_endpoint_returns_durable_summary_only_progress(tmp_path: Path) -> None:
    repository, run_id = delegation_environment(tmp_path)
    coordinator = DelegationCoordinator(
        run_id=run_id,
        repository=repository,
        plan=plan(),
        units=units(),
        authority=authority(),
    )

    async def verified(unit: DelegationUnit, _: DelegationAuthority) -> SpecialistEvidence:
        return SpecialistEvidence(
            summary=f"Verified summary for {unit.id}.",
            verified=True,
            verification_strategy="independent-tool-verifier",
            evidence={"unit": unit.id, "verified": True},
        )

    asyncio.run(coordinator.execute_next(verified))
    settings = replace(
        load_settings(),
        workspace_root=tmp_path / "workspace",
        sqlite_path=tmp_path / "workspace" / ".agent" / "agent.db",
        agent_api_token="test-token",
    )
    with TestClient(create_app(settings)) as client:
        unauthorized = client.get(f"/runs/{run_id}/delegation")
        assert unauthorized.status_code == 401
        response = client.get(
            f"/runs/{run_id}/delegation",
            headers={"Authorization": "Bearer test-token"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["plan_step_mapping"] == {"1": "inventory", "2": "readme"}
    assert payload["units"][0]["status"] == "COMPLETED"
    assert payload["units"][0]["evidence"]["summary"] == "Verified summary for inventory."
    assert "raw_output" not in str(payload)


def test_specialist_evidence_contract_rejects_raw_output_fields() -> None:
    with pytest.raises(ValueError):
        SpecialistEvidence.model_validate(
            {
                "summary": "summary",
                "verified": True,
                "verification_strategy": "test",
                "evidence": {},
                "raw_output": "sensitive tool payload",
            }
        )
