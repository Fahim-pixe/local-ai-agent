"""Coordinator-owned bounded specialist delegation with verified-evidence admission."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

from local_ai_agent.config import Settings
from local_ai_agent.db.repository import RunRepository
from local_ai_agent.schemas.contracts import (
    AgentEvent,
    AgentPlan,
    DelegationUnit,
    SpecialistEvidence,
)


class DelegationError(RuntimeError):
    """Raised when a unit would exceed coordinator-owned delegation policy."""


class DelegationStatus(StrEnum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class DelegationAuthority:
    """Read-only coordinator policy passed to specialists; it cannot authorize widening."""

    allowed_tool_names: tuple[str, ...]
    max_tool_calls: int
    max_model_turns: int
    max_retries: int
    max_units: int

    @classmethod
    def from_settings(
        cls, settings: Settings, *, allowed_tool_names: tuple[str, ...]
    ) -> DelegationAuthority:
        """Create authority from versioned runtime caps and registry-owned tool access."""
        return cls(
            allowed_tool_names=allowed_tool_names,
            max_tool_calls=settings.delegation_max_unit_tool_calls,
            max_model_turns=settings.delegation_max_unit_model_turns,
            max_retries=settings.delegation_max_retries,
            max_units=settings.delegation_max_units,
        )


@dataclass(frozen=True, slots=True)
class DelegationSnapshot:
    unit: DelegationUnit
    status: DelegationStatus
    evidence: SpecialistEvidence | None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class DelegationResult:
    unit: DelegationUnit
    evidence: SpecialistEvidence


Specialist = Callable[[DelegationUnit, DelegationAuthority], Awaitable[SpecialistEvidence]]


class DelegationCoordinator:
    """Runs short specialist units while retaining all security and retry authority."""

    def __init__(
        self,
        *,
        run_id: UUID,
        repository: RunRepository,
        plan: AgentPlan,
        units: list[DelegationUnit],
        authority: DelegationAuthority,
    ) -> None:
        self._run_id = run_id
        self._repository = repository
        self._plan = plan
        self._authority = authority
        self._unit_order = tuple(unit.id for unit in units)
        self._units = {unit.id: unit for unit in units}
        self._status = {unit.id: DelegationStatus.PENDING for unit in units}
        self._evidence: dict[str, SpecialistEvidence | None] = {unit.id: None for unit in units}
        self._detail: dict[str, str | None] = {unit.id: None for unit in units}
        self._validate_contract()
        run = self._repository.get_run(run_id)
        if run is None:
            raise DelegationError("Cannot delegate work for a missing run.")
        if self._repository.update_run(run_id=run_id, state=run.state, plan=plan) is None:
            raise DelegationError("Could not persist the coordinator-owned plan.")
        self._record(
            "delegation.plan_persisted",
            "Coordinator persisted a bounded specialist delegation plan.",
            {
                "unit_count": len(units),
                "max_unit_tool_calls": max(unit.max_tool_calls for unit in units),
                "units": [unit.model_dump(mode="json") for unit in units],
            },
        )

    def snapshot(self) -> list[DelegationSnapshot]:
        return [
            DelegationSnapshot(
                unit=unit,
                status=self._status[unit.id],
                evidence=self._evidence[unit.id],
                detail=self._detail[unit.id],
            )
            for unit_id in self._unit_order
            for unit in (self._units[unit_id],)
        ]

    async def execute_next(self, specialist: Specialist) -> DelegationResult | None:
        """Run one dependency-ready bounded unit and accept only independently verified evidence."""
        unit = self._next_ready_unit()
        if unit is None:
            return None
        self._status[unit.id] = DelegationStatus.ACTIVE
        self._record(
            "delegation.unit_started",
            "Coordinator released one bounded unit to a specialist.",
            {
                "unit_id": unit.id,
                "plan_step_id": unit.plan_step_id,
                "max_tool_calls": unit.max_tool_calls,
                "max_model_turns": unit.max_model_turns,
            },
        )
        try:
            evidence = await specialist(unit, self._authority)
        except Exception as error:
            self._status[unit.id] = DelegationStatus.FAILED
            self._detail[unit.id] = f"Specialist raised {type(error).__name__}."
            self._record(
                "delegation.unit_failed",
                "Specialist failed before returning verified evidence.",
                {"unit_id": unit.id, "error_type": type(error).__name__},
            )
            raise DelegationError(self._detail[unit.id]) from error
        if not evidence.verified:
            self._status[unit.id] = DelegationStatus.FAILED
            self._detail[unit.id] = "Specialist returned unverified evidence."
            self._record(
                "delegation.unit_failed",
                "Coordinator rejected unverified specialist evidence.",
                {"unit_id": unit.id, "verification_strategy": evidence.verification_strategy},
            )
            raise DelegationError("Coordinator rejected unverified evidence.")
        self._status[unit.id] = DelegationStatus.COMPLETED
        self._evidence[unit.id] = evidence
        self._detail[unit.id] = evidence.summary
        self._record(
            "delegation.unit_completed",
            "Coordinator accepted compact verified specialist evidence.",
            {
                "unit_id": unit.id,
                "verification_strategy": evidence.verification_strategy,
                "verified": evidence.verified,
                "evidence": evidence.evidence,
                "summary": evidence.summary,
            },
        )
        return DelegationResult(unit=unit, evidence=evidence)

    def _next_ready_unit(self) -> DelegationUnit | None:
        if any(status is DelegationStatus.ACTIVE for status in self._status.values()):
            raise DelegationError("A specialist unit is already active.")
        for unit in self._units.values():
            if self._status[unit.id] is not DelegationStatus.PENDING:
                continue
            if all(
                self._status[dependency] is DelegationStatus.COMPLETED
                for dependency in unit.depends_on
            ):
                return unit
        return None

    def _validate_contract(self) -> None:
        if not self._units:
            raise DelegationError("At least one bounded delegation unit is required.")
        if len(self._units) != len(self._unit_order):
            raise DelegationError("Delegation unit IDs must be unique.")
        if len(self._units) > self._authority.max_units:
            raise DelegationError("Unit count exceeds coordinator authority.")
        if self._authority.max_retries != 0:
            raise DelegationError("Specialist authority must not allow autonomous retries.")
        if self._authority.max_tool_calls < 0 or self._authority.max_model_turns < 1:
            raise DelegationError("Coordinator authority contains an invalid execution cap.")
        plan_steps = {step.id: step for step in self._plan.steps}
        unit_steps: set[int] = set()
        unit_by_plan_step = {unit.plan_step_id: unit for unit in self._units.values()}
        for unit in self._units.values():
            if unit.plan_step_id not in plan_steps:
                raise DelegationError(f"Unit {unit.id} references an unknown plan step.")
            if unit.plan_step_id in unit_steps:
                raise DelegationError("Each plan step may have at most one specialist unit.")
            unit_steps.add(unit.plan_step_id)
            if unit.max_tool_calls > self._authority.max_tool_calls:
                raise DelegationError("Unit tool-call cap exceeds coordinator authority.")
            if unit.max_model_turns > self._authority.max_model_turns:
                raise DelegationError("Unit model-turn cap exceeds coordinator authority.")
            if not set(unit.allowed_tool_names).issubset(self._authority.allowed_tool_names):
                raise DelegationError("Unit requests tools outside coordinator authority.")
            for dependency in unit.depends_on:
                if dependency not in self._units:
                    raise DelegationError(f"Unit {unit.id} depends on an unknown unit.")
                if dependency == unit.id:
                    raise DelegationError(f"Unit {unit.id} cannot depend on itself.")
            for step_dependency in plan_steps[unit.plan_step_id].depends_on:
                dependency_unit = unit_by_plan_step.get(step_dependency)
                if dependency_unit is not None and dependency_unit.id not in unit.depends_on:
                    raise DelegationError(
                        f"Unit {unit.id} must depend on the unit for plan step {step_dependency}."
                    )
        self._validate_cycles()

    def _validate_cycles(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(unit_id: str) -> None:
            if unit_id in visiting:
                raise DelegationError("Delegation dependencies must not contain a cycle.")
            if unit_id in visited:
                return
            visiting.add(unit_id)
            for dependency in self._units[unit_id].depends_on:
                visit(dependency)
            visiting.remove(unit_id)
            visited.add(unit_id)

        for unit_id in self._units:
            visit(unit_id)

    def _record(self, event_type: str, message: str, data: dict[str, Any]) -> None:
        run = self._repository.get_run(self._run_id)
        if run is None:
            raise DelegationError("Run disappeared during delegation.")
        self._repository.record_event(
            AgentEvent(
                run_id=self._run_id, type=event_type, state=run.state, message=message, data=data
            )
        )
