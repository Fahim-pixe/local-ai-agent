"""Runtime-owned tracking for validated agent plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from local_ai_agent.schemas.contracts import AgentPlan, PlanStep


class PlanTrackingError(ValueError):
    """Raised when a plan or requested status transition is invalid."""


class PlanStepStatus(StrEnum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class TrackedPlanStep:
    step: PlanStep
    status: PlanStepStatus
    detail: str | None = None


class PlanTracker:
    """Owns active-plan state and permits dependency-safe step progression only."""

    def __init__(self, plan: AgentPlan) -> None:
        self._plan = plan
        self._steps = {step.id: step for step in plan.steps}
        self._validate_dependencies()
        self._status = {step.id: PlanStepStatus.PENDING for step in plan.steps}
        self._details: dict[int, str | None] = {step.id: None for step in plan.steps}

    @property
    def plan(self) -> AgentPlan:
        return self._plan

    @property
    def is_complete(self) -> bool:
        return all(
            status in {PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED}
            for status in self._status.values()
        )

    @property
    def has_failed_step(self) -> bool:
        return any(status is PlanStepStatus.FAILED for status in self._status.values())

    def active_step(self) -> TrackedPlanStep | None:
        for step in self._plan.steps:
            if self._status[step.id] is PlanStepStatus.ACTIVE:
                return self._tracked_step(step.id)
        return None

    def next_ready_step(self) -> TrackedPlanStep | None:
        if self.active_step() is not None:
            raise PlanTrackingError(
                "An active step must be resolved before another step can start."
            )
        for step in self._plan.steps:
            if self._status[step.id] is PlanStepStatus.PENDING and self._dependencies_completed(
                step
            ):
                return self._tracked_step(step.id)
        return None

    def activate_next(self) -> TrackedPlanStep | None:
        next_step = self.next_ready_step()
        if next_step is None:
            return None
        self._status[next_step.step.id] = PlanStepStatus.ACTIVE
        return self._tracked_step(next_step.step.id)

    def mark_completed(self, step_id: int, detail: str | None = None) -> TrackedPlanStep:
        return self._transition(step_id, PlanStepStatus.COMPLETED, detail)

    def mark_failed(self, step_id: int, detail: str | None = None) -> TrackedPlanStep:
        return self._transition(step_id, PlanStepStatus.FAILED, detail)

    def mark_skipped(self, step_id: int, detail: str | None = None) -> TrackedPlanStep:
        if not self._dependencies_completed(self._steps[step_id]):
            raise PlanTrackingError("A step with unfinished dependencies cannot be skipped.")
        return self._transition(step_id, PlanStepStatus.SKIPPED, detail)

    def snapshot(self) -> list[TrackedPlanStep]:
        return [self._tracked_step(step.id) for step in self._plan.steps]

    def _transition(
        self, step_id: int, target: PlanStepStatus, detail: str | None
    ) -> TrackedPlanStep:
        if step_id not in self._steps:
            raise PlanTrackingError(f"Unknown plan step: {step_id}")
        if self._status[step_id] is not PlanStepStatus.ACTIVE:
            raise PlanTrackingError(f"Plan step {step_id} is not active.")
        self._status[step_id] = target
        self._details[step_id] = detail
        return self._tracked_step(step_id)

    def _dependencies_completed(self, step: PlanStep) -> bool:
        return all(
            self._status[dependency] is PlanStepStatus.COMPLETED for dependency in step.depends_on
        )

    def _tracked_step(self, step_id: int) -> TrackedPlanStep:
        return TrackedPlanStep(
            step=self._steps[step_id], status=self._status[step_id], detail=self._details[step_id]
        )

    def _validate_dependencies(self) -> None:
        if len(self._steps) != len(self._plan.steps):
            raise PlanTrackingError("Plan step IDs must be unique.")
        for step in self._plan.steps:
            for dependency in step.depends_on:
                if dependency not in self._steps:
                    raise PlanTrackingError(
                        f"Plan step {step.id} depends on unknown step {dependency}."
                    )
                if dependency == step.id:
                    raise PlanTrackingError(f"Plan step {step.id} cannot depend on itself.")

        visiting: set[int] = set()
        visited: set[int] = set()

        def visit(step_id: int) -> None:
            if step_id in visiting:
                raise PlanTrackingError("Plan dependencies must not contain a cycle.")
            if step_id in visited:
                return
            visiting.add(step_id)
            for dependency in self._steps[step_id].depends_on:
                visit(dependency)
            visiting.remove(step_id)
            visited.add(step_id)

        for step_id in self._steps:
            visit(step_id)
