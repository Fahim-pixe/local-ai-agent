"""Durable run lifecycle orchestration built on repository-owned state and locks."""

from __future__ import annotations

from dataclasses import dataclass
from secrets import compare_digest
from typing import Any
from uuid import UUID

from local_ai_agent.db.repository import RunRepository
from local_ai_agent.runtime.state_machine import InvalidStateTransition, StateMachine
from local_ai_agent.schemas.contracts import AgentEvent, AgentRun, AgentState


class WorkspaceBusyError(RuntimeError):
    """Raised when another active run already owns the requested workspace session."""


class LifecycleError(RuntimeError):
    """Raised when a durable run lifecycle operation cannot be completed."""


@dataclass(frozen=True, slots=True)
class AuthorizationRequest:
    run_id: UUID
    tool_name: str
    arguments: dict[str, Any]
    risk: str
    checkpoint_id: int | None = None


class RunLifecycleService:
    """Own per-session locking, authoritative state transitions, and durable lifecycle events."""

    def __init__(self, repository: RunRepository) -> None:
        self._repository = repository

    def register_run(self, run: AgentRun) -> AgentRun:
        created = self._repository.create_run(run)
        if not self._repository.acquire_session_lock(
            workspace_id=created.workspace_id, run_id=created.id
        ):
            failed = self._require_updated(created.id, AgentState.FAILED)
            self._record_event(
                failed,
                "run.failed",
                "Workspace is already owned by another active run.",
                {"error_code": "WORKSPACE_BUSY"},
            )
            raise WorkspaceBusyError(f"Workspace is already owned: {created.workspace_id}")
        self._record_event(created, "run.created", "Run registered and workspace lock acquired.")
        return created

    def transition(
        self, run_id: UUID, target: AgentState, message: str, data: dict[str, Any] | None = None
    ) -> AgentRun:
        current = self._require_run(run_id)
        try:
            StateMachine(current.state).transition_to(target)
        except InvalidStateTransition as error:
            raise LifecycleError(str(error)) from error
        updated = self._require_updated(run_id, target)
        self._record_event(updated, f"run.{target.value.lower()}", message, data)
        if target in {
            AgentState.COMPLETE,
            AgentState.PARTIAL,
            AgentState.FAILED,
            AgentState.CANCELLED,
        }:
            self._repository.release_session_lock(
                workspace_id=updated.workspace_id, run_id=updated.id
            )
        return updated

    def request_cancellation(self, run_id: UUID) -> AgentRun:
        run = self._require_run(run_id)
        if run.state in {
            AgentState.COMPLETE,
            AgentState.PARTIAL,
            AgentState.FAILED,
            AgentState.CANCELLED,
        }:
            raise LifecycleError("Terminal runs cannot be cancelled.")
        if not self._repository.request_cancellation(run_id):
            raise LifecycleError("Cancellation control record is unavailable.")
        self._record_event(run, "run.cancel_requested", "Cancellation was requested by the user.")
        return run

    def cancel_if_requested(self, run_id: UUID) -> bool:
        if not self._repository.cancellation_requested(run_id):
            return False
        run = self._require_run(run_id)
        if run.state in {
            AgentState.COMPLETE,
            AgentState.PARTIAL,
            AgentState.FAILED,
            AgentState.CANCELLED,
        }:
            return False
        self.transition(
            run_id, AgentState.CANCELLED, "Run cancelled after a verified cancellation request."
        )
        return True

    def require_authorization(self, request: AuthorizationRequest) -> AgentRun:
        self._require_run(request.run_id)
        action = self._repository.create_pending_action(
            run_id=request.run_id,
            tool_name=request.tool_name,
            arguments=request.arguments,
            risk_level=request.risk,
            checkpoint_id=request.checkpoint_id,
        )
        if not self._repository.set_pending_authorization(
            request.run_id,
            {
                "action_id": str(action.id),
                "tool_name": request.tool_name,
                "arguments": request.arguments,
                "risk": request.risk,
                "checkpoint_id": request.checkpoint_id,
            },
        ):
            raise LifecycleError("Authorization control record is unavailable.")
        return self.transition(
            request.run_id,
            AgentState.AUTHORIZATION_REQUIRED,
            "Runtime requires explicit authorization before the requested tool can execute.",
            {"tool_name": request.tool_name, "risk": request.risk},
        )

    def resolve_authorization(self, run_id: UUID, approved: bool) -> AgentRun:
        self._require_run(run_id)
        pending = self._repository.get_pending_authorization(run_id)
        if pending is None:
            raise LifecycleError("No authorization request is pending for this run.")
        if not self._repository.resolve_authorization(run_id):
            raise LifecycleError("Authorization control record is unavailable.")
        if approved:
            if self._repository.approve_pending_action(run_id) is None:
                raise LifecycleError("Pending action could not be approved.")
            return self.transition(
                run_id,
                AgentState.EXECUTE,
                "Authorization approved; runtime may resume the pending tool operation.",
                {"tool_name": pending["tool_name"]},
            )
        self._repository.reject_pending_action(run_id)
        return self.transition(
            run_id,
            AgentState.PARTIAL,
            "Authorization denied; the pending tool operation was not executed.",
            {"tool_name": pending["tool_name"]},
        )

    def recover_stale_executing_actions(
        self,
        *,
        now: Any = None,
        lease_seconds: int = 300,
        reason: str = "WORKER_CRASH_RECOVERY",
    ) -> list[Any]:
        """Fail abandoned worker claims and emit an authoritative recovery audit event."""
        recovered = self._repository.recover_stale_executing_actions(
            now=now, lease_seconds=lease_seconds, reason=reason
        )
        for action in recovered:
            run = self._repository.get_run(action.run_id)
            if run is None:
                continue
            self._record_event(
                run,
                "continuation.action_recovered",
                "An expired worker lease was recovered without re-executing the action.",
                {
                    "action_id": str(action.id),
                    "tool_name": action.tool_name,
                    "worker_id": action.worker_id,
                    "lease_expires_at": action.lease_expires_at.isoformat()
                    if action.lease_expires_at
                    else None,
                    "recovery_reason": action.recovery_reason,
                },
            )
        return recovered

    def require_valid_resume_token(self, run_id: UUID, resume_token: UUID) -> AgentRun:
        """Return the run only when the caller presents its persisted resume token."""
        run = self._require_run(run_id)
        if not compare_digest(str(run.resume_token), str(resume_token)):
            raise LifecycleError("Invalid resume token.")
        return run

    def pending_authorization(self, run_id: UUID) -> dict[str, Any] | None:
        self._require_run(run_id)
        return self._repository.get_pending_authorization(run_id)

    def _require_run(self, run_id: UUID) -> AgentRun:
        run = self._repository.get_run(run_id)
        if run is None:
            raise LifecycleError("Run not found.")
        return run

    def _require_updated(self, run_id: UUID, state: AgentState) -> AgentRun:
        updated = self._repository.update_run(run_id=run_id, state=state)
        if updated is None:
            raise LifecycleError("Run could not be updated.")
        return updated

    def _record_event(
        self, run: AgentRun, event_type: str, message: str, data: dict[str, Any] | None = None
    ) -> None:
        self._repository.record_event(
            AgentEvent(
                run_id=run.id,
                type=event_type,
                state=run.state,
                message=message,
                data=data or {},
            )
        )
