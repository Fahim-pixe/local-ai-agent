"""Durable replay of an authorization-paused ReAct run."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from local_ai_agent.db.repository import PendingAction, RunRepository
from local_ai_agent.runtime.checkpointing import RepositoryCheckpointSink
from local_ai_agent.runtime.lifecycle import RunLifecycleService
from local_ai_agent.runtime.react_loop import ReActLoop, ReActLoopResult
from local_ai_agent.runtime.tool_router import ToolRoutingOutcome
from local_ai_agent.schemas.contracts import AgentEvent

if TYPE_CHECKING:
    from local_ai_agent.runtime.run_executor import RunToolExecutor


class ContinuationError(RuntimeError):
    """Raised when a durable continuation cannot be safely resumed."""


@dataclass(frozen=True, slots=True)
class ContinuationResult:
    action: PendingAction
    action_outcome: ToolRoutingOutcome
    react_result: ReActLoopResult


class DurableContinuationService:
    """Execute an approved pending action exactly once and continue its stored conversation."""

    def __init__(
        self,
        *,
        run_id: UUID,
        repository: RunRepository,
        lifecycle: RunLifecycleService,
        executor: RunToolExecutor,
        react_loop: ReActLoop,
        worker_id: str,
        lease_seconds: int,
        heartbeat_seconds: int,
    ) -> None:
        self._run_id = run_id
        self._repository = repository
        self._lifecycle = lifecycle
        self._executor = executor
        self._react_loop = react_loop
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._checkpoint_sink = RepositoryCheckpointSink(run_id=run_id, repository=repository)

    async def resume_approved_action(self, *, system_prompt: str = "") -> ContinuationResult:
        """Claim, execute, and continue one approved action without reissuing the model request."""
        if self._lifecycle.cancel_if_requested(self._run_id):
            raise ContinuationError("Run was cancelled before an approved action could be claimed.")
        action = self._repository.claim_approved_action(
            self._run_id,
            worker_id=self._worker_id,
            lease_seconds=self._lease_seconds,
        )
        if action is None:
            raise ContinuationError("No approved pending action is available for continuation.")
        if action.checkpoint_id is None:
            self._repository.finish_pending_action(action.id, succeeded=False)
            raise ContinuationError(
                "Approved action does not reference a durable ReAct checkpoint."
            )
        checkpoint = self._repository.get_react_checkpoint(action.checkpoint_id)
        if checkpoint is None or checkpoint.run_id != self._run_id:
            self._repository.finish_pending_action(action.id, succeeded=False)
            raise ContinuationError("Approved action references an invalid durable checkpoint.")

        heartbeat_stop = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(action.id, heartbeat_stop))
        try:
            outcome = await self._executor.execute(
                tool_name=action.tool_name,
                arguments=action.arguments,
                authorization_granted=True,
                checkpoint_id=checkpoint.id,
            )
        finally:
            heartbeat_stop.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
        action_succeeded = outcome.result.success and outcome.result.verified
        self._repository.finish_pending_action(action.id, succeeded=action_succeeded)
        current_run = self._repository.get_run(self._run_id)
        if current_run:
            self._repository.record_event(
                AgentEvent(
                    run_id=self._run_id,
                    type="continuation.action_executed",
                    state=current_run.state,
                    message="Approved pending action was claimed and executed exactly once.",
                    data={
                        "action_id": str(action.id),
                        "tool_name": action.tool_name,
                        "success": outcome.result.success,
                        "verified": outcome.result.verified,
                    },
                )
            )
        messages = list(checkpoint.messages)
        messages.append(ReActLoop._tool_result_message(action.tool_name, outcome.result))
        await self._checkpoint_sink.checkpoint(phase="approved-action-result", messages=messages)
        react_result = await self._react_loop.run(
            objective="",
            system_prompt=system_prompt,
            initial_messages=messages,
            checkpoint_sink=self._checkpoint_sink,
        )
        current_run = self._repository.get_run(self._run_id)
        if current_run:
            self._repository.record_event(
                AgentEvent(
                    run_id=self._run_id,
                    type="continuation.replayed",
                    state=current_run.state,
                    message="ReAct continued from the checkpointed approved-action result.",
                    data={"action_id": str(action.id), "result_state": react_result.state.value},
                )
            )
        return ContinuationResult(
            action=action,
            action_outcome=outcome,
            react_result=react_result,
        )

    async def _heartbeat(self, action_id: UUID, stop: asyncio.Event) -> None:
        """Renew a live worker lease until action execution completes or ownership changes."""
        while True:
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._heartbeat_seconds)
                return
            except TimeoutError:
                if not self._repository.renew_action_lease(
                    action_id,
                    worker_id=self._worker_id,
                    lease_seconds=self._lease_seconds,
                ):
                    return
