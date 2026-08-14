"""FastAPI control plane for durable local-agent run lifecycle state."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from local_ai_agent.config import Settings, ensure_workspace, load_settings
from local_ai_agent.db.repository import RunRepository
from local_ai_agent.runtime.continuation import ContinuationError
from local_ai_agent.runtime.lifecycle import LifecycleError, RunLifecycleService, WorkspaceBusyError
from local_ai_agent.runtime.ollama_client import OllamaClient, OllamaError
from local_ai_agent.runtime.production_prompt import ProductionPromptError, load_production_prompt
from local_ai_agent.runtime.secure_run_runtime import build_secure_run_runtime
from local_ai_agent.runtime.worker_dispatch import LocalDispatchPool
from local_ai_agent.schemas.contracts import (
    AgentEvent,
    AgentRun,
    AuthorizationDecision,
    CreateRunRequest,
    OperationalMetrics,
    ResumeRunRequest,
    RunBudget,
    UserReplyRequest,
)

bearer_scheme = HTTPBearer(auto_error=False)


class EventBroker:
    """In-process notification fanout; SQLite retains the durable event history."""

    def __init__(self) -> None:
        self._queues: dict[UUID, asyncio.Queue[AgentEvent]] = defaultdict(asyncio.Queue)

    async def publish(self, event: AgentEvent) -> None:
        await self._queues[event.run_id].put(event)

    async def subscribe(self, run_id: UUID) -> AsyncIterator[AgentEvent]:
        queue = self._queues[run_id]
        while True:
            yield await queue.get()


def _default_budget(settings: Settings) -> RunBudget:
    return RunBudget(
        max_tool_calls=settings.default_max_tool_calls,
        max_runtime_seconds=settings.default_max_runtime_seconds,
        max_shell_executions=settings.default_max_shell_executions,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API app with SQLite-authoritative lifecycle controls."""
    runtime_settings = settings or load_settings()
    repository = RunRepository(runtime_settings.sqlite_path)
    lifecycle = RunLifecycleService(repository)
    broker = EventBroker()
    dispatch_pool = LocalDispatchPool(
        settings=runtime_settings,
        repository=repository,
        lifecycle=lifecycle,
        runtime_builder=build_secure_run_runtime,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        ensure_workspace(runtime_settings)
        repository.initialize()
        app.state.startup_recovered_actions = lifecycle.recover_stale_executing_actions(
            lease_seconds=runtime_settings.worker_lease_seconds
        )
        await dispatch_pool.start()
        try:
            yield
        finally:
            await dispatch_pool.stop()

    app = FastAPI(
        title="Local AI Agent",
        version="0.2.0",
        description="Policy-enforced local AI agent control plane.",
        lifespan=lifespan,
    )
    app.state.settings = runtime_settings
    app.state.repository = repository
    app.state.lifecycle = lifecycle
    app.state.broker = broker
    app.state.runtime_builder = build_secure_run_runtime
    app.state.dispatch_pool = dispatch_pool

    async def require_api_token(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    ) -> None:
        if runtime_settings.agent_api_token is None:
            return
        if credentials is None or credentials.credentials != runtime_settings.agent_api_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token."
            )

    async def publish_latest_event(run_id: UUID) -> None:
        events = repository.list_events(run_id)
        if events:
            await broker.publish(events[-1])

    def delegation_progress(run_id: UUID) -> dict[str, object]:
        """Reconstruct bounded delegation status from the durable event stream."""
        units: dict[str, dict[str, object]] = {}
        order: list[str] = []
        for event in repository.list_events(run_id):
            data = event.data
            if event.type == "delegation.plan_persisted":
                for raw_unit in data.get("units", []):
                    if not isinstance(raw_unit, dict) or not isinstance(raw_unit.get("id"), str):
                        continue
                    unit_id = raw_unit["id"]
                    units[unit_id] = {
                        "unit": raw_unit,
                        "status": "PENDING",
                        "evidence": None,
                        "detail": None,
                    }
                    order.append(unit_id)
                continue
            unit_id = data.get("unit_id")
            if not isinstance(unit_id, str) or unit_id not in units:
                continue
            if event.type == "delegation.unit_started":
                units[unit_id]["status"] = "ACTIVE"
            elif event.type == "delegation.unit_completed":
                units[unit_id]["status"] = "COMPLETED"
                units[unit_id]["detail"] = data.get("summary")
                units[unit_id]["evidence"] = {
                    "summary": data.get("summary"),
                    "verified": data.get("verified") is True,
                    "verification_strategy": data.get("verification_strategy"),
                    "evidence": data.get("evidence", {}),
                }
            elif event.type == "delegation.unit_failed":
                units[unit_id]["status"] = "FAILED"
                units[unit_id]["detail"] = event.message
        return {
            "plan_step_mapping": {
                str(units[unit_id]["unit"]["plan_step_id"]): unit_id for unit_id in order
            },
            "units": [units[unit_id] for unit_id in order],
        }

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, object]:
        ollama_status: dict[str, object] = {
            "available": False,
            "model": runtime_settings.ollama_model,
        }
        try:
            await OllamaClient(runtime_settings.ollama_base_url).health_check(
                runtime_settings.ollama_model
            )
            ollama_status["available"] = True
        except OllamaError as error:
            ollama_status["error"] = type(error).__name__
        database_available = repository.health_check()
        return {
            "status": "ok" if database_available else "degraded",
            "database": database_available,
            "workspace": runtime_settings.workspace_project_path.is_dir(),
            "ollama": ollama_status,
        }

    @app.get("/metrics/operational", response_model=OperationalMetrics, tags=["metrics"])
    async def get_operational_metrics(
        _: None = Depends(require_api_token),
    ) -> OperationalMetrics:
        return repository.operational_metrics()

    @app.post("/runs", response_model=AgentRun, status_code=status.HTTP_202_ACCEPTED, tags=["runs"])
    async def create_run(
        payload: CreateRunRequest, _: None = Depends(require_api_token)
    ) -> AgentRun:
        try:
            prompt = load_production_prompt(runtime_settings)
        except ProductionPromptError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Production system prompt is unavailable.",
            ) from error
        run = AgentRun(
            objective=payload.objective,
            workspace_id=payload.workspace_id,
            budget=payload.budget or _default_budget(runtime_settings),
            prompt_hash=prompt.sha256,
        )
        try:
            created = lifecycle.register_run(run)
        except WorkspaceBusyError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        await publish_latest_event(created.id)
        return created

    @app.get("/runs/{run_id}", response_model=AgentRun, tags=["runs"])
    async def get_run(run_id: UUID, _: None = Depends(require_api_token)) -> AgentRun:
        run = repository.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
        return run

    @app.get("/runs/{run_id}/delegation", tags=["runs"])
    async def get_delegation_progress(
        run_id: UUID, _: None = Depends(require_api_token)
    ) -> dict[str, object]:
        if repository.get_run(run_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
        return delegation_progress(run_id)

    @app.get("/runs/{run_id}/events", tags=["runs"])
    async def stream_events(
        run_id: UUID, request: Request, _: None = Depends(require_api_token)
    ) -> StreamingResponse:
        if repository.get_run(run_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")

        async def event_stream() -> AsyncIterator[str]:
            delivered_ids: set[int] = set()
            for event in repository.list_events(run_id):
                event_id = int(event.data.get("event_id", 0))
                delivered_ids.add(event_id)
                yield f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"
            async for event in broker.subscribe(run_id):
                if await request.is_disconnected():
                    break
                event_id = int(event.data.get("event_id", 0))
                if event_id and event_id in delivered_ids:
                    continue
                if event_id:
                    delivered_ids.add(event_id)
                yield f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/runs/{run_id}/cancel", status_code=status.HTTP_202_ACCEPTED, tags=["runs"])
    async def cancel_run(run_id: UUID, _: None = Depends(require_api_token)) -> dict[str, str]:
        try:
            lifecycle.request_cancellation(run_id)
        except LifecycleError as error:
            detail = str(error)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND
                if detail == "Run not found."
                else status.HTTP_409_CONFLICT,
                detail=detail,
            ) from error
        await publish_latest_event(run_id)
        return {"status": "accepted", "detail": "Cancellation was persisted for the runtime loop."}

    @app.post("/runs/{run_id}/authorize", status_code=status.HTTP_202_ACCEPTED, tags=["runs"])
    async def authorize_run(
        run_id: UUID, decision: AuthorizationDecision, _: None = Depends(require_api_token)
    ) -> dict[str, object]:
        try:
            updated = lifecycle.resolve_authorization(run_id, approved=decision.approved)
        except LifecycleError as error:
            detail = str(error)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND
                if detail == "Run not found."
                else status.HTTP_409_CONFLICT,
                detail=detail,
            ) from error
        await publish_latest_event(run_id)
        return {"status": "accepted", "approved": decision.approved, "state": updated.state.value}

    @app.post("/runs/{run_id}/continue", status_code=status.HTTP_202_ACCEPTED, tags=["runs"])
    async def continue_run(run_id: UUID, _: None = Depends(require_api_token)) -> dict[str, object]:
        if repository.get_run(run_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
        try:
            runtime = app.state.runtime_builder(
                settings=runtime_settings,
                run_id=run_id,
                repository=repository,
                lifecycle=lifecycle,
            )
            result = await runtime.continuation.resume_approved_action()
        except ContinuationError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        await publish_latest_event(run_id)
        return {
            "status": "completed",
            "action_id": str(result.action.id),
            "tool_name": result.action.tool_name,
            "action_verified": result.action_outcome.result.verified,
            "react_state": result.react_result.state.value,
            "final_response": result.react_result.final_response,
        }

    @app.post("/runs/{run_id}/resume", status_code=status.HTTP_202_ACCEPTED, tags=["runs"])
    async def resume_run(
        run_id: UUID, payload: ResumeRunRequest, _: None = Depends(require_api_token)
    ) -> dict[str, object]:
        try:
            lifecycle.require_valid_resume_token(run_id, payload.resume_token)
        except LifecycleError as error:
            detail = str(error)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND
                if detail == "Run not found."
                else status.HTTP_403_FORBIDDEN,
                detail=detail,
            ) from error
        try:
            runtime = app.state.runtime_builder(
                settings=runtime_settings,
                run_id=run_id,
                repository=repository,
                lifecycle=lifecycle,
            )
            result = await runtime.continuation.resume_approved_action()
        except ContinuationError as error:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        await publish_latest_event(run_id)
        return {
            "status": "completed",
            "action_id": str(result.action.id),
            "tool_name": result.action.tool_name,
            "action_verified": result.action_outcome.result.verified,
            "react_state": result.react_result.state.value,
            "final_response": result.react_result.final_response,
        }

    @app.get("/runs/{run_id}/pending-authorization", tags=["runs"])
    async def pending_authorization(
        run_id: UUID, _: None = Depends(require_api_token)
    ) -> dict[str, object]:
        try:
            pending = lifecycle.pending_authorization(run_id)
        except LifecycleError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        return {"pending": pending is not None, "tool": pending}

    @app.get("/workers", tags=["workers"])
    async def list_workers(_: None = Depends(require_api_token)) -> dict[str, object]:
        return {
            "workers": [
                {
                    "worker_id": worker.worker_id,
                    "hostname": worker.hostname,
                    "process_id": worker.process_id,
                    "capabilities": worker.capabilities,
                    "state": worker.state,
                    "started_at": worker.started_at,
                    "last_heartbeat_at": worker.last_heartbeat_at,
                    "stopped_at": worker.stopped_at,
                }
                for worker in repository.list_workers()
            ]
        }

    @app.get("/workers/{worker_id}", tags=["workers"])
    async def get_worker(worker_id: str, _: None = Depends(require_api_token)) -> dict[str, object]:
        worker = repository.get_worker(worker_id)
        if worker is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Worker not found.")
        return {
            "worker_id": worker.worker_id,
            "hostname": worker.hostname,
            "process_id": worker.process_id,
            "capabilities": worker.capabilities,
            "state": worker.state,
            "started_at": worker.started_at,
            "last_heartbeat_at": worker.last_heartbeat_at,
            "stopped_at": worker.stopped_at,
        }

    @app.post("/workers/{worker_id}/drain", tags=["workers"])
    async def drain_worker(
        worker_id: str, _: None = Depends(require_api_token)
    ) -> dict[str, object]:
        if not repository.drain_worker(worker_id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Worker is missing or cannot be drained from its current state.",
            )
        return {"worker_id": worker_id, "state": "DRAINING"}

    @app.get("/actions/{action_id}/attempts", tags=["workers"])
    async def list_action_attempts(
        action_id: UUID, _: None = Depends(require_api_token)
    ) -> dict[str, object]:
        action = repository.get_action(action_id)
        if action is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Action not found.")
        return {
            "action_id": str(action.id),
            "attempts": [
                {
                    "id": attempt.id,
                    "attempt": attempt.attempt,
                    "worker_id": attempt.worker_id,
                    "status": attempt.status,
                    "detail": attempt.detail,
                    "created_at": attempt.created_at,
                }
                for attempt in repository.list_action_attempts(action.id)
            ],
        }

    @app.get("/runs/{run_id}/actions", tags=["runs"])
    async def list_actions(run_id: UUID, _: None = Depends(require_api_token)) -> dict[str, object]:
        if repository.get_run(run_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
        actions = repository.list_pending_actions(run_id)
        return {
            "actions": [
                {
                    "id": str(action.id),
                    "tool_name": action.tool_name,
                    "risk_level": action.risk_level,
                    "status": action.status,
                    "worker_id": action.worker_id,
                    "recovery_class": action.recovery_class.value,
                    "operation_key_prefix": action.operation_key[:12]
                    if action.operation_key
                    else None,
                    "dispatch_attempt": action.dispatch_attempt,
                    "max_dispatch_attempts": action.max_dispatch_attempts,
                    "claimed_at": action.claimed_at,
                    "lease_expires_at": action.lease_expires_at,
                    "recovered_at": action.recovered_at,
                    "recovery_reason": action.recovery_reason,
                    "executed_at": action.executed_at,
                }
                for action in actions
            ]
        }

    @app.post("/runs/{run_id}/reply", status_code=status.HTTP_202_ACCEPTED, tags=["runs"])
    async def reply_to_run(
        run_id: UUID, reply: UserReplyRequest, _: None = Depends(require_api_token)
    ) -> dict[str, object]:
        if repository.get_run(run_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
        repository.record_event(
            AgentEvent(
                run_id=run_id,
                type="run.reply_received",
                state=repository.get_run(run_id).state,
                message="User reply persisted for runtime consumption.",
                data={"message_length": len(reply.message)},
            )
        )
        await publish_latest_event(run_id)
        return {"status": "accepted", "message_length": len(reply.message)}

    @app.get("/runs", response_model=list[AgentRun], tags=["runs"])
    async def list_runs(_: None = Depends(require_api_token)) -> list[AgentRun]:
        return repository.list_runs()

    return app
