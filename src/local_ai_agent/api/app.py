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
from local_ai_agent.runtime.secure_run_runtime import build_secure_run_runtime
from local_ai_agent.schemas.contracts import (
    AgentEvent,
    AgentRun,
    AuthorizationDecision,
    CreateRunRequest,
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

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        ensure_workspace(runtime_settings)
        repository.initialize()
        yield

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

    @app.post("/runs", response_model=AgentRun, status_code=status.HTTP_202_ACCEPTED, tags=["runs"])
    async def create_run(
        payload: CreateRunRequest, _: None = Depends(require_api_token)
    ) -> AgentRun:
        run = AgentRun(
            objective=payload.objective,
            workspace_id=payload.workspace_id,
            budget=payload.budget or _default_budget(runtime_settings),
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

    @app.get("/runs/{run_id}/pending-authorization", tags=["runs"])
    async def pending_authorization(
        run_id: UUID, _: None = Depends(require_api_token)
    ) -> dict[str, object]:
        try:
            pending = lifecycle.pending_authorization(run_id)
        except LifecycleError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(error)) from error
        return {"pending": pending is not None, "tool": pending}

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
