"""FastAPI application factory for the local agent control plane."""

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
from local_ai_agent.runtime.ollama_client import OllamaClient, OllamaError
from local_ai_agent.schemas.contracts import (
    AgentEvent,
    AgentRun,
    AgentState,
    AuthorizationDecision,
    CreateRunRequest,
    RunBudget,
    UserReplyRequest,
)

bearer_scheme = HTTPBearer(auto_error=False)


class EventBroker:
    """In-process SSE broker; persistent audit events remain in SQLite in later phases."""

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
    """Create the API app without starting model execution as part of setup."""
    runtime_settings = settings or load_settings()
    repository = RunRepository(runtime_settings.sqlite_path)
    broker = EventBroker()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        ensure_workspace(runtime_settings)
        repository.initialize()
        yield

    app = FastAPI(
        title="Local AI Agent",
        version="0.1.0",
        description="Policy-enforced local AI agent control plane.",
        lifespan=lifespan,
    )
    app.state.settings = runtime_settings
    app.state.repository = repository
    app.state.broker = broker

    async def require_api_token(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    ) -> None:
        if runtime_settings.agent_api_token is None:
            return
        if credentials is None or credentials.credentials != runtime_settings.agent_api_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token."
            )

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
        return {
            "status": "ok" if repository.health_check() else "degraded",
            "database": repository.health_check(),
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
        created = repository.create_run(run)
        await broker.publish(
            AgentEvent(
                run_id=created.id,
                type="run.created",
                state=created.state,
                message="Run created and ready for runtime validation.",
            )
        )
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
            async for event in broker.subscribe(run_id):
                if await request.is_disconnected():
                    break
                yield f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/runs/{run_id}/cancel", status_code=status.HTTP_202_ACCEPTED, tags=["runs"])
    async def cancel_run(run_id: UUID, _: None = Depends(require_api_token)) -> dict[str, str]:
        run = repository.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
        if run.state in {
            AgentState.COMPLETE,
            AgentState.PARTIAL,
            AgentState.FAILED,
            AgentState.CANCELLED,
        }:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail="Run is already terminal."
            )
        return {"status": "accepted", "detail": "Cancellation will be handled by the runtime loop."}

    @app.post("/runs/{run_id}/authorize", status_code=status.HTTP_202_ACCEPTED, tags=["runs"])
    async def authorize_run(
        run_id: UUID, decision: AuthorizationDecision, _: None = Depends(require_api_token)
    ) -> dict[str, object]:
        if repository.get_run(run_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
        return {"status": "accepted", "approved": decision.approved}

    @app.get("/runs/{run_id}/pending-authorization", tags=["runs"])
    async def pending_authorization(
        run_id: UUID, _: None = Depends(require_api_token)
    ) -> dict[str, object]:
        if repository.get_run(run_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
        return {"pending": False, "tool": None}

    @app.post("/runs/{run_id}/reply", status_code=status.HTTP_202_ACCEPTED, tags=["runs"])
    async def reply_to_run(
        run_id: UUID, reply: UserReplyRequest, _: None = Depends(require_api_token)
    ) -> dict[str, object]:
        if repository.get_run(run_id) is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found.")
        return {"status": "accepted", "message_length": len(reply.message)}

    @app.get("/runs", response_model=list[AgentRun], tags=["runs"])
    async def list_runs(_: None = Depends(require_api_token)) -> list[AgentRun]:
        return []

    return app
