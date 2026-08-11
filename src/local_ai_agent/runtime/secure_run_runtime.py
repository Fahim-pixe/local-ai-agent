"""Assembly of a persisted run's full Priority 3 tool surface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from local_ai_agent.config import Settings
from local_ai_agent.db.repository import RunRepository
from local_ai_agent.memory.context_manager import ContextManager
from local_ai_agent.memory.repository import MemoryRepository
from local_ai_agent.runtime.budget_manager import BudgetManager
from local_ai_agent.runtime.lifecycle import LifecycleError, RunLifecycleService
from local_ai_agent.runtime.loop_detector import LoopDetector
from local_ai_agent.runtime.ollama_client import OllamaClient
from local_ai_agent.runtime.permission_gate import PermissionGate
from local_ai_agent.runtime.react_loop import NativeToolChatClient, ReActLoop, ReActLoopResult
from local_ai_agent.runtime.retry_engine import RetryEngine
from local_ai_agent.runtime.run_executor import RunToolExecutor
from local_ai_agent.runtime.tool_router import ToolRouter
from local_ai_agent.runtime.verification_engine import VerificationEngine
from local_ai_agent.schemas.contracts import ToolResult
from local_ai_agent.tools.filesystem import build_read_only_filesystem_tools
from local_ai_agent.tools.memory import build_memory_tools
from local_ai_agent.tools.mutation import build_mutation_tools
from local_ai_agent.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class SecureRunRuntime:
    run_id: UUID
    repository: RunRepository
    registry: ToolRegistry
    tool_router: ToolRouter
    executor: RunToolExecutor
    react_loop: ReActLoop
    memory_repository: MemoryRepository
    context_manager: ContextManager

    async def run_with_context(
        self,
        *,
        system_prompt: str,
        recent_tool_results: list[ToolResult] | None = None,
        recent_conversation: list[dict[str, Any]] | None = None,
        completed_steps: list[str] | None = None,
        unresolved_errors: list[str] | None = None,
    ) -> ReActLoopResult:
        """Assemble bounded live context from the durable run before one ReAct interaction."""
        run = self.repository.get_run(self.run_id)
        if run is None:
            raise LifecycleError("Cannot run context assembly for a missing run.")
        plan_summary = None
        active_step = None
        if run.plan:
            plan_summary = "; ".join(step.description for step in run.plan.steps)
            active_step = run.plan.steps[0].description
        assembly = self.context_manager.assemble(
            objective=run.objective,
            state=run.state,
            plan_summary=plan_summary,
            active_step=active_step,
            unresolved_errors=unresolved_errors,
            recent_tool_results=recent_tool_results,
            recent_conversation=recent_conversation,
            completed_steps=completed_steps,
            memory_query=run.objective,
        )
        return await self.react_loop.run(
            objective=run.objective,
            system_prompt=system_prompt,
            runtime_context=assembly.as_system_context(),
        )


def build_secure_run_runtime(
    *,
    settings: Settings,
    run_id: UUID,
    repository: RunRepository,
    lifecycle: RunLifecycleService,
    client: NativeToolChatClient | None = None,
) -> SecureRunRuntime:
    """Create the per-run tool surface; high-risk handlers remain gated by ToolRouter."""
    run = repository.get_run(run_id)
    if run is None:
        raise LifecycleError("Cannot build runtime for a missing run.")
    memory_repository = MemoryRepository(settings.sqlite_path)
    memory_repository.initialize()
    context_manager = ContextManager(settings, memory_repository)
    registry = ToolRegistry()
    for definition in build_read_only_filesystem_tools(settings):
        registry.register(definition)
    for definition in build_mutation_tools(settings=settings, run_id=run_id, repository=repository):
        registry.register(definition)
    for definition in build_memory_tools(repository=memory_repository, source_run_id=run_id):
        registry.register(definition)
    router = ToolRouter(
        registry=registry,
        permission_gate=PermissionGate(),
        budget_manager=BudgetManager(run.budget),
        loop_detector=LoopDetector(),
        verification_engine=VerificationEngine(),
        retry_engine=RetryEngine(),
    )
    executor = RunToolExecutor(
        run_id=run_id,
        registry=registry,
        tool_router=router,
        repository=repository,
        lifecycle=lifecycle,
    )
    loop = ReActLoop(
        settings=settings,
        client=client or OllamaClient(settings.ollama_base_url),
        registry=registry,
        tool_router=executor,
    )
    return SecureRunRuntime(
        run_id=run_id,
        repository=repository,
        registry=registry,
        tool_router=router,
        executor=executor,
        react_loop=loop,
        memory_repository=memory_repository,
        context_manager=context_manager,
    )
