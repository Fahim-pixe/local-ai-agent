"""Assembly of the initial safe local-agent execution path."""

from __future__ import annotations

from dataclasses import dataclass

from local_ai_agent.config import Settings
from local_ai_agent.runtime.budget_manager import BudgetManager
from local_ai_agent.runtime.loop_detector import LoopDetector
from local_ai_agent.runtime.ollama_client import OllamaClient
from local_ai_agent.runtime.permission_gate import PermissionGate
from local_ai_agent.runtime.react_loop import NativeToolChatClient, ReActLoop
from local_ai_agent.runtime.retry_engine import RetryEngine
from local_ai_agent.runtime.tool_router import ToolRouter
from local_ai_agent.runtime.verification_engine import VerificationEngine
from local_ai_agent.schemas.contracts import RunBudget
from local_ai_agent.tools.filesystem import build_read_only_filesystem_tools
from local_ai_agent.tools.registry import ToolRegistry


@dataclass(frozen=True, slots=True)
class MinimalRuntime:
    """Dependencies for the first safe ReAct capability: workspace-only reads."""

    registry: ToolRegistry
    tool_router: ToolRouter
    react_loop: ReActLoop


def build_minimal_runtime(
    settings: Settings, client: NativeToolChatClient | None = None
) -> MinimalRuntime:
    """Assemble the bounded read-only tool path without registering write or shell tools."""
    registry = ToolRegistry()
    for definition in build_read_only_filesystem_tools(settings):
        registry.register(definition)
    router = ToolRouter(
        registry=registry,
        permission_gate=PermissionGate(),
        budget_manager=BudgetManager(
            RunBudget(
                max_tool_calls=settings.default_max_tool_calls,
                max_runtime_seconds=settings.default_max_runtime_seconds,
                max_shell_executions=settings.default_max_shell_executions,
            )
        ),
        loop_detector=LoopDetector(),
        verification_engine=VerificationEngine(),
        retry_engine=RetryEngine(),
    )
    loop = ReActLoop(
        settings=settings,
        client=client or OllamaClient(settings.ollama_base_url),
        registry=registry,
        tool_router=router,
    )
    return MinimalRuntime(registry=registry, tool_router=router, react_loop=loop)
