from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any

from local_ai_agent.config import ensure_workspace, load_settings
from local_ai_agent.runtime.budget_manager import BudgetManager
from local_ai_agent.runtime.loop_detector import LoopDetector
from local_ai_agent.runtime.minimal_runtime import build_minimal_runtime
from local_ai_agent.runtime.permission_gate import PermissionGate
from local_ai_agent.runtime.react_loop import ReActLoop
from local_ai_agent.runtime.retry_engine import RetryEngine
from local_ai_agent.runtime.tool_router import ToolRouter
from local_ai_agent.runtime.verification_engine import VerificationEngine
from local_ai_agent.schemas.contracts import AgentState, RunBudget
from local_ai_agent.tools.filesystem import build_read_only_filesystem_tools
from local_ai_agent.tools.registry import ToolRegistry


class FakeNativeToolClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.calls.append({"model": model, "messages": list(messages), "tools": tools})
        return self._responses.pop(0)


def configured_settings(tmp_path: Path):
    settings = replace(
        load_settings(),
        workspace_root=tmp_path / "workspace",
        sqlite_path=tmp_path / "workspace" / ".agent" / "agent.db",
        default_max_tool_calls=5,
        filesystem_max_read_bytes=64,
        filesystem_max_list_entries=10,
    )
    ensure_workspace(settings)
    return settings


def configured_router(settings) -> tuple[ToolRegistry, ToolRouter]:
    registry = ToolRegistry()
    for definition in build_read_only_filesystem_tools(settings):
        registry.register(definition)
    router = ToolRouter(
        registry=registry,
        permission_gate=PermissionGate(),
        budget_manager=BudgetManager(
            RunBudget(max_tool_calls=5, max_runtime_seconds=60, max_shell_executions=0)
        ),
        loop_detector=LoopDetector(repeat_threshold=3),
        verification_engine=VerificationEngine(),
        retry_engine=RetryEngine(),
    )
    return registry, router


def test_minimal_runtime_registers_only_read_only_filesystem_tools(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    client = FakeNativeToolClient([])

    runtime = build_minimal_runtime(settings, client=client)

    assert [tool["function"]["name"] for tool in runtime.registry.ollama_tools()] == [
        "filesystem.list_directory",
        "filesystem.read_file",
    ]


def test_list_directory_is_workspace_bounded_and_independently_verified(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    (settings.workspace_project_path / "notes").mkdir()
    (settings.workspace_project_path / "README.md").write_text("local agent", encoding="utf-8")
    registry, router = configured_router(settings)

    outcome = asyncio.run(
        router.execute(tool_name="filesystem.list_directory", arguments={"path": "."})
    )

    assert registry.get("filesystem.list_directory").risk.value == "LOW"
    assert outcome.result.success is True
    assert outcome.result.verified is True
    assert outcome.result.verification is not None
    assert outcome.result.verification.strategy == "directory-relist"
    assert outcome.result.data["entries"] == [
        {"name": "README.md", "type": "file"},
        {"name": "notes", "type": "directory"},
    ]


def test_read_file_is_bounded_and_verified_by_reread_hash(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    (settings.workspace_project_path / "note.txt").write_text("abcdef", encoding="utf-8")
    _, router = configured_router(replace(settings, filesystem_max_read_bytes=4))

    outcome = asyncio.run(
        router.execute(tool_name="filesystem.read_file", arguments={"path": "note.txt"})
    )

    assert outcome.result.success is True
    assert outcome.result.verified is True
    assert outcome.result.verification is not None
    assert outcome.result.verification.strategy == "file-reread-hash"
    assert outcome.result.data["content"] == "abcd"
    assert outcome.result.data["truncated"] is True


def test_read_only_tool_blocks_workspace_escape(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    _, router = configured_router(settings)

    outcome = asyncio.run(
        router.execute(tool_name="filesystem.read_file", arguments={"path": "../.agent/agent.db"})
    )

    assert outcome.result.success is False
    assert outcome.result.error_code == "POLICY_BLOCK"
    assert outcome.result.tool_name == "filesystem.read_file"
    assert outcome.result.verified is False


def test_react_loop_executes_native_tool_call_and_returns_final_content(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    (settings.workspace_project_path / "project.txt").write_text("ready", encoding="utf-8")
    registry, router = configured_router(settings)
    client = FakeNativeToolClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "filesystem.list_directory",
                                "arguments": {"path": "."},
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "The project contains project.txt."}},
        ]
    )
    loop = ReActLoop(settings=settings, client=client, registry=registry, tool_router=router)

    result = asyncio.run(
        loop.run(objective="List files in the project.", system_prompt="Use tools carefully.")
    )

    assert result.state is AgentState.COMPLETE
    assert result.final_response == "The project contains project.txt."
    assert len(result.tool_outcomes) == 1
    assert result.tool_outcomes[0].result.verified is True
    assert len(client.calls) == 2
    assert client.calls[0]["tools"] is not None
    assert client.calls[1]["messages"][-1]["role"] == "tool"


def test_react_loop_discards_tool_accompanying_text_before_native_tool_execution(
    tmp_path: Path,
) -> None:
    settings = configured_settings(tmp_path)
    registry, router = configured_router(settings)
    client = FakeNativeToolClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "I completed the task.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "filesystem.list_directory",
                                "arguments": {"path": "."},
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "Verified after the tool result."}},
        ]
    )
    loop = ReActLoop(settings=settings, client=client, registry=registry, tool_router=router)

    result = asyncio.run(loop.run(objective="List files.", system_prompt="Use tools carefully."))

    assert result.state is AgentState.COMPLETE
    assert result.final_response == "Verified after the tool result."
    assert len(result.tool_outcomes) == 1
    assert result.tool_outcomes[0].result.verified is True
    assert client.calls[1]["messages"][-1]["role"] == "tool"
    assert all(message.get("content") != "I completed the task." for message in result.messages)


def test_react_loop_stops_at_runtime_budget_without_final_content(tmp_path: Path) -> None:
    settings = replace(configured_settings(tmp_path), default_max_tool_calls=1)
    (settings.workspace_project_path / "project.txt").write_text("ready", encoding="utf-8")
    registry, router = configured_router(settings)
    router = ToolRouter(
        registry=registry,
        permission_gate=PermissionGate(),
        budget_manager=BudgetManager(
            RunBudget(max_tool_calls=1, max_runtime_seconds=60, max_shell_executions=0)
        ),
        loop_detector=LoopDetector(repeat_threshold=10),
        verification_engine=VerificationEngine(),
        retry_engine=RetryEngine(),
    )
    client = FakeNativeToolClient(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "filesystem.list_directory",
                                "arguments": {"path": "."},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "filesystem.list_directory",
                                "arguments": {"path": "."},
                            }
                        }
                    ],
                }
            },
        ]
    )
    loop = ReActLoop(settings=settings, client=client, registry=registry, tool_router=router)

    result = asyncio.run(
        loop.run(objective="Keep listing files.", system_prompt="Use tools carefully.")
    )

    assert result.state is AgentState.PARTIAL
    assert result.error_code == "MAX_REACT_TURNS"
    assert result.tool_outcomes[-1].result.error_code == "MAX_TOOL_CALLS"
