from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from local_ai_agent.api.app import create_app
from local_ai_agent.config import ensure_workspace, load_settings
from local_ai_agent.db.repository import RunRepository
from local_ai_agent.runtime.lifecycle import LifecycleError, RunLifecycleService
from local_ai_agent.runtime.production_prompt import load_production_prompt
from local_ai_agent.runtime.secure_run_runtime import build_secure_run_runtime
from local_ai_agent.schemas.contracts import AgentRun, AgentState, RunBudget


class RecordingNativeToolClient:
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
    prompt_path = tmp_path / "config" / "system_prompt.md"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_bytes(b"# Production Prompt v1\n\nUse verified tools only.\n")
    settings = replace(
        load_settings(),
        workspace_root=tmp_path / "workspace",
        sqlite_path=tmp_path / "workspace" / ".agent" / "agent.db",
        agent_api_token=None,
        system_prompt_path=prompt_path,
    )
    ensure_workspace(settings)
    return settings


def test_versioned_production_prompt_loads_content_and_sha256_from_configured_path(
    tmp_path: Path,
) -> None:
    settings = configured_settings(tmp_path)

    prompt = load_production_prompt(settings)

    expected_content = "# Production Prompt v1\n\nUse verified tools only.\n"
    assert prompt.content == expected_content
    assert prompt.source_path == settings.system_prompt_path
    assert prompt.sha256 == hashlib.sha256(expected_content.encode("utf-8")).hexdigest()


def test_production_prompt_renders_toml_identity_before_hashing(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    settings.system_prompt_path.write_bytes(b"# {{AGENT_NAME}}\n\nMission: {{AGENT_MISSION}}\n")

    prompt = load_production_prompt(settings)

    expected_content = f"# {settings.agent_name}\n\nMission: {settings.agent_mission}\n"
    assert prompt.content == expected_content
    assert prompt.sha256 == hashlib.sha256(expected_content.encode("utf-8")).hexdigest()
    assert "{{AGENT_NAME}}" not in prompt.content
    assert "{{AGENT_MISSION}}" not in prompt.content


def test_secure_runtime_uses_configured_production_prompt_and_native_tool_schemas(
    tmp_path: Path,
) -> None:
    settings = configured_settings(tmp_path)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    lifecycle = RunLifecycleService(repository)
    prompt = load_production_prompt(settings)
    run = lifecycle.register_run(
        AgentRun(
            objective="List the project files.",
            workspace_id="production-prompt-runtime",
            budget=RunBudget(max_tool_calls=5, max_runtime_seconds=60, max_shell_executions=0),
            prompt_hash=prompt.sha256,
        )
    )
    client = RecordingNativeToolClient(
        [{"message": {"role": "assistant", "content": "No files need to be listed."}}]
    )
    runtime = build_secure_run_runtime(
        settings=settings,
        run_id=run.id,
        repository=repository,
        lifecycle=lifecycle,
        client=client,
    )

    result = asyncio.run(runtime.run_with_context())

    assert result.state is AgentState.COMPLETE
    assert client.calls[0]["messages"][0] == {"role": "system", "content": prompt.content}
    assert client.calls[0]["tools"] is not None
    assert any(
        tool["function"]["name"] == "filesystem.list_directory" for tool in client.calls[0]["tools"]
    )


def test_secure_runtime_rejects_changed_production_prompt_for_hashed_run(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    lifecycle = RunLifecycleService(repository)
    prompt = load_production_prompt(settings)
    run = lifecycle.register_run(
        AgentRun(
            objective="List files.",
            workspace_id="prompt-drift-runtime",
            budget=RunBudget(max_tool_calls=5, max_runtime_seconds=60, max_shell_executions=0),
            prompt_hash=prompt.sha256,
        )
    )
    settings.system_prompt_path.write_text("# Production Prompt v2\n", encoding="utf-8")
    runtime = build_secure_run_runtime(
        settings=settings,
        run_id=run.id,
        repository=repository,
        lifecycle=lifecycle,
        client=RecordingNativeToolClient([]),
    )

    with pytest.raises(LifecycleError, match="does not match"):
        asyncio.run(runtime.run_with_context())


def test_run_api_persists_runtime_owned_production_prompt_hash(tmp_path: Path) -> None:
    settings = configured_settings(tmp_path)
    app = create_app(settings)
    expected_hash = load_production_prompt(settings).sha256

    with TestClient(app) as client:
        response = client.post(
            "/runs",
            json={"objective": "Read the project README.", "workspace_id": "prompt-hash-api"},
        )
        assert response.status_code == 202
        payload = response.json()
        assert payload["prompt_hash"] == expected_hash

        persisted = client.app.state.repository.get_run(payload["id"])
        assert persisted is not None
        assert persisted.prompt_hash == expected_hash
