from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from pathlib import Path

import pytest

from local_ai_agent.config import ensure_workspace, load_settings
from local_ai_agent.db.repository import RunRepository
from local_ai_agent.evaluation.coding_tasks import (
    _enter_execution,
    _failure_reason,
    coding_task_corpus,
    run_coding_evaluation,
)
from local_ai_agent.runtime.lifecycle import RunLifecycleService
from local_ai_agent.schemas.contracts import AgentRun, AgentState, RunBudget


def test_coding_task_corpus_covers_required_read_and_write_paths() -> None:
    tasks = coding_task_corpus()

    assert [task.name for task in tasks] == [
        "list-project-files",
        "read-readme",
        "write-hello-world",
    ]
    assert [task.expected_tool for task in tasks] == [
        "filesystem.list_directory",
        "filesystem.read_file",
        "filesystem.write_file",
    ]
    assert all(task.fixture_files for task in tasks)
    assert tasks[-1].expected_files["hello.py"] == "print('hello, world!')\n"


def test_evaluation_enters_execute_state_before_model_tool_calls(tmp_path: Path) -> None:
    settings = replace(
        load_settings(),
        workspace_root=tmp_path / "workspace",
        sqlite_path=tmp_path / "workspace" / ".agent" / "agent.db",
    )
    ensure_workspace(settings)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    lifecycle = RunLifecycleService(repository)
    run = lifecycle.register_run(
        AgentRun(
            objective="List files.",
            workspace_id="evaluation-lifecycle",
            budget=RunBudget(max_tool_calls=3, max_runtime_seconds=60, max_shell_executions=0),
        )
    )

    _enter_execution(lifecycle, run.id)

    assert repository.get_run(run.id).state is AgentState.EXECUTE


def test_evaluation_failure_reason_preserves_model_diagnostic() -> None:
    reason = _failure_reason(
        state_is_complete=False,
        verified_expected_tool=False,
        workspace_outcome_verified=False,
        react_error_code="MODEL_OUTPUT_INVALID",
        react_error_message="Model response contained neither tool calls nor final content.",
    )

    assert "model_error=MODEL_OUTPUT_INVALID" in reason
    assert "Model response contained neither tool calls nor final content." in reason


@pytest.mark.ollama
def test_real_ollama_coding_evaluation_is_opt_in_and_uses_verified_runtime_paths(
    tmp_path: Path,
) -> None:
    if os.getenv("RUN_OLLAMA_EVALUATION") != "1":
        pytest.skip("Set RUN_OLLAMA_EVALUATION=1 to run the local Qwen3 coding evaluation.")
    settings = replace(
        load_settings(),
        workspace_root=tmp_path / "workspace",
        sqlite_path=tmp_path / "workspace" / ".agent" / "agent.db",
    )

    results = asyncio.run(run_coding_evaluation(settings))

    assert [result.task.name for result in results] == [
        "list-project-files",
        "read-readme",
        "write-hello-world",
    ]
    assert all(result.passed for result in results), [result.failure_reason for result in results]
    assert all(result.verified_expected_tool for result in results)
    assert all(result.workspace_outcome_verified for result in results)
