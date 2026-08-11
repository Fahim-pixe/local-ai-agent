from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID

from local_ai_agent.config import Settings, ensure_workspace
from local_ai_agent.db.repository import RunRepository
from local_ai_agent.runtime.lifecycle import RunLifecycleService
from local_ai_agent.runtime.production_prompt import load_production_prompt
from local_ai_agent.runtime.secure_run_runtime import build_secure_run_runtime
from local_ai_agent.schemas.contracts import AgentRun, AgentState, RunBudget


@dataclass(frozen=True, slots=True)
class CodingTask:
    """A deterministic local coding scenario scored from runtime evidence, not model prose."""

    name: str
    objective: str
    fixture_files: dict[str, str]
    expected_files: dict[str, str]
    expected_tool: str


@dataclass(frozen=True, slots=True)
class CodingTaskEvaluationResult:
    """Evidence-backed result for one local-model coding task."""

    task: CodingTask
    run_id: str
    state: AgentState
    verified_expected_tool: bool
    workspace_outcome_verified: bool
    passed: bool
    failure_reason: str | None


def coding_task_corpus() -> tuple[CodingTask, ...]:
    """Return the fixed Phase 6 corpus that exercises native read and write tool paths."""
    return (
        CodingTask(
            name="list-project-files",
            objective=(
                "Use filesystem.list_directory with path '.' to inspect the project. "
                "Then report the verified file names."
            ),
            fixture_files={
                "README.md": "Local evaluation README.\n",
                "src/app.py": "print('ready')\n",
            },
            expected_files={
                "README.md": "Local evaluation README.\n",
                "src/app.py": "print('ready')\n",
            },
            expected_tool="filesystem.list_directory",
        ),
        CodingTask(
            name="read-readme",
            objective=(
                "Use filesystem.read_file to read README.md. "
                "Then report the exact verified phrase: Local evaluation README."
            ),
            fixture_files={"README.md": "Local evaluation README.\n"},
            expected_files={"README.md": "Local evaluation README.\n"},
            expected_tool="filesystem.read_file",
        ),
        CodingTask(
            name="write-hello-world",
            objective=(
                "Use filesystem.write_file to create hello.py with exactly this UTF-8 content: "
                "print('hello, world!') followed by one newline. Then report the verified result."
            ),
            fixture_files={"README.md": "Local evaluation README.\n"},
            expected_files={"hello.py": "print('hello, world!')\n"},
            expected_tool="filesystem.write_file",
        ),
    )


async def run_coding_evaluation(settings: Settings) -> tuple[CodingTaskEvaluationResult, ...]:
    """Run every corpus task in a separate workspace through a real configured Ollama runtime."""
    results: list[CodingTaskEvaluationResult] = []
    for task in coding_task_corpus():
        results.append(await _run_task(settings, task))
    return tuple(results)


async def _run_task(settings: Settings, task: CodingTask) -> CodingTaskEvaluationResult:
    task_settings = _task_settings(settings, task.name)
    ensure_workspace(task_settings)
    _write_fixture_files(task_settings.workspace_project_path, task.fixture_files)
    repository = RunRepository(task_settings.sqlite_path)
    repository.initialize()
    lifecycle = RunLifecycleService(repository)
    prompt = load_production_prompt(task_settings)
    run = lifecycle.register_run(
        AgentRun(
            objective=task.objective,
            workspace_id=f"evaluation-{task.name}",
            budget=RunBudget(
                max_tool_calls=task_settings.default_max_tool_calls,
                max_runtime_seconds=task_settings.default_max_runtime_seconds,
                max_shell_executions=task_settings.default_max_shell_executions,
            ),
            prompt_hash=prompt.sha256,
        )
    )
    runtime = build_secure_run_runtime(
        settings=task_settings,
        run_id=run.id,
        repository=repository,
        lifecycle=lifecycle,
    )
    react_result = await runtime.run_with_context()
    verified_expected_tool = _verified_tool_call(repository, run.id, task.expected_tool)
    workspace_outcome_verified = _expected_files_match(
        task_settings.workspace_project_path, task.expected_files
    )
    state_is_complete = react_result.state is AgentState.COMPLETE
    passed = state_is_complete and verified_expected_tool and workspace_outcome_verified
    return CodingTaskEvaluationResult(
        task=task,
        run_id=str(run.id),
        state=react_result.state,
        verified_expected_tool=verified_expected_tool,
        workspace_outcome_verified=workspace_outcome_verified,
        passed=passed,
        failure_reason=None
        if passed
        else _failure_reason(
            state_is_complete=state_is_complete,
            verified_expected_tool=verified_expected_tool,
            workspace_outcome_verified=workspace_outcome_verified,
        ),
    )


def _task_settings(settings: Settings, task_name: str) -> Settings:
    task_workspace = settings.workspace_root / "evaluations" / task_name
    return replace(
        settings,
        workspace_root=task_workspace,
        sqlite_path=task_workspace / ".agent" / "agent.db",
    )


def _write_fixture_files(project_path: Path, fixture_files: dict[str, str]) -> None:
    for relative_path, content in fixture_files.items():
        target = project_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _verified_tool_call(repository: RunRepository, run_id: UUID, tool_name: str) -> bool:
    with repository.connect() as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM tool_calls AS call
            INNER JOIN tool_results AS result ON result.tool_call_id = call.id
            WHERE call.run_id = ? AND call.tool_name = ?
              AND result.success = 1 AND result.verified = 1
            LIMIT 1
            """,
            (str(run_id), tool_name),
        ).fetchone()
    return row is not None


def _expected_files_match(project_path: Path, expected_files: dict[str, str]) -> bool:
    for relative_path, expected_content in expected_files.items():
        target = project_path / relative_path
        if not target.is_file() or target.read_text(encoding="utf-8") != expected_content:
            return False
    return True


def _failure_reason(
    *, state_is_complete: bool, verified_expected_tool: bool, workspace_outcome_verified: bool
) -> str:
    failures: list[str] = []
    if not state_is_complete:
        failures.append("run did not reach COMPLETE")
    if not verified_expected_tool:
        failures.append("expected tool was not successful and verified")
    if not workspace_outcome_verified:
        failures.append("workspace outcome did not match the task contract")
    return "; ".join(failures)
