from pathlib import Path

import pytest

from local_ai_agent.config import ensure_workspace, load_settings
from local_ai_agent.db.repository import RunRepository
from local_ai_agent.runtime.output_validator import ModelOutputValidationError, OutputValidator
from local_ai_agent.runtime.state_machine import InvalidStateTransition, StateMachine
from local_ai_agent.schemas.contracts import AgentPlan, AgentRun, AgentState, RiskLevel, RunBudget
from local_ai_agent.security.paths import WorkspacePolicyError, resolve_workspace_path


def test_tool_plan_contract_is_typed() -> None:
    plan = AgentPlan.model_validate(
        {
            "goal": "List the project files",
            "steps": [
                {
                    "id": 1,
                    "action": "filesystem.list_directory",
                    "description": "Inspect the project directory",
                    "depends_on": [],
                    "risk": "LOW",
                }
            ],
            "success_criteria": ["The directory listing is returned."],
            "rollback_strategy": "No changes are made.",
        }
    )
    assert plan.steps[0].risk is RiskLevel.LOW


def test_output_validator_rejects_ambiguous_turn() -> None:
    with pytest.raises(ModelOutputValidationError):
        OutputValidator.validate_turn(has_tool_calls=True, final_content="Task complete.")


def test_state_machine_rejects_arbitrary_transition() -> None:
    machine = StateMachine()
    with pytest.raises(InvalidStateTransition):
        machine.transition_to(AgentState.COMPLETE)


def test_workspace_boundary_resolves_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (workspace / "escape").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation requires Developer Mode or elevated privileges.")
        raise

    with pytest.raises(WorkspacePolicyError):
        resolve_workspace_path(workspace_project=workspace, candidate="escape/private.txt")


def test_sqlite_bootstrap_and_run_round_trip(tmp_path: Path) -> None:
    database_path = tmp_path / "workspace" / ".agent" / "agent.db"
    repository = RunRepository(database_path)
    repository.initialize()
    run = AgentRun(
        objective="List project files",
        workspace_id="test",
        budget=RunBudget(max_tool_calls=5, max_runtime_seconds=60, max_shell_executions=1),
    )

    repository.create_run(run)
    restored = repository.get_run(run.id)

    assert restored is not None
    assert restored.objective == "List project files"
    assert restored.state is AgentState.UNDERSTAND


def test_workspace_bootstrap_uses_configured_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path / "workspace"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "workspace" / ".agent" / "agent.db"))
    settings = load_settings()
    ensure_workspace(settings)

    assert settings.workspace_project_path.is_dir()
    assert settings.workspace_internal_path.is_dir()
