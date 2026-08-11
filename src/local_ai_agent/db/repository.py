"""SQLite persistence gateway; this module remains the source of truth for run state."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from local_ai_agent.db.schema import initialize_database
from local_ai_agent.schemas.contracts import AgentRun, AgentState


class RunRepository:
    """Minimal run persistence used by the setup scaffold and API lifecycle routes."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, isolation_level="IMMEDIATE")
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            initialize_database(connection)

    def create_run(self, run: AgentRun) -> AgentRun:
        now = datetime.now(UTC).isoformat()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (
                    id, workspace_id, objective, state, plan_json, budget_json,
                    resume_token, prompt_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run.id),
                    run.workspace_id,
                    run.objective,
                    run.state.value,
                    None,
                    run.budget.model_dump_json(),
                    str(run.resume_token),
                    run.prompt_hash,
                    now,
                    now,
                ),
            )
        return run.model_copy(
            update={
                "created_at": datetime.fromisoformat(now),
                "updated_at": datetime.fromisoformat(now),
            }
        )

    def get_run(self, run_id: UUID) -> AgentRun | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_runs WHERE id = ?", (str(run_id),)
            ).fetchone()
        if row is None:
            return None
        return AgentRun(
            id=UUID(row["id"]),
            objective=row["objective"],
            workspace_id=row["workspace_id"],
            state=AgentState(row["state"]),
            plan=json.loads(row["plan_json"]) if row["plan_json"] else None,
            budget=json.loads(row["budget_json"]),
            resume_token=UUID(row["resume_token"]),
            prompt_hash=row["prompt_hash"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def health_check(self) -> bool:
        with self.connect() as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1
