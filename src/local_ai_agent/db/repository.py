"""SQLite persistence gateway; this module remains the source of truth for run state."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from local_ai_agent.db.schema import initialize_database
from local_ai_agent.schemas.contracts import AgentEvent, AgentPlan, AgentRun, AgentState, ToolResult


class RunRepository:
    """Persist run lifecycle, audit records, control state, and workspace ownership."""

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
        now = self._now()
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
                    run.plan.model_dump_json() if run.plan else None,
                    run.budget.model_dump_json(),
                    str(run.resume_token),
                    run.prompt_hash,
                    now,
                    now,
                ),
            )
            connection.execute(
                "INSERT INTO run_controls (run_id, updated_at) VALUES (?, ?)", (str(run.id), now)
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
        return self._row_to_run(row) if row else None

    def list_runs(self, limit: int = 50) -> list[AgentRun]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM agent_runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def update_run(
        self, *, run_id: UUID, state: AgentState, plan: AgentPlan | None = None
    ) -> AgentRun | None:
        now = self._now()
        completed_at = (
            now
            if state
            in {AgentState.COMPLETE, AgentState.PARTIAL, AgentState.FAILED, AgentState.CANCELLED}
            else None
        )
        with self.connect() as connection:
            if plan is None:
                cursor = connection.execute(
                    "UPDATE agent_runs SET state = ?, updated_at = ?, completed_at = COALESCE(?, completed_at) WHERE id = ?",
                    (state.value, now, completed_at, str(run_id)),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE agent_runs
                    SET state = ?, plan_json = ?, updated_at = ?, completed_at = COALESCE(?, completed_at)
                    WHERE id = ?
                    """,
                    (state.value, plan.model_dump_json(), now, completed_at, str(run_id)),
                )
        return self.get_run(run_id) if cursor.rowcount else None

    def record_event(self, event: AgentEvent) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO agent_events (run_id, event_type, state, message, data_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event.run_id),
                    event.type,
                    event.state.value,
                    event.message,
                    json.dumps(event.data, sort_keys=True),
                    event.created_at.isoformat(),
                ),
            )

    def list_events(self, run_id: UUID, after_id: int = 0) -> list[AgentEvent]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT id, event_type, state, message, data_json, created_at
                FROM agent_events WHERE run_id = ? AND id > ? ORDER BY id ASC
                """,
                (str(run_id), after_id),
            ).fetchall()
        return [
            AgentEvent(
                run_id=run_id,
                type=row["event_type"],
                state=AgentState(row["state"]),
                message=row["message"],
                data={**json.loads(row["data_json"]), "event_id": row["id"]},
                created_at=datetime.fromisoformat(row["created_at"]),
            )
            for row in rows
        ]

    def record_tool_call(
        self, *, run_id: UUID, tool_name: str, arguments: dict[str, Any], risk_level: str
    ) -> int:
        serialized_arguments = json.dumps(
            arguments, sort_keys=True, separators=(",", ":"), default=str
        )
        args_hash = hashlib.sha256(serialized_arguments.encode("utf-8")).hexdigest()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO tool_calls (run_id, tool_name, arguments_json, risk_level, args_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (str(run_id), tool_name, serialized_arguments, risk_level, args_hash, self._now()),
            )
        return int(cursor.lastrowid)

    def record_tool_result(
        self, *, tool_call_id: int, result: ToolResult, duration_ms: int | None = None
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO tool_results (
                    tool_call_id, status, success, data_json, error_code, error_message,
                    retryable, verified, verification_json, metadata_json, duration_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tool_call_id,
                    result.status.value,
                    int(result.success),
                    json.dumps(result.data, default=str) if result.data is not None else None,
                    result.error_code,
                    result.error_message,
                    int(result.retryable),
                    int(result.verified),
                    result.verification.model_dump_json() if result.verification else None,
                    json.dumps(result.metadata, default=str, sort_keys=True),
                    duration_ms,
                    self._now(),
                ),
            )

    def record_file_backup(
        self, *, run_id: UUID, original_path: Path, backup_path: Path, original_hash: str
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO file_backups (run_id, original_path, backup_path, original_hash, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(run_id), str(original_path), str(backup_path), original_hash, self._now()),
            )
        return int(cursor.lastrowid)

    def mark_backup_restored(self, backup_id: int) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE file_backups SET restored_at = ? WHERE id = ?", (self._now(), backup_id)
            )

    def request_cancellation(self, run_id: UUID) -> bool:
        return self._update_control(run_id, "cancel_requested = 1")

    def cancellation_requested(self, run_id: UUID) -> bool:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT cancel_requested FROM run_controls WHERE run_id = ?", (str(run_id),)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def set_pending_authorization(self, run_id: UUID, payload: dict[str, Any]) -> bool:
        return self._update_control(
            run_id,
            "pending_authorization_json = ?",
            (json.dumps(payload, default=str, sort_keys=True),),
        )

    def get_pending_authorization(self, run_id: UUID) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT pending_authorization_json FROM run_controls WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
        return (
            json.loads(row["pending_authorization_json"])
            if row and row["pending_authorization_json"]
            else None
        )

    def resolve_authorization(self, run_id: UUID) -> bool:
        return self._update_control(run_id, "pending_authorization_json = NULL")

    def acquire_session_lock(self, *, workspace_id: str, run_id: UUID) -> bool:
        try:
            with self.connect() as connection:
                connection.execute(
                    "INSERT INTO session_locks (workspace_id, run_id, acquired_at) VALUES (?, ?, ?)",
                    (workspace_id, str(run_id), self._now()),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def release_session_lock(self, *, workspace_id: str, run_id: UUID) -> None:
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM session_locks WHERE workspace_id = ? AND run_id = ?",
                (workspace_id, str(run_id)),
            )

    def health_check(self) -> bool:
        with self.connect() as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1

    def _update_control(self, run_id: UUID, assignment: str, values: tuple[Any, ...] = ()) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                f"UPDATE run_controls SET {assignment}, updated_at = ? WHERE run_id = ?",
                (*values, self._now(), str(run_id)),
            )
        return bool(cursor.rowcount)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> AgentRun:
        return AgentRun(
            id=UUID(row["id"]),
            objective=row["objective"],
            workspace_id=row["workspace_id"],
            state=AgentState(row["state"]),
            plan=AgentPlan.model_validate_json(row["plan_json"]) if row["plan_json"] else None,
            budget=json.loads(row["budget_json"]),
            resume_token=UUID(row["resume_token"]),
            prompt_hash=row["prompt_hash"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
