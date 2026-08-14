"""SQLite persistence gateway; this module remains the source of truth for run state."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from local_ai_agent.db.schema import initialize_database
from local_ai_agent.schemas.contracts import (
    AgentEvent,
    AgentPlan,
    AgentRun,
    AgentState,
    OperationalMetrics,
    RecoveryClass,
    ToolResult,
)


@dataclass(frozen=True, slots=True)
class ReActCheckpoint:
    id: int
    run_id: UUID
    sequence: int
    phase: str
    messages: list[dict[str, Any]]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class PendingAction:
    id: UUID
    run_id: UUID
    tool_name: str
    arguments: dict[str, Any]
    risk_level: str
    checkpoint_id: int | None
    status: str
    approved_at: datetime | None
    claimed_at: datetime | None
    worker_id: str | None
    lease_expires_at: datetime | None
    recovered_at: datetime | None
    recovery_reason: str | None
    recovery_class: RecoveryClass
    recovery_contract_version: int
    operation_key: str | None
    dispatch_attempt: int
    max_dispatch_attempts: int
    available_at: datetime | None
    previous_worker_id: str | None
    recovery_verification: dict[str, Any] | None
    executed_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class WorkerRecord:
    worker_id: str
    hostname: str
    process_id: int
    capabilities: tuple[str, ...]
    state: str
    started_at: datetime
    last_heartbeat_at: datetime
    stopped_at: datetime | None


@dataclass(frozen=True, slots=True)
class ActionAttempt:
    id: int
    action_id: UUID
    attempt: int
    worker_id: str
    status: str
    detail: dict[str, Any] | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class DispatchClaim:
    action: PendingAction
    worker_id: str
    attempt: int
    lease_expires_at: datetime


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

    def operational_metrics(self) -> OperationalMetrics:
        """Return privacy-safe aggregates from the authoritative local audit store."""
        with self.connect() as connection:
            state_rows = connection.execute(
                "SELECT state, COUNT(*) AS count FROM agent_runs GROUP BY state ORDER BY state"
            ).fetchall()
            tool_rows = connection.execute(
                """
                SELECT
                    COUNT(*) AS result_count,
                    COALESCE(SUM(CASE WHEN success = 1 AND verified = 1 THEN 1 ELSE 0 END), 0)
                        AS verified_successes,
                    COALESCE(SUM(CASE WHEN error_code = 'LOOP_DETECTED' THEN 1 ELSE 0 END), 0)
                        AS loop_stops,
                    COALESCE(SUM(CASE WHEN error_code LIKE 'MAX_%' THEN 1 ELSE 0 END), 0)
                        AS budget_stops
                FROM tool_results
                """
            ).fetchone()
            action_rows = connection.execute(
                """
                SELECT
                    COUNT(*) AS request_count,
                    COALESCE(SUM(CASE WHEN approved_at IS NOT NULL THEN 1 ELSE 0 END), 0)
                        AS approved_count,
                    COALESCE(SUM(CASE WHEN status = 'REJECTED' THEN 1 ELSE 0 END), 0)
                        AS denied_count,
                    COALESCE(SUM(CASE WHEN status = 'EXECUTED' THEN 1 ELSE 0 END), 0)
                        AS executed_count
                FROM pending_actions
                """
            ).fetchone()
            runs_total = int(connection.execute("SELECT COUNT(*) FROM agent_runs").fetchone()[0])
            tool_calls_total = int(
                connection.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0]
            )
            action_recoveries = int(
                connection.execute(
                    "SELECT COUNT(*) FROM action_attempts WHERE status = 'RECOVERED'"
                ).fetchone()[0]
            )
            continuations_replayed = int(
                connection.execute(
                    "SELECT COUNT(*) FROM agent_events WHERE event_type = 'continuation.replayed'"
                ).fetchone()[0]
            )
        return OperationalMetrics(
            runs_total=runs_total,
            runs_by_state={str(row["state"]): int(row["count"]) for row in state_rows},
            tool_calls_total=tool_calls_total,
            tool_results_total=int(tool_rows["result_count"]),
            verified_tool_successes=int(tool_rows["verified_successes"]),
            loop_stops=int(tool_rows["loop_stops"]),
            budget_stops=int(tool_rows["budget_stops"]),
            authorization_requests=int(action_rows["request_count"]),
            authorization_approved=int(action_rows["approved_count"]),
            authorization_denied=int(action_rows["denied_count"]),
            authorization_executed=int(action_rows["executed_count"]),
            action_recoveries=action_recoveries,
            continuations_replayed=continuations_replayed,
        )

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

    def save_react_checkpoint(
        self, *, run_id: UUID, phase: str, messages: list[dict[str, Any]]
    ) -> ReActCheckpoint:
        now = self._now()
        with self.connect() as connection:
            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM react_checkpoints WHERE run_id = ?",
                    (str(run_id),),
                ).fetchone()[0]
            )
            cursor = connection.execute(
                """
                INSERT INTO react_checkpoints (run_id, sequence, phase, messages_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(run_id), sequence, phase, json.dumps(messages, default=str), now),
            )
        return ReActCheckpoint(
            id=int(cursor.lastrowid),
            run_id=run_id,
            sequence=sequence,
            phase=phase,
            messages=messages,
            created_at=datetime.fromisoformat(now),
        )

    def get_react_checkpoint(self, checkpoint_id: int) -> ReActCheckpoint | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM react_checkpoints WHERE id = ?", (checkpoint_id,)
            ).fetchone()
        return self._row_to_checkpoint(row) if row else None

    def latest_react_checkpoint(self, run_id: UUID) -> ReActCheckpoint | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM react_checkpoints WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
                (str(run_id),),
            ).fetchone()
        return self._row_to_checkpoint(row) if row else None

    def create_pending_action(
        self,
        *,
        run_id: UUID,
        tool_name: str,
        arguments: dict[str, Any],
        risk_level: str,
        checkpoint_id: int | None,
        recovery_class: RecoveryClass = RecoveryClass.NEVER_RECLAIM,
        recovery_contract_version: int = 1,
        operation_key: str | None = None,
        max_dispatch_attempts: int = 1,
        available_at: datetime | None = None,
    ) -> PendingAction:
        now = self._now()
        action_id = uuid4()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO pending_actions (
                    id, run_id, tool_name, arguments_json, risk_level, checkpoint_id,
                    recovery_class, recovery_contract_version, operation_key,
                    max_dispatch_attempts, available_at, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?)
                """,
                (
                    str(action_id),
                    str(run_id),
                    tool_name,
                    json.dumps(arguments, sort_keys=True, default=str),
                    risk_level,
                    checkpoint_id,
                    recovery_class.value,
                    recovery_contract_version,
                    operation_key,
                    max_dispatch_attempts,
                    available_at.isoformat() if available_at else None,
                    now,
                    now,
                ),
            )
        return self.get_pending_action(run_id)  # type: ignore[return-value]

    def register_worker(
        self,
        *,
        worker_id: str,
        hostname: str,
        process_id: int,
        capabilities: tuple[str, ...],
        now: datetime | None = None,
    ) -> WorkerRecord:
        timestamp = (now or datetime.now(UTC)).isoformat()
        capabilities_json = json.dumps(sorted(capabilities), separators=(",", ":"))
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO workers (
                    worker_id, hostname, process_id, capabilities_json, state,
                    started_at, last_heartbeat_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    hostname = excluded.hostname,
                    process_id = excluded.process_id,
                    capabilities_json = excluded.capabilities_json,
                    state = 'ACTIVE',
                    last_heartbeat_at = excluded.last_heartbeat_at,
                    stopped_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    worker_id,
                    hostname,
                    process_id,
                    capabilities_json,
                    timestamp,
                    timestamp,
                    timestamp,
                    timestamp,
                ),
            )
        return self.get_worker(worker_id)  # type: ignore[return-value]

    def get_worker(self, worker_id: str) -> WorkerRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM workers WHERE worker_id = ?", (worker_id,)
            ).fetchone()
        return self._row_to_worker(row) if row else None

    def list_workers(self) -> list[WorkerRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM workers ORDER BY last_heartbeat_at DESC, worker_id ASC"
            ).fetchall()
        return [self._row_to_worker(row) for row in rows]

    def heartbeat_worker(self, worker_id: str, *, now: datetime | None = None) -> bool:
        timestamp = (now or datetime.now(UTC)).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE workers SET last_heartbeat_at = ?, updated_at = ?
                WHERE worker_id = ? AND state = 'ACTIVE'
                """,
                (timestamp, timestamp, worker_id),
            )
        return bool(cursor.rowcount)

    def drain_worker(self, worker_id: str, *, now: datetime | None = None) -> bool:
        timestamp = (now or datetime.now(UTC)).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE workers SET state = 'DRAINING', updated_at = ?
                WHERE worker_id = ? AND state = 'ACTIVE'
                """,
                (timestamp, worker_id),
            )
        return bool(cursor.rowcount)

    def stop_worker(self, worker_id: str, *, now: datetime | None = None) -> bool:
        timestamp = (now or datetime.now(UTC)).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE workers SET state = 'STOPPED', stopped_at = ?, updated_at = ?
                WHERE worker_id = ? AND state IN ('ACTIVE', 'DRAINING')
                """,
                (timestamp, timestamp, worker_id),
            )
        return bool(cursor.rowcount)

    def list_action_attempts(self, action_id: UUID) -> list[ActionAttempt]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM action_attempts WHERE action_id = ? ORDER BY id ASC",
                (str(action_id),),
            ).fetchall()
        return [self._row_to_action_attempt(row) for row in rows]

    def claim_next_dispatchable_action(
        self,
        *,
        worker_id: str,
        capabilities: tuple[str, ...],
        lease_seconds: int,
        now: datetime | None = None,
    ) -> DispatchClaim | None:
        """Atomically claim one eligible action for an active worker without executing it."""
        if not capabilities:
            return None
        claim_time = now or datetime.now(UTC)
        claim_value = claim_time.isoformat()
        lease_value = (claim_time + timedelta(seconds=lease_seconds)).isoformat()
        with self.connect() as connection:
            worker = connection.execute(
                "SELECT capabilities_json FROM workers WHERE worker_id = ? AND state = 'ACTIVE'",
                (worker_id,),
            ).fetchone()
            if worker is None:
                return None
            advertised = set(json.loads(worker["capabilities_json"]))
            eligible_capabilities = tuple(tool for tool in capabilities if tool in advertised)
            if not eligible_capabilities:
                return None
            placeholders = ", ".join("?" for _ in eligible_capabilities)
            row = connection.execute(
                f"""
                SELECT * FROM pending_actions
                WHERE status = 'APPROVED'
                  AND dispatch_attempt < max_dispatch_attempts
                  AND (available_at IS NULL OR available_at <= ?)
                  AND tool_name IN ({placeholders})
                ORDER BY approved_at ASC, created_at ASC
                LIMIT 1
                """,
                (claim_value, *eligible_capabilities),
            ).fetchone()
            if row is None:
                return None
            cursor = connection.execute(
                """
                UPDATE pending_actions
                SET status = 'EXECUTING', worker_id = ?, claimed_at = ?, lease_expires_at = ?,
                    dispatch_attempt = dispatch_attempt + 1, recovered_at = NULL,
                    recovery_reason = NULL, updated_at = ?
                WHERE id = ? AND status = 'APPROVED'
                  AND dispatch_attempt < max_dispatch_attempts
                  AND (available_at IS NULL OR available_at <= ?)
                """,
                (worker_id, claim_value, lease_value, claim_value, row["id"], claim_value),
            )
            if cursor.rowcount != 1:
                return None
            claimed = connection.execute(
                "SELECT * FROM pending_actions WHERE id = ?", (row["id"],)
            ).fetchone()
            attempt = int(claimed["dispatch_attempt"])
            connection.execute(
                """
                INSERT INTO action_attempts (action_id, attempt, worker_id, status, detail_json, created_at)
                VALUES (?, ?, ?, 'CLAIMED', ?, ?)
                """,
                (
                    row["id"],
                    attempt,
                    worker_id,
                    json.dumps(
                        {
                            "lease_expires_at": lease_value,
                            "recovery_class": claimed["recovery_class"],
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    claim_value,
                ),
            )
        action = self._row_to_pending_action(claimed)
        return DispatchClaim(
            action=action,
            worker_id=worker_id,
            attempt=attempt,
            lease_expires_at=datetime.fromisoformat(lease_value),
        )

    def get_pending_action(self, run_id: UUID) -> PendingAction | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM pending_actions
                WHERE run_id = ? AND status IN ('PENDING', 'APPROVED', 'EXECUTING')
                ORDER BY created_at DESC LIMIT 1
                """,
                (str(run_id),),
            ).fetchone()
        return self._row_to_pending_action(row) if row else None

    def get_action(self, action_id: UUID) -> PendingAction | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM pending_actions WHERE id = ?", (str(action_id),)
            ).fetchone()
        return self._row_to_pending_action(row) if row else None

    def list_pending_actions(self, run_id: UUID) -> list[PendingAction]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM pending_actions WHERE run_id = ? ORDER BY created_at ASC",
                (str(run_id),),
            ).fetchall()
        return [self._row_to_pending_action(row) for row in rows]

    def approve_pending_action(self, run_id: UUID) -> PendingAction | None:
        now = self._now()
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE pending_actions SET status = 'APPROVED', approved_at = ?, updated_at = ?
                WHERE run_id = ? AND status = 'PENDING'
                """,
                (now, now, str(run_id)),
            )
        return self.get_pending_action(run_id)

    def reject_pending_action(self, run_id: UUID) -> None:
        now = self._now()
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE pending_actions SET status = 'REJECTED', updated_at = ?
                WHERE run_id = ? AND status = 'PENDING'
                """,
                (now, str(run_id)),
            )

    def claim_approved_action(
        self,
        run_id: UUID,
        *,
        worker_id: str = "runtime",
        lease_seconds: int = 300,
        now: datetime | None = None,
    ) -> PendingAction | None:
        claim_time = now or datetime.now(UTC)
        now_value = claim_time.isoformat()
        lease_expires_at = (claim_time + timedelta(seconds=lease_seconds)).isoformat()
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM pending_actions
                WHERE run_id = ? AND status = 'APPROVED' ORDER BY approved_at ASC LIMIT 1
                """,
                (str(run_id),),
            ).fetchone()
            if row is None:
                return None
            cursor = connection.execute(
                """
                UPDATE pending_actions
                SET status = 'EXECUTING', claimed_at = ?, worker_id = ?, lease_expires_at = ?,
                    recovered_at = NULL, recovery_reason = NULL, updated_at = ?
                WHERE id = ? AND status = 'APPROVED'
                """,
                (now_value, worker_id, lease_expires_at, now_value, row["id"]),
            )
            if not cursor.rowcount:
                return None
            claimed = connection.execute(
                "SELECT * FROM pending_actions WHERE id = ?", (row["id"],)
            ).fetchone()
        return self._row_to_pending_action(claimed)

    def renew_action_lease(
        self,
        action_id: UUID,
        *,
        worker_id: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> bool:
        """Extend an executing action lease only when it remains owned by this worker."""
        renewal_time = now or datetime.now(UTC)
        lease_expires_at = (renewal_time + timedelta(seconds=lease_seconds)).isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                """
                UPDATE pending_actions SET lease_expires_at = ?, updated_at = ?
                WHERE id = ? AND status = 'EXECUTING' AND worker_id = ?
                """,
                (lease_expires_at, renewal_time.isoformat(), str(action_id), worker_id),
            )
        return bool(cursor.rowcount)

    def recover_stale_executing_actions(
        self,
        *,
        now: datetime | None = None,
        lease_seconds: int = 300,
        reason: str = "WORKER_CRASH_RECOVERY",
    ) -> list[PendingAction]:
        """Fail expired executing actions without making a potentially unsafe tool call twice."""
        recovery_time = now or datetime.now(UTC)
        recovery_value = recovery_time.isoformat()
        legacy_cutoff = (recovery_time - timedelta(seconds=lease_seconds)).isoformat()
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM pending_actions
                WHERE status = 'EXECUTING'
                  AND (
                    (lease_expires_at IS NOT NULL AND lease_expires_at <= ?)
                    OR (lease_expires_at IS NULL AND claimed_at <= ?)
                  )
                ORDER BY claimed_at ASC
                """,
                (recovery_value, legacy_cutoff),
            ).fetchall()
            recovered: list[PendingAction] = []
            for row in rows:
                cursor = connection.execute(
                    """
                    UPDATE pending_actions
                    SET status = 'FAILED', recovered_at = ?, recovery_reason = ?, updated_at = ?
                    WHERE id = ? AND status = 'EXECUTING'
                    """,
                    (recovery_value, reason, recovery_value, row["id"]),
                )
                if cursor.rowcount:
                    updated = connection.execute(
                        "SELECT * FROM pending_actions WHERE id = ?", (row["id"],)
                    ).fetchone()
                    connection.execute(
                        """
                        INSERT INTO action_attempts (
                            action_id, attempt, worker_id, status, detail_json, created_at
                        ) VALUES (?, ?, ?, 'RECOVERED', ?, ?)
                        """,
                        (
                            row["id"],
                            int(updated["dispatch_attempt"]),
                            updated["worker_id"] or "unknown",
                            json.dumps(
                                {
                                    "recovery_class": updated["recovery_class"],
                                    "recovery_reason": reason,
                                },
                                separators=(",", ":"),
                                sort_keys=True,
                            ),
                            recovery_value,
                        ),
                    )
                    recovered.append(self._row_to_pending_action(updated))
        return recovered

    def finish_pending_action(self, action_id: UUID, succeeded: bool) -> None:
        now = self._now()
        with self.connect() as connection:
            row = connection.execute(
                "SELECT worker_id, dispatch_attempt FROM pending_actions WHERE id = ?",
                (str(action_id),),
            ).fetchone()
            cursor = connection.execute(
                """
                UPDATE pending_actions
                SET status = ?, executed_at = ?, updated_at = ?
                WHERE id = ? AND status = 'EXECUTING'
                """,
                ("EXECUTED" if succeeded else "FAILED", now, now, str(action_id)),
            )
            if cursor.rowcount and row is not None:
                connection.execute(
                    """
                    INSERT INTO action_attempts (action_id, attempt, worker_id, status, detail_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(action_id),
                        int(row["dispatch_attempt"]),
                        row["worker_id"] or "runtime",
                        "EXECUTED" if succeeded else "FAILED",
                        json.dumps({"verified": succeeded}, separators=(",", ":"), sort_keys=True),
                        now,
                    ),
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
    def _row_to_checkpoint(row: sqlite3.Row) -> ReActCheckpoint:
        return ReActCheckpoint(
            id=int(row["id"]),
            run_id=UUID(row["run_id"]),
            sequence=int(row["sequence"]),
            phase=row["phase"],
            messages=json.loads(row["messages_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_pending_action(row: sqlite3.Row) -> PendingAction:
        return PendingAction(
            id=UUID(row["id"]),
            run_id=UUID(row["run_id"]),
            tool_name=row["tool_name"],
            arguments=json.loads(row["arguments_json"]),
            risk_level=row["risk_level"],
            checkpoint_id=row["checkpoint_id"],
            status=row["status"],
            approved_at=datetime.fromisoformat(row["approved_at"]) if row["approved_at"] else None,
            claimed_at=datetime.fromisoformat(row["claimed_at"]) if row["claimed_at"] else None,
            worker_id=row["worker_id"],
            lease_expires_at=(
                datetime.fromisoformat(row["lease_expires_at"]) if row["lease_expires_at"] else None
            ),
            recovered_at=(
                datetime.fromisoformat(row["recovered_at"]) if row["recovered_at"] else None
            ),
            recovery_reason=row["recovery_reason"],
            recovery_class=RecoveryClass(row["recovery_class"]),
            recovery_contract_version=int(row["recovery_contract_version"]),
            operation_key=row["operation_key"],
            dispatch_attempt=int(row["dispatch_attempt"]),
            max_dispatch_attempts=int(row["max_dispatch_attempts"]),
            available_at=(
                datetime.fromisoformat(row["available_at"]) if row["available_at"] else None
            ),
            previous_worker_id=row["previous_worker_id"],
            recovery_verification=(
                json.loads(row["recovery_verification_json"])
                if row["recovery_verification_json"]
                else None
            ),
            executed_at=datetime.fromisoformat(row["executed_at"]) if row["executed_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    @staticmethod
    def _row_to_worker(row: sqlite3.Row) -> WorkerRecord:
        return WorkerRecord(
            worker_id=row["worker_id"],
            hostname=row["hostname"],
            process_id=int(row["process_id"]),
            capabilities=tuple(json.loads(row["capabilities_json"])),
            state=row["state"],
            started_at=datetime.fromisoformat(row["started_at"]),
            last_heartbeat_at=datetime.fromisoformat(row["last_heartbeat_at"]),
            stopped_at=datetime.fromisoformat(row["stopped_at"]) if row["stopped_at"] else None,
        )

    @staticmethod
    def _row_to_action_attempt(row: sqlite3.Row) -> ActionAttempt:
        return ActionAttempt(
            id=int(row["id"]),
            action_id=UUID(row["action_id"]),
            attempt=int(row["attempt"]),
            worker_id=row["worker_id"],
            status=row["status"],
            detail=json.loads(row["detail_json"]) if row["detail_json"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def operation_key(
        *,
        tool_name: str,
        arguments: dict[str, Any],
        workspace_id: str,
        recovery_contract_version: int,
    ) -> str:
        canonical = json.dumps(
            {
                "arguments": arguments,
                "recovery_contract_version": recovery_contract_version,
                "tool_name": tool_name,
                "workspace_id": workspace_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

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
