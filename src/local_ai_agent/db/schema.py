"""SQLite schema bootstrap for the local agent's authoritative state and audit records."""

from __future__ import annotations

import sqlite3

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    objective TEXT NOT NULL,
    state TEXT NOT NULL,
    plan_json TEXT,
    budget_json TEXT NOT NULL,
    resume_token TEXT NOT NULL UNIQUE,
    prompt_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_run_id ON tool_calls(run_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_loop ON tool_calls(run_id, tool_name, args_hash);

CREATE TABLE IF NOT EXISTS tool_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_call_id INTEGER NOT NULL REFERENCES tool_calls(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    success INTEGER NOT NULL,
    data_json TEXT,
    error_code TEXT,
    error_message TEXT,
    retryable INTEGER NOT NULL DEFAULT 0,
    verified INTEGER NOT NULL DEFAULT 0,
    verification_json TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    duration_ms INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence TEXT NOT NULL,
    source_run_id TEXT REFERENCES agent_runs(id) ON DELETE SET NULL,
    expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(category, memory_key)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    memory_key,
    value
);

CREATE TABLE IF NOT EXISTS agent_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    state TEXT NOT NULL,
    message TEXT NOT NULL,
    data_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_events_run_id ON agent_events(run_id, id);

CREATE TABLE IF NOT EXISTS file_backups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    original_path TEXT NOT NULL,
    backup_path TEXT NOT NULL,
    original_hash TEXT NOT NULL,
    restored_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_controls (
    run_id TEXT PRIMARY KEY REFERENCES agent_runs(id) ON DELETE CASCADE,
    cancel_requested INTEGER NOT NULL DEFAULT 0,
    pending_authorization_json TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_locks (
    workspace_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    acquired_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS react_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    phase TEXT NOT NULL,
    messages_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_react_checkpoints_run ON react_checkpoints(run_id, sequence DESC);

CREATE TABLE IF NOT EXISTS pending_actions (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    tool_name TEXT NOT NULL,
    arguments_json TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    checkpoint_id INTEGER REFERENCES react_checkpoints(id) ON DELETE SET NULL,
    status TEXT NOT NULL,
    approved_at TEXT,
    claimed_at TEXT,
    worker_id TEXT,
    lease_expires_at TEXT,
    recovered_at TEXT,
    recovery_reason TEXT,
    recovery_class TEXT NOT NULL DEFAULT 'NEVER_RECLAIM',
    recovery_contract_version INTEGER NOT NULL DEFAULT 1,
    operation_key TEXT,
    dispatch_attempt INTEGER NOT NULL DEFAULT 0,
    max_dispatch_attempts INTEGER NOT NULL DEFAULT 1,
    available_at TEXT,
    previous_worker_id TEXT,
    recovery_verification_json TEXT,
    executed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pending_actions_run_status ON pending_actions(run_id, status);

CREATE TABLE IF NOT EXISTS workers (
    worker_id TEXT PRIMARY KEY,
    hostname TEXT NOT NULL,
    process_id INTEGER NOT NULL,
    capabilities_json TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('STARTING', 'ACTIVE', 'DRAINING', 'STOPPED')),
    started_at TEXT NOT NULL,
    last_heartbeat_at TEXT NOT NULL,
    stopped_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workers_active_heartbeat
    ON workers(state, last_heartbeat_at);

CREATE TABLE IF NOT EXISTS action_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_id TEXT NOT NULL REFERENCES pending_actions(id) ON DELETE CASCADE,
    attempt INTEGER NOT NULL,
    worker_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('CLAIMED', 'HEARTBEAT', 'EXECUTED', 'FAILED', 'RECOVERED')),
    detail_json TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_action_attempts_action_attempt
    ON action_attempts(action_id, attempt, created_at);
"""


def _ensure_pending_action_recovery_columns(connection: sqlite3.Connection) -> None:
    """Add worker lease fields when opening a database created by an earlier runtime version."""
    existing_columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(pending_actions)").fetchall()
    }
    for name, definition in (
        ("worker_id", "TEXT"),
        ("lease_expires_at", "TEXT"),
        ("recovered_at", "TEXT"),
        ("recovery_reason", "TEXT"),
        ("recovery_class", "TEXT NOT NULL DEFAULT 'NEVER_RECLAIM'"),
        ("recovery_contract_version", "INTEGER NOT NULL DEFAULT 1"),
        ("operation_key", "TEXT"),
        ("dispatch_attempt", "INTEGER NOT NULL DEFAULT 0"),
        ("max_dispatch_attempts", "INTEGER NOT NULL DEFAULT 1"),
        ("available_at", "TEXT"),
        ("previous_worker_id", "TEXT"),
        ("recovery_verification_json", "TEXT"),
    ):
        if name not in existing_columns:
            connection.execute(f"ALTER TABLE pending_actions ADD COLUMN {name} {definition}")


def initialize_database(connection: sqlite3.Connection) -> None:
    """Create the schema and rebuild legacy external-content FTS indexes safely."""
    with connection:
        connection.executescript(SCHEMA_SQL)
        _ensure_pending_action_recovery_columns(connection)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_pending_actions_stale_lease "
            "ON pending_actions(status, lease_expires_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_pending_actions_dispatchable "
            "ON pending_actions(status, available_at, dispatch_attempt, max_dispatch_attempts)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_pending_actions_operation_key "
            "ON pending_actions(operation_key)"
        )
        definition = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'memory_fts'"
        ).fetchone()[0]
        if "content='memories'" in definition.replace(" ", ""):
            connection.execute("DROP TABLE memory_fts")
            connection.execute("CREATE VIRTUAL TABLE memory_fts USING fts5(memory_key, value)")
            connection.execute(
                "INSERT INTO memory_fts (rowid, memory_key, value) SELECT id, memory_key, value FROM memories"
            )
