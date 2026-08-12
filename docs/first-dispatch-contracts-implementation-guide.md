# First Commit Guide: Dispatch Contracts, Migrations, and Atomic Claims

**Target commit:** `Add dispatch contracts and atomic claims`
**Prerequisite commit:** `b827289 Add worker lease recovery`
**Execution environment:** Native Ubuntu, Python 3.12+, SQLite, Docker Engine, Ollama.
**Safety posture:** This commit adds metadata and claim primitives only. It does **not** start background workers, enable automatic re-claim, or repeat a tool action after a crash.

## 1. Commit Scope

The first dispatch commit establishes the contracts needed by later worker processes. It must be small enough to validate independently.

| Include in this commit | Explicitly defer |
| --- | --- |
| Runtime-owned recovery classification | Worker process launcher and polling loop |
| Canonical operation key | Automatic re-claim/requeue |
| Migration-safe SQLite columns and append-only attempt history | Any retry of shell, Python, delete, or external-side-effect tools |
| Worker registration and heartbeat persistence primitives | Multi-host deployment or external broker |
| Atomic `APPROVED → EXECUTING` claim primitive | API endpoints and SSE views, except tests if needed |
| Deterministic concurrency, migration, and foreign-worker tests | Real worker orchestration service |

> **Invariant for this commit:** every existing tool has `NEVER_RECLAIM`; `max_dispatch_attempts` is `1`; the existing startup sweep continues to mark expired `EXECUTING` work as `FAILED`.

## 2. Native Ubuntu Commands

Run these commands from the repository root.

```bash
cd ~/local-ai-agent
git pull --ff-only origin main
git switch -c feature/dispatch-contracts

source .venv/bin/activate
export PYTHONPATH=src

python -m pytest -m 'not docker and not ollama'
ruff check src tests
ruff format --check src tests
```

After implementing the changes below, use this validation sequence.

```bash
# Focused contract and repository tests while iterating.
PYTHONPATH=src python -m pytest tests/test_dispatch_contracts.py -q

# Formatting and static validation.
ruff format src tests
ruff check src tests
ruff format --check src tests

# Database bootstrap and complete deterministic suite.
python scripts/bootstrap_workspace.py
PYTHONPATH=src python -m pytest -m 'not docker and not ollama'

# Required release validation on native Ubuntu.
docker build --tag local-ai-agent-sandbox:latest docker/sandbox
RUN_DOCKER_INTEGRATION=1 PYTHONPATH=src python -m pytest
RUN_OLLAMA_EVALUATION=1 PYTHONPATH=src python -m pytest -m ollama

git diff --check
git status --short
```

Publish only after all checks pass:

```bash
git add config/agent.toml .env.example \
  src/local_ai_agent/config.py \
  src/local_ai_agent/schemas/contracts.py \
  src/local_ai_agent/tools/registry.py \
  src/local_ai_agent/db/schema.py \
  src/local_ai_agent/db/repository.py \
  tests/test_dispatch_contracts.py \
  docs/multi-process-worker-dispatch-design.md

git commit -m 'Add dispatch contracts and atomic claims'
git push -u origin feature/dispatch-contracts
```

Merge to `main` only after review and after the full native-Ubuntu validation is recorded.

## 3. Contract Changes

### 3.1 Add a Runtime-Owned Recovery Enum

**File:** `src/local_ai_agent/schemas/contracts.py`

Add this near the existing risk/state enums.

```python
class RecoveryClass(str, Enum):
    """Runtime-owned rule for handling a tool action left uncertain by a worker crash."""

    NEVER_RECLAIM = "NEVER_RECLAIM"
    VERIFY_BEFORE_RECLAIM = "VERIFY_BEFORE_RECLAIM"
    IDEMPOTENT_RECLAIM = "IDEMPOTENT_RECLAIM"
```

This enum must not be accepted from model output, user request payloads, or tool arguments. It is assigned only by registered Python tool definitions.

### 3.2 Extend `ToolDefinition`

**File:** `src/local_ai_agent/tools/registry.py`

Import the enum and add two runtime-only fields. Neither field belongs in `ollama_tools()`.

```python
from local_ai_agent.schemas.contracts import (
    RecoveryClass,
    RiskLevel,
    ToolResult,
    VerificationResult,
)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    risk: RiskLevel
    handler: ToolHandler
    verification: VerificationHandler | None = None
    arguments_validator: ArgumentsValidator | None = None
    recovery_class: RecoveryClass = RecoveryClass.NEVER_RECLAIM
    recovery_contract_version: int = 1
```

The default matters: every existing tool remains non-reclaimable until a future dedicated proof changes its definition. Do not classify write operations, shell, Python, delete, or future external/network tools as reclaimable in this commit.

### 3.3 Add Dispatch Configuration

**File:** `config/agent.toml`

Add this block after `[worker]`.

```toml
[dispatch]
enabled = false
max_workers = 1
claim_poll_seconds = 1.0
claim_jitter_seconds = 0.25
worker_stale_seconds = 120
recovery_sweep_seconds = 30
max_dispatch_attempts = 1
heartbeat_event_sample_seconds = 30
```

**File:** `.env.example`

Add optional overrides.

```dotenv
DISPATCH_ENABLED=false
DISPATCH_MAX_WORKERS=1
DISPATCH_CLAIM_POLL_SECONDS=1.0
DISPATCH_CLAIM_JITTER_SECONDS=0.25
DISPATCH_WORKER_STALE_SECONDS=120
DISPATCH_RECOVERY_SWEEP_SECONDS=30
DISPATCH_MAX_ATTEMPTS=1
DISPATCH_HEARTBEAT_EVENT_SAMPLE_SECONDS=30
```

**File:** `src/local_ai_agent/config.py`

Add a `dispatch = raw["dispatch"]` section and these fields to `Settings`.

```python
    dispatch_enabled: bool
    dispatch_max_workers: int
    dispatch_claim_poll_seconds: float
    dispatch_claim_jitter_seconds: float
    dispatch_worker_stale_seconds: int
    dispatch_recovery_sweep_seconds: int
    dispatch_max_attempts: int
    dispatch_heartbeat_event_sample_seconds: int
```

Load them through existing environment helpers:

```python
        dispatch_enabled=_env_bool("DISPATCH_ENABLED", dispatch["enabled"]),
        dispatch_max_workers=_env_int("DISPATCH_MAX_WORKERS", dispatch["max_workers"]),
        dispatch_claim_poll_seconds=_env_float(
            "DISPATCH_CLAIM_POLL_SECONDS", dispatch["claim_poll_seconds"]
        ),
        dispatch_claim_jitter_seconds=_env_float(
            "DISPATCH_CLAIM_JITTER_SECONDS", dispatch["claim_jitter_seconds"]
        ),
        dispatch_worker_stale_seconds=_env_int(
            "DISPATCH_WORKER_STALE_SECONDS", dispatch["worker_stale_seconds"]
        ),
        dispatch_recovery_sweep_seconds=_env_int(
            "DISPATCH_RECOVERY_SWEEP_SECONDS", dispatch["recovery_sweep_seconds"]
        ),
        dispatch_max_attempts=_env_int("DISPATCH_MAX_ATTEMPTS", dispatch["max_dispatch_attempts"]),
        dispatch_heartbeat_event_sample_seconds=_env_int(
            "DISPATCH_HEARTBEAT_EVENT_SAMPLE_SECONDS",
            dispatch["heartbeat_event_sample_seconds"],
        ),
```

Build `settings = Settings(...)`, validate it, then return it. Add a helper:

```python
def _validate_dispatch_settings(settings: Settings) -> None:
    if settings.worker_heartbeat_seconds <= 0:
        raise ValueError("WORKER_HEARTBEAT_SECONDS must be positive.")
    if settings.worker_heartbeat_seconds >= settings.worker_lease_seconds:
        raise ValueError("WORKER_HEARTBEAT_SECONDS must be less than WORKER_LEASE_SECONDS.")
    if settings.dispatch_max_workers < 1:
        raise ValueError("DISPATCH_MAX_WORKERS must be at least one.")
    if settings.dispatch_claim_poll_seconds <= 0:
        raise ValueError("DISPATCH_CLAIM_POLL_SECONDS must be positive.")
    if settings.dispatch_claim_jitter_seconds < 0:
        raise ValueError("DISPATCH_CLAIM_JITTER_SECONDS cannot be negative.")
    if settings.dispatch_worker_stale_seconds < settings.worker_heartbeat_seconds:
        raise ValueError("DISPATCH_WORKER_STALE_SECONDS must allow at least one heartbeat.")
    if settings.dispatch_recovery_sweep_seconds <= 0:
        raise ValueError("DISPATCH_RECOVERY_SWEEP_SECONDS must be positive.")
    if settings.dispatch_max_attempts != 1:
        raise ValueError(
            "DISPATCH_MAX_ATTEMPTS must remain 1 until a reviewed idempotency pilot is enabled."
        )
```

The `max_dispatch_attempts == 1` constraint is deliberate. This commit records future retry metadata without changing recovery behavior.

## 4. Migration-Safe SQLite Changes

### 4.1 New Tables and Columns

**File:** `src/local_ai_agent/db/schema.py`

Add the following to `SCHEMA_SQL` after `pending_actions`.

```sql
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
```

Add the following columns to the *new table* definition for `pending_actions`.

```sql
    recovery_class TEXT NOT NULL DEFAULT 'NEVER_RECLAIM',
    recovery_contract_version INTEGER NOT NULL DEFAULT 1,
    operation_key TEXT,
    dispatch_attempt INTEGER NOT NULL DEFAULT 0,
    max_dispatch_attempts INTEGER NOT NULL DEFAULT 1,
    available_at TEXT,
    previous_worker_id TEXT,
    recovery_verification_json TEXT,
```

Create lookup indexes only after migration has added all relevant columns.

```sql
CREATE INDEX IF NOT EXISTS idx_pending_actions_dispatchable
    ON pending_actions(status, available_at, dispatch_attempt, max_dispatch_attempts);

CREATE INDEX IF NOT EXISTS idx_pending_actions_operation_key
    ON pending_actions(operation_key);
```

### 4.2 Generic Column Migration Helper

The current repository already has `_ensure_pending_action_recovery_columns`. Replace it with a broader migration helper. Do not create an index that references a new column until *after* this helper runs.

```python
def _ensure_pending_action_dispatch_columns(connection: sqlite3.Connection) -> None:
    existing = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(pending_actions)").fetchall()
    }
    required = (
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
    )
    for name, definition in required:
        if name not in existing:
            connection.execute(f"ALTER TABLE pending_actions ADD COLUMN {name} {definition}")
```

In `initialize_database`, preserve this order:

```python
with connection:
    connection.executescript(SCHEMA_SQL)
    _ensure_pending_action_dispatch_columns(connection)
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
```

### 4.3 Persisted Dataclasses

**File:** `src/local_ai_agent/db/repository.py`

Extend `PendingAction`.

```python
from local_ai_agent.schemas.contracts import RecoveryClass

@dataclass(frozen=True, slots=True)
class PendingAction:
    # Existing fields...
    recovery_class: RecoveryClass
    recovery_contract_version: int
    operation_key: str | None
    dispatch_attempt: int
    max_dispatch_attempts: int
    available_at: datetime | None
    previous_worker_id: str | None
    recovery_verification: dict[str, Any] | None
```

Add dataclasses for persistence boundaries.

```python
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
class DispatchClaim:
    action: PendingAction
    worker_id: str
    attempt: int
    lease_expires_at: datetime
```

Use `json.dumps(..., sort_keys=True, separators=(",", ":"))` for persisted structured values. The operation key is deterministic only when arguments are canonicalized.

## 5. Canonical Operation Key

**File:** `src/local_ai_agent/db/repository.py` or a new small `runtime/dispatch.py` helper.

```python
import hashlib
import json


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
```

This key is an audit/correlation value. It must not permit a retry by itself.

## 6. Atomic Dispatch Primitives

### 6.1 Worker Registration Primitives

Add repository methods. These are data-layer operations; they must not call a model or a tool.

```python
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
```

### 6.2 Atomic Eligible-Action Claim

The critical rule is that SQLite decides the winner with one conditional update. Never implement a `SELECT` followed by an unconditional `UPDATE`, and never keep a transaction open while executing the tool.

```python
def claim_next_dispatchable_action(
    self,
    *,
    worker_id: str,
    capabilities: tuple[str, ...],
    lease_seconds: int,
    now: datetime | None = None,
) -> DispatchClaim | None:
    claim_time = now or datetime.now(UTC)
    claim_value = claim_time.isoformat()
    lease_value = (claim_time + timedelta(seconds=lease_seconds)).isoformat()
    tool_placeholders = ", ".join("?" for _ in capabilities)
    if not capabilities:
        return None

    with self.connect() as connection:
        # The connection is configured with isolation_level='IMMEDIATE'. Start
        # the writer transaction before selecting so another process cannot
        # interleave a competing eligible-action claim in this decision window.
        connection.execute("BEGIN IMMEDIATE")

        worker = connection.execute(
            "SELECT state FROM workers WHERE worker_id = ?", (worker_id,)
        ).fetchone()
        if worker is None or worker["state"] != "ACTIVE":
            return None

        row = connection.execute(
            f"""
            SELECT * FROM pending_actions
            WHERE status = 'APPROVED'
              AND dispatch_attempt < max_dispatch_attempts
              AND (available_at IS NULL OR available_at <= ?)
              AND tool_name IN ({tool_placeholders})
            ORDER BY approved_at ASC, created_at ASC
            LIMIT 1
            """,
            (claim_value, *capabilities),
        ).fetchone()
        if row is None:
            return None

        cursor = connection.execute(
            """
            UPDATE pending_actions
            SET status = 'EXECUTING',
                previous_worker_id = worker_id,
                worker_id = ?,
                claimed_at = ?,
                lease_expires_at = ?,
                dispatch_attempt = dispatch_attempt + 1,
                recovered_at = NULL,
                recovery_reason = NULL,
                updated_at = ?
            WHERE id = ?
              AND status = 'APPROVED'
              AND dispatch_attempt < max_dispatch_attempts
              AND (available_at IS NULL OR available_at <= ?)
            """,
            (
                worker_id,
                claim_value,
                lease_value,
                claim_value,
                row["id"],
                claim_value,
            ),
        )
        if cursor.rowcount != 1:
            return None

        claimed_row = connection.execute(
            "SELECT * FROM pending_actions WHERE id = ?", (row["id"],)
        ).fetchone()
        attempt = int(claimed_row["dispatch_attempt"])
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
                        "recovery_class": claimed_row["recovery_class"],
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                claim_value,
            ),
        )

    action = self._row_to_pending_action(claimed_row)
    return DispatchClaim(
        action=action,
        worker_id=worker_id,
        attempt=attempt,
        lease_expires_at=datetime.fromisoformat(lease_value),
    )
```

### 6.3 Important Corrections to the Skeleton

Before committing, apply these safeguards.

| Concern | Required correction |
| --- | --- |
| Worker capability representation | The initial worker should register actual tool names, for example `("filesystem.list_directory", "filesystem.read_file")`; do not use model-provided capability strings. |
| `previous_worker_id` | Set it to the prior owner only when an action is reclaimed in a future phase. For the first claim it should remain `NULL`; add a separate `claimed_from_worker_id` expression only after re-claim is enabled. |
| `BEGIN IMMEDIATE` nesting | If `connect()` remains `isolation_level="IMMEDIATE"`, use explicit `BEGIN IMMEDIATE` only after changing `connect()` to `isolation_level=None`, or rely on the first conditional `UPDATE`. Choose one transaction strategy and test it. Recommended: retain current connection settings and remove explicit `BEGIN IMMEDIATE`; the conditional update remains the ownership guard. |
| Empty capability list | Return `None` before building `IN (...)`. |
| Recovery class | Claiming does not authorize re-claim. The class is recorded for audit only in this commit. |
| Tool execution | The repository method returns before any `ToolRouter` call. The worker runtime added in a later commit owns execution. |

For the existing repository configuration, use the safer adapted implementation below. It preserves `isolation_level="IMMEDIATE"` and relies on the conditional update guard.

```python
def claim_next_dispatchable_action(... ) -> DispatchClaim | None:
    # Calculate timestamps and reject an empty capability list as above.
    with self.connect() as connection:
        row = connection.execute(...).fetchone()
        if row is None:
            return None
        cursor = connection.execute(
            """
            UPDATE pending_actions
            SET status = 'EXECUTING', worker_id = ?, claimed_at = ?,
                lease_expires_at = ?, dispatch_attempt = dispatch_attempt + 1,
                recovered_at = NULL, recovery_reason = NULL, updated_at = ?
            WHERE id = ? AND status = 'APPROVED'
              AND dispatch_attempt < max_dispatch_attempts
              AND (available_at IS NULL OR available_at <= ?)
            """,
            (...),
        )
        if cursor.rowcount != 1:
            return None
        # Read the claimed row and append CLAIMED attempt in this same transaction.
```

Even if two workers select the same row before either writes, SQLite permits only one conditional update. The loser observes `rowcount == 0` and does not execute the tool.

## 7. Action Creation Integration

This first commit should calculate recovery metadata at the Python runtime boundary where the tool definition is already known. Do not let `AuthorizationRequest` accept a raw recovery class from an API payload.

Extend `AuthorizationRequest` internally:

```python
@dataclass(frozen=True, slots=True)
class AuthorizationRequest:
    run_id: UUID
    tool_name: str
    arguments: dict[str, Any]
    risk: str
    checkpoint_id: int | None = None
    recovery_class: RecoveryClass = RecoveryClass.NEVER_RECLAIM
    recovery_contract_version: int = 1
    operation_key: str | None = None
    max_dispatch_attempts: int = 1
```

In `RunToolExecutor.execute`, obtain the tool definition from the runtime registry, calculate `operation_key(...)` from validated arguments plus the run workspace, and pass the definition-owned recovery metadata into `lifecycle.require_authorization(...)`.

Then extend `RunRepository.create_pending_action(...)` to persist that immutable metadata. Existing callers should default to `NEVER_RECLAIM` and `max_dispatch_attempts=1` for migration safety.

## 8. Focused Test File

Create **`tests/test_dispatch_contracts.py`**. The initial tests should use a repository with a temporary SQLite path and deterministic timestamps. Do not start actual processes in this commit.

```python
def test_all_current_tool_definitions_default_to_never_reclaim() -> None:
    registry = ToolRegistry()
    # Register one representative definition from each current tool builder.
    # Assert every definition.recovery_class is RecoveryClass.NEVER_RECLAIM.


def test_legacy_database_gains_dispatch_columns(tmp_path: Path) -> None:
    # Create a pre-dispatch pending_actions table fixture.
    # repository.initialize()
    # Assert PRAGMA table_info contains every new column and old action data remains readable.


def test_register_worker_and_heartbeat_are_durable(tmp_path: Path) -> None:
    # Register then heartbeat a worker with fixed timestamps.
    # Assert state ACTIVE and last_heartbeat_at moves forward.


def test_two_workers_can_claim_one_approved_action_only_once(tmp_path: Path) -> None:
    # Create an APPROVED action with max_dispatch_attempts=1.
    # Claim once as worker-a, then attempt worker-b claim.
    # Assert worker-a has EXECUTING action, worker-b gets None,
    # dispatch_attempt == 1, and exactly one CLAIMED action_attempt exists.


def test_claim_rejects_draining_or_unknown_worker(tmp_path: Path) -> None:
    # Assert no state transition occurs.


def test_claim_respects_available_at_and_attempt_cap(tmp_path: Path) -> None:
    # Assert future actions and exhausted actions remain APPROVED.


def test_operation_key_is_stable_for_key_order_and_changes_for_contract_version() -> None:
    # {"a": 1, "b": 2} and reversed key order match;
    # a version or workspace change produces a different hash.
```

Add a concurrency test that uses two Python threads plus a `threading.Barrier`, each with its own repository connection:

```python
def test_concurrent_claim_has_one_winner(tmp_path: Path) -> None:
    barrier = threading.Barrier(2)
    outcomes: list[DispatchClaim | None] = []

    def claim(worker_id: str) -> None:
        barrier.wait()
        repository = RunRepository(database_path)
        outcomes.append(
            repository.claim_next_dispatchable_action(
                worker_id=worker_id,
                capabilities=("filesystem.list_directory",),
                lease_seconds=30,
            )
        )

    # Start worker-a and worker-b, join both, assert exactly one non-None result.
```

Do not assert a deterministic winner. SQLite scheduling determines which worker obtains the claim.

## 9. First-Commit Acceptance Checklist

Before publishing, verify all of the following.

- [ ] Existing pending-action rows migrate without data loss.
- [ ] Every existing tool persists `NEVER_RECLAIM` and a maximum dispatch attempt of one.
- [ ] Operation keys are deterministic and use only runtime-owned inputs.
- [ ] A worker must be active before it can claim work.
- [ ] Two concurrent claim attempts yield exactly one `EXECUTING` row and one append-only `CLAIMED` attempt record.
- [ ] No code path starts a worker loop, requeues an action, or repeats a tool action.
- [ ] The full existing suite stays green, including Docker isolation and the real Ollama corpus on native Ubuntu.
- [ ] The commit contains only contracts, migrations, repository primitives, and tests—not worker orchestration.

## 10. What Comes Immediately After This Commit

The second commit should add a bounded local worker service that registers, heartbeats, claims one action, calls the existing continuation path, and drains safely. It must use the atomic repository primitive from this document. It still must not enable automatic re-claim; the recovery enum remains audit-only until a separate idempotency pilot is reviewed.
