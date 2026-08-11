# Specification Alignment

This initialization converts the supplied **Local AI Agent Architecture Specification v2.0** into a repository foundation. It intentionally creates the contracts, boundaries, and infrastructure that must exist before implementation of autonomous tool execution.

## Core requirements mapped to the repository

| Specification requirement | Repository initialization | Status |
| --- | --- | --- |
| Python 3.12+, FastAPI, SQLite, Ollama, Qwen3 | `pyproject.toml`, `Dockerfile`, `src/local_ai_agent/` | Ready |
| Local Qwen decision layer; Python enforcement | Package layout separates `runtime`, `security`, `tools`, `db`, and API | Ready |
| Native Ollama tool schemas | `runtime/ollama_client.py` and `tools/registry.py` | Ready boundary |
| Pydantic `ToolResult` and verification distinction | `schemas/contracts.py` | Implemented |
| State machine with valid transitions | `runtime/state_machine.py` | Implemented |
| Output validator ahead of runtime actions | `runtime/output_validator.py` | Implemented |
| SQLite source of truth for runs, tools, results, events, memory, backups | `db/schema.py` and `db/repository.py` | Initialized |
| SQLite FTS5 initial semantic-search path | `memory_fts` virtual table | Initialized |
| Symlink-safe workspace enforcement | `security/paths.py` | Implemented foundation |
| Docker as tool sandbox boundary | `docker/sandbox/Dockerfile` and sandbox settings | Ready boundary |
| API lifecycle + SSE | `api/app.py` | Scaffolded |
| Environment-controlled paths, models, budgets, limits | `config/agent.toml` and `.env.example` | Ready |
| Contract, boundary, and persistence tests | `tests/test_foundation.py` | Ready |

## Workspace model

The committed empty directories preserve the intended runtime layout while the `.gitignore` policy prevents generated database records, user project content, backups, logs, and run traces from entering source control.

| Directory | Owner and role |
| --- | --- |
| `workspace/project/` | Agent-accessible project files; this is the only intended tool filesystem root. |
| `workspace/backups/` | Runtime-managed pre-change snapshots. |
| `workspace/.agent/` | Runtime internals, including `agent.db`; inaccessible to future filesystem tools. |
| `workspace/runs/` | Run-specific artifacts. |
| `workspace/logs/` | Local logs and audit exports. |
| `workspace/system/` | Read-only system/reference assets. |

## Deliberately deferred work

The initial commit does not pretend to implement the full autonomous runtime. The following remain separate implementation phases because they carry consequential security and correctness implications:

| Deferred component | Reason for deferral |
| --- | --- |
| ReAct orchestration loop | Must be built atop the now-established schemas, state machine, and output validator. |
| Filesystem and shell tool handlers | Require full policy pipeline, risk gating, backup semantics, and operation-aware verification. |
| Docker executor | Requires a runtime policy adapter that launches the sandbox image with immutable command construction. |
| Persistent event audit and session lock enforcement | Require full lifecycle transitions and transaction management. |
| Memory indexing/retrieval and context compression | Need deliberate chunking, confidence, staleness, and token-budget semantics. |
| Authorization pause/resume | Depends on persisted runtime continuation state and tool transaction management. |
| System prompt | Must be loaded only after the runtime’s tool and output contracts are complete. |

## Immediate next development milestone

The next milestone should implement exactly the specification's Phase 1: read-only `filesystem.list_directory` and `filesystem.read_file` tool handlers, a minimal runtime loop, tool-result persistence, and independent post-tool verification. High-risk operations should not be enabled before the policy and sandbox stages are complete.
