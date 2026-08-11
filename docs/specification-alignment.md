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
| PlanTracker with dependency-safe status progression | `runtime/plan_tracker.py` | Implemented |
| Permission gating, budget enforcement, retries, loop blocking, and operation-aware verification | `runtime/permission_gate.py`, `budget_manager.py`, `retry_engine.py`, `loop_detector.py`, `verification_engine.py`, and `tool_router.py` | Implemented and unit-tested |
| SQLite source of truth for runs, tools, results, events, memory, backups | `db/schema.py` and `db/repository.py` | Initialized |
| SQLite FTS5 initial semantic-search path | `memory_fts` virtual table | Initialized |
| Symlink-safe workspace enforcement | `security/paths.py` | Implemented foundation |
| Docker as tool sandbox boundary | `runtime/docker_sandbox.py`, `docker/sandbox/Dockerfile`, and centrally validated sandbox settings | Implemented and integration-tested |
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
| ReAct orchestration loop | Must invoke the now-established PlanTracker and ToolRouter around model-native tool requests. |
| Filesystem and shell tool handlers | Must register concrete validators, handlers, and operation-specific verification functions with the implemented ToolRouter; write-capable operations also need backup semantics. |
| Tool-pipeline integration around Docker | The executor is implemented, but future tool handlers still need allowlisting, authorization, backups, verification, audit persistence, and transactional rollback before delegation. |
| Persistent event audit and session lock enforcement | Require full lifecycle transitions and transaction management. |
| Memory indexing/retrieval and context compression | Need deliberate chunking, confidence, staleness, and token-budget semantics. |
| Authorization pause/resume | Depends on persisted runtime continuation state and tool transaction management. |
| System prompt | Must be loaded only after the runtime’s tool and output contracts are complete. |

## Immediate next development milestone

The next milestone should implement exactly the specification's Phase 1: read-only `filesystem.list_directory` and `filesystem.read_file` tool handlers, a minimal runtime loop, tool-result persistence, and independent post-tool verification. The Docker executor is now available as the execution boundary for future approved actions, but high-risk operations should remain disabled until their full policy and transaction stages are complete.
