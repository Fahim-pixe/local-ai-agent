# Specification Alignment

This repository maps the supplied **Local AI Agent Architecture Specification v2.0** into a contract-first local runtime. It now includes the initial autonomous read path and the Priority 3 persistence, transaction, authorization, and sandbox boundaries required before broader tool expansion.

## Core requirements mapped to the repository

| Specification requirement | Repository initialization | Status |
| --- | --- | --- |
| Python 3.12+, FastAPI, SQLite, Ollama, Qwen3 | `pyproject.toml`, `Dockerfile`, `src/local_ai_agent/` | Ready |
| Local Qwen decision layer; Python enforcement | Package layout separates `runtime`, `security`, `tools`, `db`, and API | Ready |
| Native Ollama tool schemas | `runtime/ollama_client.py`, `tools/registry.py`, `runtime/react_loop.py`, and the Phase 6 corpus | Implemented with production prompt wiring and an opt-in real-model validation path |
| Pydantic `ToolResult` and verification distinction | `schemas/contracts.py` | Implemented |
| State machine with valid transitions | `runtime/state_machine.py` | Implemented |
| Output validator ahead of runtime actions | `runtime/output_validator.py` | Implemented |
| PlanTracker with dependency-safe status progression | `runtime/plan_tracker.py` | Implemented |
| Minimal native-Ollama ReAct loop with low-risk filesystem reads | `runtime/react_loop.py`, `runtime/minimal_runtime.py`, and `tools/filesystem.py` | Implemented and unit-tested |
| Permission gating, budget enforcement, retries, loop blocking, and operation-aware verification | `runtime/permission_gate.py`, `budget_manager.py`, `retry_engine.py`, `loop_detector.py`, `verification_engine.py`, and `tool_router.py` | Implemented and unit-tested |
| SQLite source of truth for runs, tools, results, events, memory, backups | `db/schema.py` and `db/repository.py` | Implemented for runs, events, tool audit, backups, cancellation, authorization, and session locks |
| Durable lifecycle, cancellation, authorization, per-session lock, and resume token | `runtime/lifecycle.py`, `runtime/run_executor.py`, SQLite `run_controls` / `session_locks`, and `POST /runs/{id}/resume` | Implemented and API-tested; the resume route requires constant-time validation of the persisted run token before durable continuation |
| Transactional writes and deletes | `runtime/transaction_manager.py` and `tools/mutation.py` | Implemented with snapshots, atomic writes, independent verification, and rollback tests |
| Command policy and output secret scrubbing | `security/command_policy.py` and `security/output_scrubber.py` | Implemented and tested |
| Sandboxed high-risk execution tools | `shell.execute` and `python.execute` through `tools/mutation.py` and `runtime/docker_sandbox.py` | Implemented; explicit authorization and command policy required |
| Durable long-term and semantic memory | `memory/repository.py`, `memories`, and rebuilt-safe `memory_fts` | Implemented with FTS5 retrieval, confidence, expiry, and stale-memory treatment |
| Priority-tiered context assembly | `memory/context_manager.py` | Implemented with P0/P1/P2/P3 policy, explicit untrusted-memory labels, and deterministic line-range truncation |
| Verified memory persistence tool | `tools/memory.py` registered through the secure run runtime | Implemented with retrieval verification and output scrubbing |
| SQLite FTS5 initial semantic-search path | `memory_fts` virtual table | Implemented and migration-tested |
| Symlink-safe workspace enforcement | `security/paths.py` | Implemented foundation |
| Docker as tool sandbox boundary | `runtime/docker_sandbox.py`, `docker/sandbox/Dockerfile`, and centrally validated sandbox settings | Implemented and integration-tested |
| Durable ReAct checkpoints and pending actions | `react_checkpoints`, `pending_actions`, `runtime/checkpointing.py`, and `runtime/continuation.py` | Implemented with append-only messages, approval, single claim, execution result checkpointing, and exact replay |
| API lifecycle + SSE | `api/app.py` | Implemented with durable creation, cancellation, authorization, replies, listing, historical events, live SSE notifications, approved-action continuation, and token-validated `POST /runs/{id}/resume` |
| Versioned production prompt and SHA-256 provenance | `config/agent.toml`, `config/system_prompt.md`, `runtime/production_prompt.py`, `Settings`, and `agent_runs.prompt_hash` | Implemented; TOML-backed name and mission render into the UTF-8 prompt before its exact message bytes are hashed and persisted for every API-created run |
| Opt-in end-to-end local coding evaluation | `evaluation/coding_tasks.py`, `tests/test_coding_evaluation.py`, and `make test-ollama` | Implemented for list, read, and write scenarios; it grades durable verified tool evidence and deterministic workspace outcomes when `RUN_OLLAMA_EVALUATION=1` |
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
| Run-worker orchestration and crash recovery | Continuation is durable and single-claim, but background worker ownership, abandoned-claim recovery, and observability classification remain explicit follow-on work. |
| Broader local-model benchmark baseline | The Phase 6 corpus is ready, but it requires target hardware with Ollama and Qwen3 available to collect an operational baseline and broaden scenario coverage. |
| Additional secure operations | Any new tool still requires operation-specific schema, path/command policy, risk gate, transaction semantics where applicable, verification, and audit coverage. |

## Immediate next development milestone

The next milestone should add explicit worker ownership, abandoned-action recovery, and continuation observability. The opt-in Phase 6 local-model corpus should be run and baselined on target hardware before expanding the tool surface. Priority 4 uses local FTS5 as the semantic retrieval baseline; embeddings should remain a separately evaluated enhancement.
