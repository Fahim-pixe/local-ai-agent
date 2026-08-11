# Local AI Agent

A **contract-first, local-first** autonomous agent runtime. The local model decides what to do; the Python runtime validates whether it is allowed; tools run only through runtime policy; SQLite records the authoritative state.

> **Current status:** The contract-first foundation, minimal ReAct loop, durable run lifecycle, transactional workspace mutation, authorization controls, and sandboxed execution boundary are implemented. Context/memory retrieval, resume replay, and broader production evaluation remain subsequent milestones.

## Architecture

| Layer | Responsibility |
| --- | --- |
| Local model | Qwen proposes plans and native Ollama tool calls. It never directly executes tools. |
| Python runtime | Enforces state transitions, schema validation, budgets, policy, authorization, verification, retry, and audit. |
| Tool registry | Defines runtime-owned tool schemas, risk levels, handlers, and verification functions. |
| FastAPI control plane | Exposes run lifecycle routes and SSE-ready event streaming. |
| SQLite | Stores runs, tool calls/results, events, memories, and file backups. |
| Workspace | Restricts agent-accessible files to `workspace/project/`; runtime-only data stays under `.agent/` and other protected directories. |

## Implemented foundation

| Specification priority | Initialized component |
| --- | --- |
| Typed tool contract | `schemas/contracts.py` provides Pydantic `ToolResult`, `VerificationResult`, plans, budgets, runs, and events. |
| Runtime state authority | `runtime/state_machine.py` enforces documented valid transitions. |
| Model-output validation | `runtime/output_validator.py` rejects malformed plans and ambiguous tool-plus-answer turns. |
| Plan and execution controls | `PlanTracker` owns dependency-safe plan steps; `ToolRouter` applies runtime argument validation, authorization, budgets, loop detection, handler execution, operation-aware verification, and retry decisions in fixed order. |
| Minimal ReAct capability | `ReActLoop` uses Ollama native tool calls and returns only verified results from `filesystem.list_directory` and `filesystem.read_file` inside `workspace/project/`. |
| Local model boundary | `runtime/ollama_client.py` uses native Ollama `tools` payloads and explicit local failure types. |
| SQLite source of truth | `db/schema.py` creates runs, tool call/result, memory, event, backup, and FTS5 tables. |
| Workspace isolation | `security/paths.py` resolves symlinks before containment checks. |
| Durable lifecycle and concurrency | `RunLifecycleService` persists state transitions, audit events, cancellation requests, authorization pauses, and a per-workspace SQLite lock. |
| Transactional mutations | `TransactionManager` snapshots regular files, atomically writes or deletes, verifies outcomes, and rolls back failed operations. |
| Secure tool surface | `filesystem.write_file`, `filesystem.delete_file`, `shell.execute`, and `python.execute` are risk-classified, verified, audited, and gated by runtime policy. |
| API lifecycle | `api/app.py` provides token-gated lifecycle endpoints, durable event history with SSE fanout, cancellation, authorization, replies, and run listing. |
| Sandbox boundary | `runtime/docker_sandbox.py` owns a Docker invocation with no network, read-only root, dropped capabilities, no-new-privileges, resource limits, an unprivileged user, and one validated writable workspace mount. |

## Prerequisites

The specification targets **Python 3.12+**, a local [Ollama](https://ollama.com/) service, Docker, and the following local models:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

The intended target machine is a local GPU-capable system. Model availability is checked by the `/health` endpoint; this repository does not send prompts or data to a hosted LLM API.

## Quick start

```bash
# 1. Create a Python 3.12 virtual environment, install dependencies, and copy the config template.
make setup

# 2. Set a strong AGENT_API_TOKEN in .env, then create workspace state and SQLite schema.
make bootstrap

# 3. Build the isolated execution sandbox image.
make sandbox-image

# 4. Run tests and static checks.
make test
make lint

# 5. With Docker running and the sandbox image built, execute the real isolation check.
RUN_DOCKER_INTEGRATION=1 PYTHONPATH=src .venv/bin/pytest -m docker

# 6. Start the FastAPI control plane.
make run
```

The API is served at `http://127.0.0.1:8000`, with interactive documentation at `/docs`. `GET /health` reports API, SQLite/workspace, and local Ollama model availability.

To start in a container, copy `.env.example` to `.env`, set `AGENT_API_TOKEN`, ensure host Ollama is available, then run:

```bash
docker compose up --build
```

## Configuration and security

Versioned defaults belong in `config/agent.toml`. Secrets and environment-specific overrides belong in `.env`, which is excluded from version control. The primary controls are summarized below.

| Setting | Purpose |
| --- | --- |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Local inference endpoint and Qwen model. |
| `EMBEDDING_MODEL` | Local embedding model for the later RAG pipeline. |
| `WORKSPACE_ROOT`, `SQLITE_PATH` | Isolated workspace and authoritative SQLite store. |
| `DEFAULT_MAX_*` | Runtime budget ceilings for tools, duration, and shell actions. |
| `DOCKER_SANDBOX_*` | Mandatory resource, user, filesystem, and network restrictions for sandboxed high-risk tools. |
| `AGENT_API_TOKEN` | Bearer token for run lifecycle endpoints. |

The repository deliberately keeps security decisions in the runtime layer. Shell and Python execution require explicit authorization, command policy approval, the Docker process boundary, output redaction, independent verification, and durable audit records. Workspace writes are snapshot-backed and only commit after post-operation verification.

## API lifecycle scaffold

| Method | Route | Initial behavior |
| --- | --- | --- |
| `GET` | `/health` | Checks SQLite, workspace, and local Ollama model readiness. |
| `POST` | `/runs` | Persists a new run, acquires its workspace lock, and emits a durable creation event. |
| `GET` | `/runs/{run_id}` | Retrieves persisted run metadata. |
| `GET` | `/runs/{run_id}/events` | Streams durable historical events followed by live SSE notifications. |
| `POST` | `/runs/{run_id}/cancel` | Persists cancellation for the execution boundary to honor before the next tool. |
| `POST` | `/runs/{run_id}/authorize` | Resolves an explicit pending authorization and moves the run to execute or partial state. |
| `GET` | `/runs/{run_id}/pending-authorization` | Returns the sanitized persisted pending tool action, when present. |
| `POST` | `/runs/{run_id}/reply` | Persists a user reply event for a paused runtime. |
| `GET` | `/runs` | Returns recent persisted runs. |

## Implementation order

The next commits should follow the specification's trust-first order:

1. Add context assembly, memory persistence/retrieval, stale-memory policy, and local FTS5 indexing.
2. Add durable ReAct resume replay, pending-action execution after authorization, and explicit run-worker orchestration.
3. Add production system-prompt versioning, prompt hashing, and end-to-end coding-task evaluation.
4. Extend secure tools only when each new operation has path policy, transaction semantics where applicable, independent verification, and audit coverage.

See [`docs/specification-alignment.md`](docs/specification-alignment.md) for the setup-to-specification mapping.
