# Local AI Agent

A **contract-first, local-first** autonomous agent runtime. The local model decides what to do; the Python runtime validates whether it is allowed; tools run only through runtime policy; SQLite records the authoritative state.

> **Current status:** The contract-first runtime now includes a versioned production system prompt, SHA-256 prompt provenance on each API-created run, native Ollama tool calls, checkpointed ReAct continuation, verified memory/context assembly, transactional workspace mutation, sandboxed execution, and bounded coordinator-to-specialist delegation. An opt-in local Qwen3 coding corpus validates the complete path when Ollama is available.

## Architecture

| Layer | Responsibility |
| --- | --- |
| Local model | Qwen receives the versioned production prompt and proposes native Ollama tool calls. It never directly executes tools. |
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
| Bounded coordinator delegation | `DelegationCoordinator` persists an explicit `AgentPlan`, admits only dependency-ready specialist units, and accepts only compact, independently verified `SpecialistEvidence`. Specialists receive frozen `DelegationAuthority`, so they cannot add tools, widen permissions, increase budgets, or alter retry policy. |
| Minimal ReAct capability | `ReActLoop` uses Ollama native tool calls and returns only verified results from `filesystem.list_directory` and `filesystem.read_file` inside `workspace/project/`. |
| Local model boundary | `runtime/ollama_client.py` uses native Ollama `tools` payloads and explicit local failure types. |
| SQLite source of truth | `db/schema.py` creates runs, tool call/result, memory, event, backup, and FTS5 tables. |
| Workspace isolation | `security/paths.py` resolves symlinks before containment checks. |
| Durable lifecycle and concurrency | `RunLifecycleService` persists state transitions, audit events, cancellation requests, authorization pauses, and a per-workspace SQLite lock. |
| Transactional mutations | `TransactionManager` snapshots regular files, atomically writes or deletes, verifies outcomes, and rolls back failed operations. |
| Secure tool surface | `filesystem.write_file`, `filesystem.delete_file`, `shell.execute`, and `python.execute` are risk-classified, verified, audited, and gated by runtime policy. |
| Memory and retrieval | `MemoryRepository` upserts confidence-labeled records, promotes expired entries to `STALE`, and retrieves semantic/long-term memory through rebuilt-safe SQLite FTS5. |
| Context assembly | `ContextManager` preserves P0 runtime state, keeps fitting P1 evidence and retrieved memory, truncates P2 history with line-range hints, and drops P3 duplicate/old verified outputs. |
| Verified memory tool | `memory.store` is a registered medium-risk tool with secret redaction and retrieval-based verification. |
| Production prompt provenance | `config/agent.toml` provides agent name and mission, which render into `config/system_prompt.md` before SHA-256 hashing; every API-created run records the exact runtime-owned prompt hash in SQLite. |
| Durable ReAct continuation and dispatch | Append-only message checkpoints bind high-risk requests to a single pending action. When explicitly enabled, the API supervises a bounded set of independent local worker processes that register, heartbeat, atomically claim, and execute one owned action; all current tools are `NEVER_RECLAIM`, so expired `EXECUTING` work fails safely rather than being re-executed. |
| API lifecycle | `api/app.py` provides token-gated lifecycle endpoints, durable event history with SSE fanout, cancellation, authorization, replies, run listing, approved-action continuation, and resume-token validation. |
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

# 6. Run the opt-in real local-model coding evaluation after confirming Ollama and qwen3:8b are available.
make test-ollama

# 7. Start the FastAPI control plane.
make run
```

The API is served at `http://127.0.0.1:8000`, with interactive documentation at `/docs`. `GET /health` reports API, SQLite/workspace, and local Ollama model availability.

Dispatch remains disabled by default. After validating the native Ubuntu environment, enable the bounded local worker supervisor explicitly for an API process with:

```bash
DISPATCH_ENABLED=true make run
```

The supervisor starts at most `DISPATCH_MAX_WORKERS` independent worker processes. They share only SQLite durable state; they do not share Python memory and never automatically re-execute a recovered action.

To start in a container, copy `.env.example` to `.env`, set `AGENT_API_TOKEN`, ensure host Ollama is available, then run:

```bash
docker compose up --build
```

## Configuration and security

Versioned defaults belong in `config/agent.toml`. Secrets and environment-specific overrides belong in `.env`, which is excluded from version control. The primary controls are summarized below.

| Setting | Purpose |
| --- | --- |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL` | Local inference endpoint and Qwen model. |
| `AGENT_NAME`, `AGENT_MISSION` | Optional overrides for the canonical versioned `[agent]` identity in `config/agent.toml`; rendered into the production prompt before hashing. |
| `SYSTEM_PROMPT_PATH` | Optional override for the checked-in versioned production prompt source. |
| `EMBEDDING_MODEL` | Local embedding model for the later RAG pipeline. |
| `WORKSPACE_ROOT`, `SQLITE_PATH` | Isolated workspace and authoritative SQLite store. |
| `DEFAULT_MAX_*` | Runtime budget ceilings for tools, duration, and shell actions. |
| `RAG_*`, `CONTEXT_CHARS_PER_TOKEN` | FTS retrieval count and token-estimation/truncation controls for assembled runtime context. |
| `WORKER_LEASE_SECONDS`, `WORKER_HEARTBEAT_SECONDS` | Lease expiry and renewal interval for executing approved actions; stale claims are safely recovered as failures at startup. |
| `DISPATCH_*` | Disabled-by-default bounded local worker count, polling, recovery sweep, worker staleness, and attempt-cap policy. `DISPATCH_MAX_ATTEMPTS` remains `1` until a separately reviewed idempotency pilot. |
| `DELEGATION_*` | Coordinator-owned caps for specialist unit count, per-unit tool calls, model turns, and retries. `DELEGATION_MAX_RETRIES` remains `0`; specialist units fail closed rather than retrying autonomously. |
| `DOCKER_SANDBOX_*` | Mandatory resource, user, filesystem, and network restrictions for sandboxed high-risk tools. |
| `AGENT_API_TOKEN` | Bearer token for run lifecycle endpoints. |

The repository deliberately keeps security decisions in the runtime layer. Shell and Python execution require explicit authorization, command policy approval, the Docker process boundary, output redaction, independent verification, and durable audit records. Workspace writes are snapshot-backed and only commit after post-operation verification. Coordinator delegation follows the same boundary: the coordinator fixes a specialist unit’s allowed tools, tool-call cap, model-turn cap, and zero-retry policy before execution; raw tool output is excluded from later model context, and only compact verified evidence summaries are retained.

## API lifecycle scaffold

| Method | Route | Initial behavior |
| --- | --- | --- |
| `GET` | `/health` | Checks SQLite, workspace, and local Ollama model readiness. |
| `POST` | `/runs` | Persists a new run, acquires its workspace lock, and emits a durable creation event. |
| `GET` | `/runs/{run_id}` | Retrieves persisted run metadata. |
| `GET` | `/runs/{run_id}/delegation` | Returns the durable coordinator plan-step mapping, current unit states, and verified evidence summaries only. |
| `GET` | `/runs/{run_id}/events` | Streams durable historical events followed by live SSE notifications. |
| `POST` | `/runs/{run_id}/cancel` | Persists cancellation for the execution boundary to honor before the next tool. |
| `POST` | `/runs/{run_id}/authorize` | Resolves an explicit pending authorization and marks its checkpoint-linked action approved or rejected. |
| `POST` | `/runs/{run_id}/continue` | Atomically claims one approved action, executes it through the secure tool path, checkpoints the result, and replays the stored ReAct conversation. |
| `POST` | `/runs/{run_id}/resume` | Requires the run’s persisted `resume_token` and then atomically resumes one approved checkpointed action. |
| `GET` | `/runs/{run_id}/pending-authorization` | Returns the sanitized persisted pending tool action, when present. |
| `GET` | `/runs/{run_id}/actions` | Returns durable action history, including worker owner, recovery class, operation-key prefix, dispatch count, lease expiry, and crash-recovery metadata. |
| `GET` | `/workers`, `/workers/{worker_id}` | Returns registered local worker identity, capabilities, state, and heartbeat metadata. |
| `POST` | `/workers/{worker_id}/drain` | Prevents an active worker from claiming additional work and supports graceful shutdown. |
| `GET` | `/actions/{action_id}/attempts` | Returns append-only claim, terminal, and crash-recovery evidence for one durable action. |
| `POST` | `/runs/{run_id}/reply` | Persists a user reply event for a paused runtime. |
| `GET` | `/runs` | Returns recent persisted runs. |

## Implementation order

The next commits should follow the specification's trust-first order:

1. Bind the bounded coordinator to an explicitly reviewed specialist runner only after preserving the current immutable-authority, verified-summary, and zero-retry contract; no specialist may issue direct tool calls outside the runtime router.
2. Baseline configuration-gated local dispatch under native Ubuntu, then add sampled lease metrics and consider a narrowly reviewed verifier-backed idempotency pilot; no high-risk tool may be automatically re-executed.
3. Run and baseline the opt-in local Qwen3 coding corpus on target hardware before expanding its task set.
4. Extend semantic memory with local embeddings only after evaluating the FTS5 baseline and preserving the current confidence/staleness model.
5. Extend secure tools only when each new operation has path policy, transaction semantics where applicable, verification, recovery classification, and audit coverage.

See [`docs/specification-alignment.md`](docs/specification-alignment.md) for the setup-to-specification mapping.
