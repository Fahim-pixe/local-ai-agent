# Local AI Agent

A **contract-first, local-first** autonomous agent runtime. The local model decides what to do; the Python runtime validates whether it is allowed; tools run only through runtime policy; SQLite records the authoritative state.

> **Current status:** This repository is initialized as the secure foundation for the engineering specification. It provides typed contracts, state transitions, output validation, SQLite bootstrap, workspace boundaries, FastAPI lifecycle scaffolding, local Ollama integration boundaries, Docker definitions, and focused tests. The complete execution loop and tool implementations are intentionally next-phase work.

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
| Local model boundary | `runtime/ollama_client.py` uses native Ollama `tools` payloads and explicit local failure types. |
| SQLite source of truth | `db/schema.py` creates runs, tool call/result, memory, event, backup, and FTS5 tables. |
| Workspace isolation | `security/paths.py` resolves symlinks before containment checks. |
| Session/API scaffold | `api/app.py` initializes FastAPI, token-gated lifecycle endpoints, health checks, and SSE-ready event streams. |
| Sandbox boundary | `docker/sandbox/Dockerfile` is the dedicated base image for future approved tool execution. |

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

# 3. Build the future execution sandbox image.
make sandbox-image

# 4. Run tests and static checks.
make test
make lint

# 5. Start the FastAPI control plane.
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
| `DOCKER_SANDBOX_*` | Resource and network restrictions for future tool execution. |
| `AGENT_API_TOKEN` | Bearer token for run lifecycle endpoints. |

The repository deliberately keeps security decisions in the runtime layer. The future shell/Python tools must run only after schema validation, policy checks, path sanitization, risk/authorization checks, budget and loop checks, snapshots, execution, verification, and audit logging.

## API lifecycle scaffold

| Method | Route | Initial behavior |
| --- | --- | --- |
| `GET` | `/health` | Checks SQLite, workspace, and local Ollama model readiness. |
| `POST` | `/runs` | Validates and persists a new run request. |
| `GET` | `/runs/{run_id}` | Retrieves persisted run metadata. |
| `GET` | `/runs/{run_id}/events` | Opens an SSE-ready event stream. |
| `POST` | `/runs/{run_id}/cancel` | Accepts a cancellation request for the future runtime loop. |
| `POST` | `/runs/{run_id}/authorize` | Receives approval/denial for a gated future tool action. |
| `GET` | `/runs/{run_id}/pending-authorization` | Provides the current authorization placeholder. |
| `POST` | `/runs/{run_id}/reply` | Accepts user clarification for a paused future run. |
| `GET` | `/runs` | Reserved for paginated run listing implementation. |

## Implementation order

The next commits should follow the specification's trust-first order:

1. Implement the **minimal ReAct loop** and read-only filesystem tools through the `ToolRegistry`.
2. Add **permission gates, budgets, retries, argument-hash loop detection, and operation-aware verification** to the execution pipeline.
3. Implement transactional writes, a Docker-backed executor, secret scrubbing, and per-session locking.
4. Add context assembly, memory persistence/retrieval, and local FTS5 indexing.
5. Complete API lifecycle behavior, persistent SSE audit events, authorization pause/resume, and resume tokens.
6. Add the production system prompt and end-to-end coding-task evaluation suite.

See [`docs/specification-alignment.md`](docs/specification-alignment.md) for the setup-to-specification mapping.
