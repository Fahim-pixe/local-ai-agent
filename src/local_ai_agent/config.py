"""Configuration loading and workspace bootstrap helpers."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "agent.toml"


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings; environment variables override versioned defaults."""

    project_root: Path
    workspace_root: Path
    sqlite_path: Path
    ollama_base_url: str
    ollama_model: str
    embedding_model: str
    model_context_tokens: int
    agent_api_token: str | None
    default_max_tool_calls: int
    default_max_runtime_seconds: int
    default_max_shell_executions: int
    default_max_retries: int
    context_reserve_tokens: int
    recent_tool_results: int
    recent_conversation_messages: int
    rag_chunk_tokens: int
    rag_chunk_overlap_tokens: int
    rag_top_k: int
    context_chars_per_token: int
    filesystem_max_read_bytes: int
    filesystem_max_list_entries: int
    shell_allowlist: tuple[str, ...]
    docker_binary: str
    docker_sandbox_image: str
    docker_sandbox_network: str
    docker_sandbox_memory: str
    docker_sandbox_cpus: float
    docker_sandbox_pids_limit: int
    docker_sandbox_user: str
    docker_sandbox_tmpfs_size: str
    docker_sandbox_read_only_root: bool

    @property
    def workspace_project_path(self) -> Path:
        return self.workspace_root / "project"

    @property
    def workspace_backups_path(self) -> Path:
        return self.workspace_root / "backups"

    @property
    def workspace_internal_path(self) -> Path:
        return self.workspace_root / ".agent"

    @property
    def workspace_runs_path(self) -> Path:
        return self.workspace_root / "runs"

    @property
    def workspace_logs_path(self) -> Path:
        return self.workspace_root / "logs"

    @property
    def workspace_system_path(self) -> Path:
        return self.workspace_root / "system"


def load_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Load TOML defaults, then apply explicit environment-variable overrides."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    raw = _read_config(config_path)
    ollama = raw["ollama"]
    limits = raw["limits"]
    context = raw["context"]
    filesystem = raw["filesystem"]
    tool_policy = raw["tool_policy"]
    sandbox = raw["sandbox"]
    workspace = raw["workspace"]

    project_root = PROJECT_ROOT
    workspace_root = _env_path("WORKSPACE_ROOT", project_root / workspace["root"])
    sqlite_path = _env_path("SQLITE_PATH", workspace_root / ".agent" / "agent.db")

    return Settings(
        project_root=project_root,
        workspace_root=workspace_root,
        sqlite_path=sqlite_path,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", ollama["base_url"]).rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", ollama["model"]),
        embedding_model=os.getenv("EMBEDDING_MODEL", ollama["embedding_model"]),
        model_context_tokens=_env_int("MODEL_CONTEXT_TOKENS", ollama["model_context_tokens"]),
        agent_api_token=os.getenv("AGENT_API_TOKEN"),
        default_max_tool_calls=_env_int("DEFAULT_MAX_TOOL_CALLS", limits["default_max_tool_calls"]),
        default_max_runtime_seconds=_env_int(
            "DEFAULT_MAX_RUNTIME_SECONDS", limits["default_max_runtime_seconds"]
        ),
        default_max_shell_executions=_env_int(
            "DEFAULT_MAX_SHELL_EXECUTIONS", limits["default_max_shell_executions"]
        ),
        default_max_retries=_env_int("DEFAULT_MAX_RETRIES", limits["default_max_retries"]),
        context_reserve_tokens=_env_int("CONTEXT_RESERVE_TOKENS", context["reserve_tokens"]),
        recent_tool_results=_env_int("RECENT_TOOL_RESULTS", context["recent_tool_results"]),
        recent_conversation_messages=_env_int(
            "RECENT_CONVERSATION", context["recent_conversation_messages"]
        ),
        rag_chunk_tokens=_env_int("RAG_CHUNK_TOKENS", context["rag_chunk_tokens"]),
        rag_chunk_overlap_tokens=_env_int(
            "RAG_CHUNK_OVERLAP_TOKENS", context["rag_chunk_overlap_tokens"]
        ),
        rag_top_k=_env_int("RAG_TOP_K", context["rag_top_k"]),
        context_chars_per_token=_env_int("CONTEXT_CHARS_PER_TOKEN", context["chars_per_token"]),
        filesystem_max_read_bytes=_env_int(
            "FILESYSTEM_MAX_READ_BYTES", filesystem["max_read_bytes"]
        ),
        filesystem_max_list_entries=_env_int(
            "FILESYSTEM_MAX_LIST_ENTRIES", filesystem["max_list_entries"]
        ),
        shell_allowlist=tuple(tool_policy["shell_allowlist"]),
        docker_binary=os.getenv("DOCKER_BINARY", sandbox["docker_binary"]),
        docker_sandbox_image=os.getenv("DOCKER_SANDBOX_IMAGE", sandbox["image"]),
        docker_sandbox_network=os.getenv("DOCKER_SANDBOX_NETWORK", sandbox["network"]),
        docker_sandbox_memory=os.getenv("DOCKER_SANDBOX_MEMORY", sandbox["memory"]),
        docker_sandbox_cpus=_env_float("DOCKER_SANDBOX_CPUS", sandbox["cpus"]),
        docker_sandbox_pids_limit=_env_int("DOCKER_SANDBOX_PIDS_LIMIT", sandbox["pids_limit"]),
        docker_sandbox_user=os.getenv("DOCKER_SANDBOX_USER", sandbox["user"]),
        docker_sandbox_tmpfs_size=os.getenv("DOCKER_SANDBOX_TMPFS_SIZE", sandbox["tmpfs_size"]),
        docker_sandbox_read_only_root=_env_bool(
            "DOCKER_SANDBOX_READ_ONLY_ROOT", sandbox["read_only_root"]
        ),
    )


def ensure_workspace(settings: Settings) -> None:
    """Create the runtime-owned workspace structure before database initialization."""
    for directory in (
        settings.workspace_root,
        settings.workspace_project_path,
        settings.workspace_backups_path,
        settings.workspace_internal_path,
        settings.workspace_runs_path,
        settings.workspace_logs_path,
        settings.workspace_system_path,
    ):
        directory.mkdir(parents=True, exist_ok=True)
