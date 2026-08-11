"""Runtime-owned Docker execution boundary for approved future tool actions.

This module deliberately owns the Docker command construction. Model output never
selects Docker flags, mounts arbitrary host paths, passes host environment values,
or changes resource limits.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from local_ai_agent.config import Settings
from local_ai_agent.schemas.contracts import ToolResult, ToolStatus


class SandboxPolicyError(ValueError):
    """Raised when a requested sandbox execution violates host-side policy."""


class DockerSandboxExecutor:
    """Execute a validated command in a configuration-driven, isolated container.

    This executor is intentionally a process boundary only. Future tool handlers
    remain responsible for schema validation, allowlisting, authorization, backup,
    verification, and audit persistence before they delegate approved work here.
    """

    _CONTAINER_WORKSPACE = "/workspace/project"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build_command(
        self, *, command: str, workspace_path: Path, container_name: str | None = None
    ) -> list[str]:
        """Build a host-shell-free Docker invocation with mandatory isolation controls."""
        if not command.strip():
            raise SandboxPolicyError("Sandbox command must not be empty.")
        self._validate_policy()

        resolved_workspace = self._validate_workspace(workspace_path)
        resolved_container_name = container_name or self._new_container_name()
        docker_command = [
            self._settings.docker_binary,
            "run",
            "--rm",
            "--name",
            resolved_container_name,
            "--network",
            self._settings.docker_sandbox_network,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            str(self._settings.docker_sandbox_pids_limit),
            "--memory",
            self._settings.docker_sandbox_memory,
            "--cpus",
            str(self._settings.docker_sandbox_cpus),
            "--user",
            self._settings.docker_sandbox_user,
            "--tmpfs",
            f"/tmp:rw,noexec,nosuid,size={self._settings.docker_sandbox_tmpfs_size}",
            "--workdir",
            self._CONTAINER_WORKSPACE,
            "--mount",
            (f"type=bind,src={resolved_workspace},dst={self._CONTAINER_WORKSPACE},readonly=false"),
        ]
        if self._settings.docker_sandbox_read_only_root:
            docker_command.append("--read-only")
        docker_command.extend([self._settings.docker_sandbox_image, command])
        return docker_command

    def execute(
        self,
        *,
        tool_name: str,
        command: str,
        workspace_path: Path | None = None,
        timeout_seconds: int | None = None,
    ) -> ToolResult:
        """Run an approved command and serialize its observed process outcome.

        No host shell is invoked. The only caller-provided value passed to Docker is
        the final command argument handled by the sandbox image's fixed entrypoint.
        """
        timeout = timeout_seconds or self._settings.default_max_runtime_seconds
        if timeout <= 0:
            raise SandboxPolicyError("Sandbox timeout must be a positive number of seconds.")

        container_name = self._new_container_name()
        command_args = self.build_command(
            command=command,
            workspace_path=workspace_path or self._settings.workspace_project_path,
            container_name=container_name,
        )
        started = perf_counter()
        try:
            completed = subprocess.run(
                command_args,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return self._failure_result(
                tool_name=tool_name,
                error_code="DOCKER_UNAVAILABLE",
                error_message=f"Docker executable was not found: {self._settings.docker_binary}",
                retryable=False,
                metadata={"container_name": container_name},
            )
        except subprocess.TimeoutExpired as error:
            self._remove_timed_out_container(container_name)
            return self._failure_result(
                tool_name=tool_name,
                status=ToolStatus.TIMEOUT,
                error_code="SANDBOX_TIMEOUT",
                error_message=f"Sandbox execution exceeded {timeout} seconds.",
                retryable=True,
                data={
                    "stdout": self._as_text(error.stdout),
                    "stderr": self._as_text(error.stderr),
                    "exit_code": None,
                },
                metadata={
                    "container_name": container_name,
                    "duration_ms": self._duration_ms(started),
                    "timeout_seconds": timeout,
                },
            )
        except OSError as error:
            return self._failure_result(
                tool_name=tool_name,
                error_code="SANDBOX_LAUNCH_FAILED",
                error_message=f"Docker sandbox could not start: {error}",
                retryable=True,
                metadata={"container_name": container_name},
            )

        data = {
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        metadata = {
            "container_name": container_name,
            "duration_ms": self._duration_ms(started),
            "image": self._settings.docker_sandbox_image,
            "network": self._settings.docker_sandbox_network,
            "workspace_mount": self._CONTAINER_WORKSPACE,
        }
        if completed.returncode == 0:
            return ToolResult(
                tool_name=tool_name,
                status=ToolStatus.SUCCESS,
                success=True,
                data=data,
                metadata=metadata,
            )
        return self._failure_result(
            tool_name=tool_name,
            error_code="SANDBOX_COMMAND_FAILED",
            error_message=f"Sandbox command exited with code {completed.returncode}.",
            data=data,
            metadata=metadata,
        )

    def _validate_policy(self) -> None:
        """Reject configuration that would weaken the non-negotiable runtime boundary."""
        if self._settings.docker_sandbox_network != "none":
            raise SandboxPolicyError("Sandbox network must be disabled.")
        if not self._settings.docker_sandbox_read_only_root:
            raise SandboxPolicyError("Sandbox root filesystem must be read-only.")
        if self._settings.docker_sandbox_user.split(":", maxsplit=1)[0] in {"0", "root"}:
            raise SandboxPolicyError("Sandbox must not run as root.")
        if self._settings.docker_sandbox_pids_limit < 1:
            raise SandboxPolicyError("Sandbox PID limit must be positive.")

    def _validate_workspace(self, workspace_path: Path) -> Path:
        resolved_root = self._settings.workspace_project_path.resolve()
        resolved_workspace = workspace_path.resolve()
        if not resolved_workspace.is_dir():
            raise SandboxPolicyError(
                f"Sandbox workspace does not exist or is not a directory: {workspace_path}"
            )
        try:
            resolved_workspace.relative_to(resolved_root)
        except ValueError as error:
            raise SandboxPolicyError(
                "Sandbox workspace must be inside the approved project workspace."
            ) from error
        return resolved_workspace

    def _remove_timed_out_container(self, container_name: str) -> None:
        """Best-effort cleanup if terminating the Docker client leaves a container behind."""
        try:
            subprocess.run(
                [self._settings.docker_binary, "rm", "--force", container_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except OSError:
            return

    @staticmethod
    def _as_text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value

    @staticmethod
    def _duration_ms(started: float) -> int:
        return round((perf_counter() - started) * 1_000)

    @staticmethod
    def _new_container_name() -> str:
        return f"local-ai-agent-{uuid4().hex}"

    @staticmethod
    def _failure_result(
        *,
        tool_name: str,
        error_code: str,
        error_message: str,
        retryable: bool = False,
        status: ToolStatus = ToolStatus.ERROR,
        data: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            status=status,
            success=False,
            data=data,
            error_code=error_code,
            error_message=error_message,
            retryable=retryable,
            metadata=metadata or {},
        )
