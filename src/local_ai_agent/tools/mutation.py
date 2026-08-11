"""Transactional workspace mutation and authorization-gated sandbox execution tools."""

from __future__ import annotations

import hashlib
import shlex
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_ai_agent.config import Settings
from local_ai_agent.db.repository import RunRepository
from local_ai_agent.runtime.docker_sandbox import DockerSandboxExecutor, SandboxPolicyError
from local_ai_agent.runtime.transaction_manager import TransactionError, TransactionManager
from local_ai_agent.schemas.contracts import RiskLevel, ToolResult, ToolStatus, VerificationResult
from local_ai_agent.security.command_policy import CommandPolicy, CommandPolicyError
from local_ai_agent.security.output_scrubber import SecretScrubber
from local_ai_agent.security.paths import WorkspacePolicyError, resolve_workspace_path
from local_ai_agent.tools.registry import ToolDefinition


class WriteFileArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=4_000)
    content: str = Field(max_length=1_000_000)


class DeleteFileArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=4_000)


class ShellArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(min_length=1, max_length=8_000)


class PythonArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=16_000)


def build_mutation_tools(
    *, settings: Settings, run_id: UUID, repository: RunRepository
) -> list[ToolDefinition]:
    """Build tools that require a durable run context for backups and audit records."""
    service = MutationTools(settings=settings, run_id=run_id, repository=repository)
    return [
        ToolDefinition(
            name="filesystem.write_file",
            description="Transactionally write a UTF-8 file inside the approved workspace.",
            input_schema=WriteFileArguments.model_json_schema(),
            risk=RiskLevel.MEDIUM,
            handler=service.write_file,
            verification=service.verify_write_file,
            arguments_validator=_validator(WriteFileArguments),
        ),
        ToolDefinition(
            name="filesystem.delete_file",
            description="Delete a workspace file only after explicit authorization and a backup snapshot.",
            input_schema=DeleteFileArguments.model_json_schema(),
            risk=RiskLevel.HIGH,
            handler=service.delete_file,
            verification=service.verify_delete_file,
            arguments_validator=_validator(DeleteFileArguments),
        ),
        ToolDefinition(
            name="shell.execute",
            description="Execute an explicitly authorized allowlisted command in the isolated Docker sandbox.",
            input_schema=ShellArguments.model_json_schema(),
            risk=RiskLevel.HIGH,
            handler=service.execute_shell,
            verification=service.verify_process,
            arguments_validator=_validator(ShellArguments),
        ),
        ToolDefinition(
            name="python.execute",
            description="Execute explicitly authorized Python code in the isolated Docker sandbox.",
            input_schema=PythonArguments.model_json_schema(),
            risk=RiskLevel.HIGH,
            handler=service.execute_python,
            verification=service.verify_process,
            arguments_validator=_validator(PythonArguments),
        ),
    ]


def _validator(model: type[BaseModel]):
    def validate(arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            return model.model_validate(arguments).model_dump()
        except ValidationError as error:
            raise ValueError(f"Invalid {model.__name__} arguments.") from error

    return validate


class MutationTools:
    """Handlers with reversible workspace changes and sandboxed process execution."""

    def __init__(self, *, settings: Settings, run_id: UUID, repository: RunRepository) -> None:
        self._settings = settings
        self._run_id = run_id
        self._repository = repository
        self._transactions = TransactionManager(settings)
        self._command_policy = CommandPolicy.from_allowlist(settings.shell_allowlist)
        self._sandbox = DockerSandboxExecutor(settings)
        self._scrubber = SecretScrubber()

    async def write_file(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            transaction = self._transactions.write_text(
                run_id=self._run_id, candidate=arguments["path"], content=arguments["content"]
            )
            backup_id = self._repository.record_file_backup(
                run_id=self._run_id,
                original_path=transaction.snapshot.original_path,
                backup_path=transaction.snapshot.backup_path,
                original_hash=transaction.snapshot.original_hash,
            )
            return ToolResult(
                tool_name="filesystem.write_file",
                status=ToolStatus.SUCCESS,
                success=True,
                data={
                    "path": arguments["path"],
                    "content_sha256": transaction.content_hash,
                    "operation": transaction.operation,
                },
                metadata={"backup_id": backup_id},
            )
        except WorkspacePolicyError as error:
            return self._policy_block("filesystem.write_file", error)
        except TransactionError as error:
            return ToolResult(
                tool_name="filesystem.write_file",
                status=ToolStatus.ERROR,
                success=False,
                error_code="TRANSACTION_FAILED",
                error_message=str(error),
            )

    async def delete_file(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            transaction = self._transactions.delete_file(
                run_id=self._run_id, candidate=arguments["path"]
            )
            backup_id = self._repository.record_file_backup(
                run_id=self._run_id,
                original_path=transaction.snapshot.original_path,
                backup_path=transaction.snapshot.backup_path,
                original_hash=transaction.snapshot.original_hash,
            )
            return ToolResult(
                tool_name="filesystem.delete_file",
                status=ToolStatus.SUCCESS,
                success=True,
                data={"path": arguments["path"], "operation": transaction.operation},
                metadata={"backup_id": backup_id},
            )
        except WorkspacePolicyError as error:
            return self._policy_block("filesystem.delete_file", error)
        except TransactionError as error:
            return ToolResult(
                tool_name="filesystem.delete_file",
                status=ToolStatus.ERROR,
                success=False,
                error_code="TRANSACTION_FAILED",
                error_message=str(error),
            )

    async def execute_shell(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            command = self._command_policy.validate(arguments["command"])
            return self._scrub_result(
                self._sandbox.execute(
                    tool_name="shell.execute",
                    command=command,
                    workspace_path=self._settings.workspace_project_path,
                )
            )
        except (CommandPolicyError, SandboxPolicyError) as error:
            return self._policy_block("shell.execute", error)

    async def execute_python(self, arguments: dict[str, Any]) -> ToolResult:
        command = f"python3 -c {shlex.quote(arguments['code'])}"
        try:
            self._command_policy.validate(command)
            return self._scrub_result(
                self._sandbox.execute(
                    tool_name="python.execute",
                    command=command,
                    workspace_path=self._settings.workspace_project_path,
                )
            )
        except (CommandPolicyError, SandboxPolicyError) as error:
            return self._policy_block("python.execute", error)

    async def verify_write_file(
        self, arguments: dict[str, Any], result: ToolResult
    ) -> VerificationResult:
        try:
            target = self._resolve(arguments["path"])
            expected_hash = (
                result.data.get("content_sha256") if isinstance(result.data, dict) else None
            )
            actual_hash = self._hash_file(target) if target.is_file() else None
            verified = actual_hash is not None and actual_hash == expected_hash
            return VerificationResult(
                verified=verified,
                strategy="file-reread-hash",
                evidence={"content_sha256": actual_hash},
                message=None
                if verified
                else "Written file hash did not match the committed result.",
            )
        except WorkspacePolicyError:
            return VerificationResult(
                verified=False,
                strategy="file-reread-hash",
                message="Written path violates workspace policy.",
            )

    async def verify_delete_file(
        self, arguments: dict[str, Any], _: ToolResult
    ) -> VerificationResult:
        try:
            target = self._resolve(arguments["path"])
            verified = not target.exists()
            return VerificationResult(
                verified=verified,
                strategy="path-absence",
                evidence={"exists": target.exists()},
                message=None if verified else "Deleted file is still present.",
            )
        except WorkspacePolicyError:
            return VerificationResult(
                verified=False,
                strategy="path-absence",
                message="Deleted path violates workspace policy.",
            )

    async def verify_process(self, _: dict[str, Any], result: ToolResult) -> VerificationResult:
        data = result.data if isinstance(result.data, dict) else {}
        exit_code = data.get("exit_code")
        stderr = data.get("stderr")
        verified = exit_code == 0 and not stderr
        return VerificationResult(
            verified=verified,
            strategy="process-exit-and-stderr",
            evidence={"exit_code": exit_code, "stderr_empty": not bool(stderr)},
            message=None if verified else "Sandboxed process did not exit cleanly.",
        )

    def _resolve(self, candidate: str) -> Path:
        return resolve_workspace_path(
            workspace_project=self._settings.workspace_project_path,
            candidate=candidate,
            protected_paths=(
                self._settings.workspace_internal_path,
                self._settings.workspace_backups_path,
            ),
        )

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(64 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _scrub_result(self, result: ToolResult) -> ToolResult:
        return result.model_copy(
            update={
                "data": self._scrubber.scrub(result.data),
                "error_message": self._scrubber.scrub_text(result.error_message),
                "metadata": self._scrubber.scrub(result.metadata),
            }
        )

    @staticmethod
    def _policy_block(tool_name: str, error: Exception) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            status=ToolStatus.ERROR,
            success=False,
            error_code="POLICY_BLOCK",
            error_message=str(error),
        )
