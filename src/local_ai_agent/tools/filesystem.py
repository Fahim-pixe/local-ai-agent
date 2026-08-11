"""Read-only filesystem tools registered through the runtime-owned ToolRegistry."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_ai_agent.config import Settings
from local_ai_agent.schemas.contracts import RiskLevel, ToolResult, ToolStatus, VerificationResult
from local_ai_agent.security.paths import WorkspacePolicyError, resolve_workspace_path
from local_ai_agent.tools.registry import ToolDefinition


class ReadOnlyPathArguments(BaseModel):
    """Shared strict argument contract for the initial read-only filesystem operations."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=4_000)


def build_read_only_filesystem_tools(settings: Settings) -> list[ToolDefinition]:
    """Build the initial LOW-risk tool definitions from central runtime configuration."""
    service = ReadOnlyFilesystemTools(settings)
    schema = ReadOnlyPathArguments.model_json_schema()
    return [
        ToolDefinition(
            name="filesystem.list_directory",
            description="List entries in a directory inside the approved project workspace.",
            input_schema=schema,
            risk=RiskLevel.LOW,
            handler=service.list_directory,
            verification=service.verify_list_directory,
            arguments_validator=validate_read_only_path_arguments,
        ),
        ToolDefinition(
            name="filesystem.read_file",
            description="Read a bounded UTF-8 representation of a file inside the approved project workspace.",
            input_schema=schema,
            risk=RiskLevel.LOW,
            handler=service.read_file,
            verification=service.verify_read_file,
            arguments_validator=validate_read_only_path_arguments,
        ),
    ]


def validate_read_only_path_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate native tool-call arguments before policy or handler execution."""
    try:
        return ReadOnlyPathArguments.model_validate(arguments).model_dump()
    except ValidationError as error:
        raise ValueError("Invalid read-only filesystem arguments.") from error


class ReadOnlyFilesystemTools:
    """Handlers and independent verifiers for initial workspace-scoped read operations."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def list_directory(self, arguments: dict[str, Any]) -> ToolResult:
        target_or_error = self._resolve("filesystem.list_directory", arguments["path"])
        if isinstance(target_or_error, ToolResult):
            return target_or_error
        target = target_or_error
        if not target.exists():
            return self._not_found("filesystem.list_directory", arguments["path"])
        if not target.is_dir():
            return self._invalid_target("filesystem.list_directory", "Path is not a directory.")

        entries, truncated = self._list_entries(target)
        return ToolResult(
            tool_name="filesystem.list_directory",
            status=ToolStatus.SUCCESS,
            success=True,
            data={
                "path": arguments["path"],
                "entries": entries,
                "truncated": truncated,
                "entry_limit": self._settings.filesystem_max_list_entries,
            },
        )

    async def read_file(self, arguments: dict[str, Any]) -> ToolResult:
        target_or_error = self._resolve("filesystem.read_file", arguments["path"])
        if isinstance(target_or_error, ToolResult):
            return target_or_error
        target = target_or_error
        if not target.exists():
            return self._not_found("filesystem.read_file", arguments["path"])
        if not target.is_file():
            return self._invalid_target("filesystem.read_file", "Path is not a regular file.")

        content, content_hash, truncated = self._read_bounded(target)
        return ToolResult(
            tool_name="filesystem.read_file",
            status=ToolStatus.SUCCESS,
            success=True,
            data={
                "path": arguments["path"],
                "content": content,
                "content_sha256": content_hash,
                "truncated": truncated,
                "byte_limit": self._settings.filesystem_max_read_bytes,
            },
        )

    async def verify_list_directory(
        self, arguments: dict[str, Any], result: ToolResult
    ) -> VerificationResult:
        target_or_error = self._resolve("filesystem.list_directory", arguments["path"])
        if isinstance(target_or_error, ToolResult) or not target_or_error.is_dir():
            return VerificationResult(
                verified=False,
                strategy="directory-relist",
                message="Directory is no longer accessible for verification.",
            )
        expected_entries, expected_truncated = self._list_entries(target_or_error)
        actual_data = result.data if isinstance(result.data, dict) else {}
        verified = (
            actual_data.get("entries") == expected_entries
            and actual_data.get("truncated") == expected_truncated
        )
        return VerificationResult(
            verified=verified,
            strategy="directory-relist",
            evidence={"entry_count": len(expected_entries), "truncated": expected_truncated},
            message=None
            if verified
            else "Directory contents changed between execution and verification.",
        )

    async def verify_read_file(
        self, arguments: dict[str, Any], result: ToolResult
    ) -> VerificationResult:
        target_or_error = self._resolve("filesystem.read_file", arguments["path"])
        if isinstance(target_or_error, ToolResult) or not target_or_error.is_file():
            return VerificationResult(
                verified=False,
                strategy="file-reread-hash",
                message="File is no longer accessible for verification.",
            )
        _, expected_hash, expected_truncated = self._read_bounded(target_or_error)
        actual_data = result.data if isinstance(result.data, dict) else {}
        verified = (
            actual_data.get("content_sha256") == expected_hash
            and actual_data.get("truncated") == expected_truncated
        )
        return VerificationResult(
            verified=verified,
            strategy="file-reread-hash",
            evidence={"content_sha256": expected_hash, "truncated": expected_truncated},
            message=None
            if verified
            else "File contents changed between execution and verification.",
        )

    def _resolve(self, tool_name: str, candidate: str) -> Path | ToolResult:
        try:
            return resolve_workspace_path(
                workspace_project=self._settings.workspace_project_path,
                candidate=candidate,
                protected_paths=(
                    self._settings.workspace_internal_path,
                    self._settings.workspace_backups_path,
                ),
            )
        except WorkspacePolicyError as error:
            return ToolResult(
                tool_name=tool_name,
                status=ToolStatus.ERROR,
                success=False,
                error_code="POLICY_BLOCK",
                error_message=str(error),
            )

    def _list_entries(self, target: Path) -> tuple[list[dict[str, str]], bool]:
        entries = sorted(target.iterdir(), key=lambda item: item.name)
        visible_entries = entries[: self._settings.filesystem_max_list_entries]
        return [self._entry_metadata(entry) for entry in visible_entries], len(entries) > len(
            visible_entries
        )

    @staticmethod
    def _entry_metadata(entry: Path) -> dict[str, str]:
        if entry.is_symlink():
            entry_type = "symlink"
        elif entry.is_dir():
            entry_type = "directory"
        elif entry.is_file():
            entry_type = "file"
        else:
            entry_type = "other"
        return {"name": entry.name, "type": entry_type}

    def _read_bounded(self, target: Path) -> tuple[str, str, bool]:
        with target.open("rb") as source:
            content = source.read(self._settings.filesystem_max_read_bytes + 1)
        truncated = len(content) > self._settings.filesystem_max_read_bytes
        visible_content = content[: self._settings.filesystem_max_read_bytes]
        return (
            visible_content.decode("utf-8", errors="replace"),
            hashlib.sha256(visible_content).hexdigest(),
            truncated,
        )

    @staticmethod
    def _not_found(tool_name: str, path: str) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            status=ToolStatus.ERROR,
            success=False,
            error_code="NOT_FOUND",
            error_message=f"Workspace path was not found: {path}",
        )

    @staticmethod
    def _invalid_target(tool_name: str, message: str) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            status=ToolStatus.ERROR,
            success=False,
            error_code="INVALID_INPUT",
            error_message=message,
        )
