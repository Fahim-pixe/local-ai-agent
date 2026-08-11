"""Transactional workspace mutation with durable snapshots and rollback."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from local_ai_agent.config import Settings
from local_ai_agent.security.paths import resolve_workspace_path


class TransactionError(RuntimeError):
    """Raised when a workspace mutation cannot be committed and rollback was attempted."""


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    run_id: UUID
    original_path: Path
    backup_path: Path
    original_hash: str
    existed: bool


@dataclass(frozen=True, slots=True)
class TransactionResult:
    operation: str
    path: Path
    content_hash: str | None
    snapshot: FileSnapshot
    committed: bool


class TransactionManager:
    """Perform reversible workspace writes and deletes beneath the policy-approved root."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def write_text(self, *, run_id: UUID, candidate: str, content: str) -> TransactionResult:
        target = self._resolve(candidate)
        snapshot = self._snapshot(run_id, target)
        try:
            encoded = content.encode("utf-8")
            temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
            with temporary.open("wb") as destination:
                destination.write(encoded)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, target)
            content_hash = self._hash_file(target)
            expected_hash = hashlib.sha256(encoded).hexdigest()
            if content_hash != expected_hash:
                raise TransactionError("Post-write content hash did not match expected content.")
            return TransactionResult("write", target, content_hash, snapshot, committed=True)
        except Exception as error:
            self._rollback(snapshot)
            raise TransactionError(
                f"Write transaction failed and rollback was attempted: {error}"
            ) from error

    def delete_file(self, *, run_id: UUID, candidate: str) -> TransactionResult:
        target = self._resolve(candidate)
        if not target.exists() or not target.is_file():
            raise TransactionError("Delete transaction requires an existing regular file.")
        snapshot = self._snapshot(run_id, target)
        try:
            target.unlink()
            if target.exists():
                raise TransactionError("Post-delete verification found the file still exists.")
            return TransactionResult("delete", target, None, snapshot, committed=True)
        except Exception as error:
            self._rollback(snapshot)
            raise TransactionError(
                f"Delete transaction failed and rollback was attempted: {error}"
            ) from error

    def rollback(self, snapshot: FileSnapshot) -> None:
        self._rollback(snapshot)

    def _resolve(self, candidate: str) -> Path:
        return resolve_workspace_path(
            workspace_project=self._settings.workspace_project_path,
            candidate=candidate,
            protected_paths=(
                self._settings.workspace_internal_path,
                self._settings.workspace_backups_path,
            ),
        )

    def _snapshot(self, run_id: UUID, target: Path) -> FileSnapshot:
        relative = target.relative_to(self._settings.workspace_project_path.resolve())
        backup_path = self._settings.workspace_backups_path / str(run_id) / relative
        existed = target.exists()
        original_hash = self._hash_file(target) if existed and target.is_file() else "MISSING"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if existed:
            if not target.is_file():
                raise TransactionError("Transactions may mutate regular files only.")
            shutil.copy2(target, backup_path)
        return FileSnapshot(run_id, target, backup_path, original_hash, existed)

    def _rollback(self, snapshot: FileSnapshot) -> None:
        if snapshot.existed:
            snapshot.original_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot.backup_path, snapshot.original_path)
        elif snapshot.original_path.exists():
            snapshot.original_path.unlink()

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(64 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
