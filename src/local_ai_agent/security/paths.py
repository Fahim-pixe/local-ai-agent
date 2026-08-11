"""Symlink-aware workspace path validation."""

from __future__ import annotations

from pathlib import Path


class WorkspacePolicyError(PermissionError):
    """Raised when a tool path escapes the permitted project workspace."""


def resolve_workspace_path(
    *, workspace_project: Path, candidate: str | Path, protected_paths: tuple[Path, ...] = ()
) -> Path:
    """Resolve a tool path and enforce that it remains inside the project workspace.

    `Path.resolve()` follows symlinks before the containment check, which prevents the
    common prefix-check bypass where an in-workspace symlink targets a sensitive path.
    """
    root = workspace_project.resolve()
    raw_candidate = Path(candidate)
    target = (
        (root / raw_candidate).resolve()
        if not raw_candidate.is_absolute()
        else raw_candidate.resolve()
    )

    try:
        target.relative_to(root)
    except ValueError as error:
        raise WorkspacePolicyError(f"Path escapes the project workspace: {candidate}") from error

    for protected_path in protected_paths:
        protected = protected_path.resolve()
        if target == protected or protected in target.parents:
            raise WorkspacePolicyError(f"Path targets a protected runtime directory: {candidate}")

    return target
