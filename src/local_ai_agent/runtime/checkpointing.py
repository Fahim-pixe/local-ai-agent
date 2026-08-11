"""Repository-backed durable checkpoints for ReAct message state."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from local_ai_agent.db.repository import RunRepository


class RepositoryCheckpointSink:
    """Persist append-only message snapshots owned by one run."""

    def __init__(self, *, run_id: UUID, repository: RunRepository) -> None:
        self._run_id = run_id
        self._repository = repository

    async def checkpoint(self, *, phase: str, messages: list[dict[str, Any]]) -> int:
        checkpoint = self._repository.save_react_checkpoint(
            run_id=self._run_id,
            phase=phase,
            messages=messages,
        )
        return checkpoint.id
