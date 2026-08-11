"""Verified runtime tool for storing durable memory records."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_ai_agent.memory.repository import MemoryRepository
from local_ai_agent.schemas.contracts import (
    ConfidenceLevel,
    MemoryCategory,
    MemoryRecord,
    RiskLevel,
    ToolResult,
    ToolStatus,
    VerificationResult,
)
from local_ai_agent.security.output_scrubber import SecretScrubber
from local_ai_agent.tools.registry import ToolDefinition


class MemoryStoreArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: MemoryCategory
    key: str = Field(min_length=1, max_length=512)
    value: str = Field(min_length=1, max_length=50_000)
    confidence: ConfidenceLevel = ConfidenceLevel.POSSIBLE
    expires_in_seconds: int | None = Field(default=None, ge=1, le=31_536_000)


def build_memory_tools(
    *, repository: MemoryRepository, source_run_id: UUID
) -> list[ToolDefinition]:
    service = MemoryTools(repository=repository, source_run_id=source_run_id)
    return [
        ToolDefinition(
            name="memory.store",
            description="Store a structured, confidence-labeled durable memory record.",
            input_schema=MemoryStoreArguments.model_json_schema(),
            risk=RiskLevel.MEDIUM,
            handler=service.store,
            verification=service.verify_store,
            arguments_validator=validate_memory_store_arguments,
        )
    ]


def validate_memory_store_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        return MemoryStoreArguments.model_validate(arguments).model_dump()
    except ValidationError as error:
        raise ValueError("Invalid memory.store arguments.") from error


class MemoryTools:
    """Memory store handler with retrieval verification and secret redaction."""

    def __init__(self, *, repository: MemoryRepository, source_run_id: UUID) -> None:
        self._repository = repository
        self._source_run_id = source_run_id
        self._scrubber = SecretScrubber()

    async def store(self, arguments: dict[str, Any]) -> ToolResult:
        expires_at = (
            datetime.now(UTC) + timedelta(seconds=arguments["expires_in_seconds"])
            if arguments.get("expires_in_seconds")
            else None
        )
        value = self._scrubber.scrub_text(arguments["value"])
        memory = self._repository.upsert(
            MemoryRecord(
                category=arguments["category"],
                key=arguments["key"],
                value=value,
                confidence=arguments["confidence"],
                source_run_id=self._source_run_id,
                expires_at=expires_at,
            )
        )
        return ToolResult(
            tool_name="memory.store",
            status=ToolStatus.SUCCESS,
            success=True,
            data={
                "category": memory.category.value,
                "key": memory.key,
                "confidence": memory.confidence.value,
                "expires_at": memory.expires_at.isoformat() if memory.expires_at else None,
            },
        )

    async def verify_store(self, arguments: dict[str, Any], _: ToolResult) -> VerificationResult:
        stored = self._repository.get(category=arguments["category"], key=arguments["key"])
        verified = stored is not None and stored.value == self._scrubber.scrub_text(
            arguments["value"]
        )
        return VerificationResult(
            verified=verified,
            strategy="memory-retrieve",
            evidence={"key": arguments["key"], "found": stored is not None},
            message=None
            if verified
            else "Stored memory could not be retrieved with matching content.",
        )
