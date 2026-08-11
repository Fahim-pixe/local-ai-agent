"""Pydantic contracts shared across the agent runtime, tools, database, and API."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class AgentState(StrEnum):
    UNDERSTAND = "UNDERSTAND"
    VALIDATE = "VALIDATE"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    OBSERVE = "OBSERVE"
    VERIFY = "VERIFY"
    RECOVER = "RECOVER"
    INPUT_REQUIRED = "INPUT_REQUIRED"
    AUTHORIZATION_REQUIRED = "AUTHORIZATION_REQUIRED"
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ToolStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ConfidenceLevel(StrEnum):
    CONFIRMED = "CONFIRMED"
    LIKELY = "LIKELY"
    POSSIBLE = "POSSIBLE"
    STALE = "STALE"


class MemoryCategory(StrEnum):
    PREFERENCE = "PREFERENCE"
    FACT = "FACT"
    TASK_OUTCOME = "TASK_OUTCOME"
    SEMANTIC = "SEMANTIC"


class MemoryRecord(BaseModel):
    """Durable memory with confidence, provenance, and staleness metadata."""

    model_config = ConfigDict(extra="forbid")

    category: MemoryCategory
    key: str = Field(min_length=1, max_length=512)
    value: str = Field(min_length=1, max_length=50_000)
    confidence: ConfidenceLevel = ConfidenceLevel.POSSIBLE
    source_run_id: UUID | None = None
    expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VerificationResult(BaseModel):
    """Evidence that distinguishes execution success from verified success."""

    model_config = ConfigDict(extra="forbid")

    verified: bool
    strategy: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None


class ToolResult(BaseModel):
    """Authoritative, serializable outcome returned for every registered tool call."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    status: ToolStatus
    success: bool
    data: Any | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    verified: bool = False
    verification: VerificationResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=1)
    action: str
    description: str
    depends_on: list[int] = Field(default_factory=list)
    risk: RiskLevel


class AgentPlan(BaseModel):
    """Structured model proposal validated and owned by the runtime."""

    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, max_length=4_000)
    steps: list[PlanStep] = Field(min_length=1, max_length=30)
    success_criteria: list[str] = Field(min_length=1, max_length=20)
    rollback_strategy: str = Field(min_length=1, max_length=4_000)


class RunBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tool_calls: int = Field(ge=1, le=100)
    max_runtime_seconds: int = Field(ge=1, le=7_200)
    max_shell_executions: int = Field(ge=0, le=100)


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=1, max_length=8_000)
    workspace_id: str = Field(default="default", min_length=1, max_length=128)
    budget: RunBudget | None = None


class AuthorizationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: bool
    reason: str | None = Field(default=None, max_length=2_000)


class UserReplyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8_000)


class AgentRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    objective: str
    workspace_id: str
    state: AgentState = AgentState.UNDERSTAND
    plan: AgentPlan | None = None
    budget: RunBudget
    resume_token: UUID = Field(default_factory=uuid4)
    prompt_hash: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    type: str
    state: AgentState
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any] = Field(default_factory=dict)
