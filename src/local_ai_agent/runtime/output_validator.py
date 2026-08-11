"""Validation boundary for structured outputs received from the local model."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from local_ai_agent.schemas.contracts import AgentPlan


class ModelOutputValidationError(ValueError):
    """Raised when a local model response violates the runtime's output contract."""


class OutputValidator:
    """Keeps malformed or ambiguous model output from reaching tools."""

    @staticmethod
    def validate_plan(candidate: Any) -> AgentPlan:
        try:
            return AgentPlan.model_validate(candidate)
        except ValidationError as error:
            raise ModelOutputValidationError(f"Invalid plan payload: {error}") from error

    @staticmethod
    def validate_turn(*, has_tool_calls: bool, final_content: str | None) -> None:
        """Reject turns that simultaneously request tools and claim a final answer."""
        if has_tool_calls and final_content and final_content.strip():
            raise ModelOutputValidationError(
                "Model response cannot include both tool calls and final user-facing content."
            )
