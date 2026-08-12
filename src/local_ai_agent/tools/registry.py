"""Machine-readable tool registration contracts for runtime-controlled execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from local_ai_agent.schemas.contracts import (
    RecoveryClass,
    RiskLevel,
    ToolResult,
    VerificationResult,
)

ToolHandler = Callable[[dict[str, Any]], Awaitable[ToolResult]]
VerificationHandler = Callable[[dict[str, Any], ToolResult], Awaitable[VerificationResult]]
ArgumentsValidator = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    risk: RiskLevel
    handler: ToolHandler
    verification: VerificationHandler | None = None
    arguments_validator: ArgumentsValidator | None = None
    recovery_class: RecoveryClass = RecoveryClass.NEVER_RECLAIM
    recovery_contract_version: int = 1

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Apply the registered runtime validator; tool handlers never validate themselves."""
        return self.arguments_validator(arguments) if self.arguments_validator else arguments


class ToolRegistry:
    """Registry owned by Python, never by model-provided tool definitions."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as error:
            raise KeyError(f"Unknown tool: {name}") from error

    def ollama_tools(self) -> list[dict[str, Any]]:
        """Return the native Ollama tool-call schema, excluding runtime policy metadata."""
        return [
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.input_schema,
                },
            }
            for definition in self._tools.values()
        ]
