"""Minimal native-Ollama ReAct orchestration for registered runtime-controlled tools."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from local_ai_agent.config import Settings
from local_ai_agent.runtime.ollama_client import OllamaError
from local_ai_agent.runtime.output_validator import ModelOutputValidationError, OutputValidator
from local_ai_agent.runtime.tool_router import ToolRoutingOutcome
from local_ai_agent.schemas.contracts import AgentState, ToolResult, ToolStatus
from local_ai_agent.tools.registry import ToolRegistry


class NativeToolExecutor(Protocol):
    async def execute(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        authorization_granted: bool = False,
        attempts: int = 1,
        checkpoint_id: int | None = None,
    ) -> ToolRoutingOutcome: ...


class ReActCheckpointSink(Protocol):
    async def checkpoint(self, *, phase: str, messages: list[dict[str, Any]]) -> int: ...


class NativeToolChatClient(Protocol):
    """The subset of Ollama's native chat interface required by the minimal loop."""

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ReActLoopResult:
    state: AgentState
    final_response: str | None
    messages: list[dict[str, Any]]
    tool_outcomes: list[ToolRoutingOutcome]
    error_code: str | None = None
    error_message: str | None = None


class ReActLoop:
    """Run a bounded native-tool conversation without giving the model execution authority."""

    def __init__(
        self,
        *,
        settings: Settings,
        client: NativeToolChatClient,
        registry: ToolRegistry,
        tool_router: NativeToolExecutor,
    ) -> None:
        self._settings = settings
        self._client = client
        self._registry = registry
        self._tool_router = tool_router

    async def run(
        self,
        *,
        objective: str,
        system_prompt: str,
        runtime_context: str | None = None,
        initial_messages: list[dict[str, Any]] | None = None,
        checkpoint_sink: ReActCheckpointSink | None = None,
    ) -> ReActLoopResult:
        """Execute or continue one bounded ReAct run with optional durable checkpoints."""
        if initial_messages is not None:
            messages = list(initial_messages)
            checkpoint_phase = "react-resumed"
        else:
            messages = [{"role": "system", "content": system_prompt}]
            if runtime_context:
                messages.append({"role": "system", "content": runtime_context})
            messages.append({"role": "user", "content": objective})
            checkpoint_phase = "react-started"
        if checkpoint_sink:
            await checkpoint_sink.checkpoint(phase=checkpoint_phase, messages=messages)
        outcomes: list[ToolRoutingOutcome] = []
        max_turns = self._settings.default_max_tool_calls + 1

        for _ in range(max_turns):
            try:
                response = await self._client.chat(
                    model=self._settings.ollama_model,
                    messages=messages,
                    tools=self._registry.ollama_tools(),
                )
                message = self._validated_message(response)
                content = message.get("content")
                tool_calls = message.get("tool_calls") or []
                if tool_calls and isinstance(content, str) and content.strip():
                    # Native-tool calls are authoritative for this turn. Qwen3 may emit
                    # incidental prose with a valid call; do not treat or persist it as
                    # a final answer before the verified tool result is available.
                    message = {**message, "content": ""}
                    content = ""
                OutputValidator.validate_turn(
                    has_tool_calls=bool(tool_calls),
                    final_content=content if isinstance(content, str) else None,
                )
            except (OllamaError, ModelOutputValidationError, ValueError) as error:
                return self._failed_result(
                    messages=messages,
                    outcomes=outcomes,
                    error_code="MODEL_OUTPUT_INVALID"
                    if not isinstance(error, OllamaError)
                    else type(error).__name__,
                    error_message=str(error),
                )

            messages.append(message)
            assistant_checkpoint_id = (
                await checkpoint_sink.checkpoint(phase="assistant-response", messages=messages)
                if checkpoint_sink
                else None
            )
            if not tool_calls:
                if not isinstance(content, str) or not content.strip():
                    return self._failed_result(
                        messages=messages,
                        outcomes=outcomes,
                        error_code="MODEL_OUTPUT_INVALID",
                        error_message="Model response contained neither tool calls nor final content.",
                    )
                if checkpoint_sink:
                    await checkpoint_sink.checkpoint(phase="react-complete", messages=messages)
                return ReActLoopResult(
                    state=AgentState.COMPLETE,
                    final_response=content,
                    messages=messages,
                    tool_outcomes=outcomes,
                )

            for tool_call in tool_calls:
                try:
                    tool_name, arguments = self._parse_tool_call(tool_call)
                except ValueError as error:
                    return self._failed_result(
                        messages=messages,
                        outcomes=outcomes,
                        error_code="MODEL_OUTPUT_INVALID",
                        error_message=str(error),
                    )
                if assistant_checkpoint_id is None:
                    outcome = await self._tool_router.execute(
                        tool_name=tool_name,
                        arguments=arguments,
                    )
                else:
                    outcome = await self._tool_router.execute(
                        tool_name=tool_name,
                        arguments=arguments,
                        checkpoint_id=assistant_checkpoint_id,
                    )
                outcomes.append(outcome)
                if outcome.authorization_required:
                    if checkpoint_sink:
                        await checkpoint_sink.checkpoint(
                            phase="authorization-required", messages=messages
                        )
                    return ReActLoopResult(
                        state=AgentState.AUTHORIZATION_REQUIRED,
                        final_response=None,
                        messages=messages,
                        tool_outcomes=outcomes,
                        error_code=outcome.result.error_code,
                        error_message=outcome.result.error_message,
                    )
                messages.append(self._tool_result_message(tool_name, outcome.result))
                if checkpoint_sink:
                    await checkpoint_sink.checkpoint(phase="tool-result", messages=messages)
                if outcome.result.status is ToolStatus.CANCELLED:
                    return ReActLoopResult(
                        state=AgentState.CANCELLED,
                        final_response=None,
                        messages=messages,
                        tool_outcomes=outcomes,
                        error_code=outcome.result.error_code,
                        error_message=outcome.result.error_message,
                    )

        if checkpoint_sink:
            await checkpoint_sink.checkpoint(phase="react-partial", messages=messages)
        return ReActLoopResult(
            state=AgentState.PARTIAL,
            final_response=None,
            messages=messages,
            tool_outcomes=outcomes,
            error_code="MAX_REACT_TURNS",
            error_message="The bounded ReAct loop reached its maximum number of turns.",
        )

    @staticmethod
    def _validated_message(response: dict[str, Any]) -> dict[str, Any]:
        message = response.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama response did not contain a message object.")
        role = message.get("role")
        if role != "assistant":
            raise ValueError("Ollama response message must have assistant role.")
        if "tool_calls" in message and not isinstance(message["tool_calls"], list):
            raise ValueError("Ollama tool_calls must be a list when present.")
        return message

    @staticmethod
    def _parse_tool_call(tool_call: Any) -> tuple[str, dict[str, Any]]:
        if not isinstance(tool_call, dict):
            raise ValueError("Ollama tool call must be an object.")
        function = tool_call.get("function")
        if not isinstance(function, dict):
            raise ValueError("Ollama tool call must contain a function object.")
        name = function.get("name")
        raw_arguments = function.get("arguments", {})
        if not isinstance(name, str) or not name:
            raise ValueError("Ollama tool call function name is invalid.")
        if isinstance(raw_arguments, str):
            try:
                raw_arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as error:
                raise ValueError("Ollama tool call arguments are not valid JSON.") from error
        if not isinstance(raw_arguments, dict):
            raise ValueError("Ollama tool call arguments must be an object.")
        return name, raw_arguments

    @staticmethod
    def _tool_result_message(tool_name: str, result: ToolResult) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_name": tool_name,
            "content": result.model_dump_json(),
        }

    @staticmethod
    def _failed_result(
        *,
        messages: list[dict[str, Any]],
        outcomes: Sequence[ToolRoutingOutcome],
        error_code: str,
        error_message: str,
    ) -> ReActLoopResult:
        return ReActLoopResult(
            state=AgentState.FAILED,
            final_response=None,
            messages=messages,
            tool_outcomes=list(outcomes),
            error_code=error_code,
            error_message=error_message,
        )
