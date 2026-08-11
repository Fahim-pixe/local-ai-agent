"""Runtime budget accounting for tool execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from local_ai_agent.schemas.contracts import RunBudget


class BudgetExceededError(RuntimeError):
    """Raised before execution when an authoritative run limit has been reached."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    tool_calls: int
    shell_executions: int
    elapsed_seconds: float
    remaining_tool_calls: int
    remaining_shell_executions: int
    remaining_runtime_seconds: float


class BudgetManager:
    """Enforces budget limits before work begins; it never trusts model self-reporting."""

    _SHELL_TOOL_NAMES = frozenset({"shell.execute", "python.execute"})

    def __init__(self, budget: RunBudget, clock: Callable[[], float] = monotonic) -> None:
        self._budget = budget
        self._clock = clock
        self._started_at = clock()
        self._tool_calls = 0
        self._shell_executions = 0

    def reserve_tool_call(self, tool_name: str) -> BudgetSnapshot:
        """Atomically check and reserve a tool-call slot before routing a tool."""
        self._raise_if_runtime_exhausted()
        if self._tool_calls >= self._budget.max_tool_calls:
            raise BudgetExceededError(
                "MAX_TOOL_CALLS", "Maximum tool-call budget has been exhausted."
            )
        is_shell_tool = tool_name in self._SHELL_TOOL_NAMES
        if is_shell_tool and self._shell_executions >= self._budget.max_shell_executions:
            raise BudgetExceededError(
                "MAX_SHELL_EXECUTIONS", "Maximum shell-execution budget has been exhausted."
            )
        self._tool_calls += 1
        if is_shell_tool:
            self._shell_executions += 1
        return self.snapshot()

    def snapshot(self) -> BudgetSnapshot:
        elapsed = self._elapsed_seconds()
        return BudgetSnapshot(
            tool_calls=self._tool_calls,
            shell_executions=self._shell_executions,
            elapsed_seconds=elapsed,
            remaining_tool_calls=max(self._budget.max_tool_calls - self._tool_calls, 0),
            remaining_shell_executions=max(
                self._budget.max_shell_executions - self._shell_executions, 0
            ),
            remaining_runtime_seconds=max(self._budget.max_runtime_seconds - elapsed, 0.0),
        )

    def _raise_if_runtime_exhausted(self) -> None:
        if self._elapsed_seconds() >= self._budget.max_runtime_seconds:
            raise BudgetExceededError(
                "MAX_RUNTIME_SECONDS", "Maximum runtime budget has been exhausted."
            )

    def _elapsed_seconds(self) -> float:
        return max(self._clock() - self._started_at, 0.0)
