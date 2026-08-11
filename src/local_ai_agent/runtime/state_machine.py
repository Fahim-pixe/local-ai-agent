"""Authoritative agent-state transitions enforced by the Python runtime."""

from __future__ import annotations

from dataclasses import dataclass

from local_ai_agent.schemas.contracts import AgentState


class InvalidStateTransition(ValueError):
    """Raised when a caller attempts an illegal transition."""


VALID_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.UNDERSTAND: frozenset(
        {AgentState.VALIDATE, AgentState.FAILED, AgentState.CANCELLED}
    ),
    AgentState.VALIDATE: frozenset(
        {AgentState.PLAN, AgentState.INPUT_REQUIRED, AgentState.FAILED, AgentState.CANCELLED}
    ),
    AgentState.PLAN: frozenset({AgentState.EXECUTE, AgentState.FAILED, AgentState.CANCELLED}),
    AgentState.EXECUTE: frozenset(
        {
            AgentState.OBSERVE,
            AgentState.AUTHORIZATION_REQUIRED,
            AgentState.RECOVER,
            AgentState.FAILED,
            AgentState.CANCELLED,
        }
    ),
    AgentState.OBSERVE: frozenset(
        {
            AgentState.EXECUTE,
            AgentState.VERIFY,
            AgentState.RECOVER,
            AgentState.INPUT_REQUIRED,
            AgentState.FAILED,
            AgentState.CANCELLED,
        }
    ),
    AgentState.VERIFY: frozenset(
        {
            AgentState.EXECUTE,
            AgentState.COMPLETE,
            AgentState.PARTIAL,
            AgentState.RECOVER,
            AgentState.FAILED,
        }
    ),
    AgentState.RECOVER: frozenset(
        {
            AgentState.EXECUTE,
            AgentState.INPUT_REQUIRED,
            AgentState.PARTIAL,
            AgentState.FAILED,
            AgentState.CANCELLED,
        }
    ),
    AgentState.INPUT_REQUIRED: frozenset(
        {AgentState.VALIDATE, AgentState.CANCELLED, AgentState.FAILED}
    ),
    AgentState.AUTHORIZATION_REQUIRED: frozenset(
        {AgentState.EXECUTE, AgentState.CANCELLED, AgentState.FAILED}
    ),
    AgentState.COMPLETE: frozenset(),
    AgentState.PARTIAL: frozenset(),
    AgentState.FAILED: frozenset(),
    AgentState.CANCELLED: frozenset(),
}


@dataclass(slots=True)
class StateMachine:
    """Holds one run's state and permits only documented valid transitions."""

    state: AgentState = AgentState.UNDERSTAND

    def transition_to(self, target: AgentState) -> AgentState:
        if target not in VALID_TRANSITIONS[self.state]:
            raise InvalidStateTransition(f"Cannot transition from {self.state} to {target}.")
        self.state = target
        return self.state

    @property
    def is_terminal(self) -> bool:
        return not VALID_TRANSITIONS[self.state]
