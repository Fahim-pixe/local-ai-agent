"""Authoritative authorization decisions for risk-classified tool calls."""

from __future__ import annotations

from dataclasses import dataclass

from local_ai_agent.schemas.contracts import RiskLevel


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    allowed: bool
    authorization_required: bool
    error_code: str | None
    reason: str


class PermissionGate:
    """Enforces risk-based approval independent of model or tool-handler preference."""

    _REQUIRES_EXPLICIT_APPROVAL = frozenset({RiskLevel.HIGH, RiskLevel.CRITICAL})

    def decide(self, *, risk: RiskLevel, authorization_granted: bool = False) -> PermissionDecision:
        if risk not in self._REQUIRES_EXPLICIT_APPROVAL:
            return PermissionDecision(True, False, None, "Risk level permits automatic execution.")
        if not authorization_granted:
            return PermissionDecision(
                False,
                True,
                "AUTHORIZATION_REQUIRED",
                f"{risk.value} risk requires explicit user authorization before execution.",
            )
        return PermissionDecision(
            True, False, None, "Explicit user authorization was verified by runtime."
        )
