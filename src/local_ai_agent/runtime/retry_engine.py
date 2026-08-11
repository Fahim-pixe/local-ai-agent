"""Error-code and risk-aware retry policy for runtime-controlled recovery."""

from __future__ import annotations

from dataclasses import dataclass

from local_ai_agent.schemas.contracts import RiskLevel, ToolResult


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int
    initial_delay_seconds: float
    multiplier: float


@dataclass(frozen=True, slots=True)
class RetryDecision:
    should_retry: bool
    delay_seconds: float | None
    reason: str


class RetryEngine:
    """Computes retry decisions; the caller schedules retries and persists the evidence."""

    _POLICIES = {
        "TRANSIENT": RetryPolicy(max_attempts=3, initial_delay_seconds=1.0, multiplier=2.0),
        "TIMEOUT": RetryPolicy(max_attempts=2, initial_delay_seconds=2.0, multiplier=2.0),
        "TOOL_FAILURE": RetryPolicy(max_attempts=2, initial_delay_seconds=1.0, multiplier=2.0),
        "SANDBOX_LAUNCH_FAILED": RetryPolicy(
            max_attempts=2, initial_delay_seconds=1.0, multiplier=2.0
        ),
    }
    _RISK_MAX_ATTEMPTS = {
        RiskLevel.LOW: 3,
        RiskLevel.MEDIUM: 3,
        RiskLevel.HIGH: 1,
        RiskLevel.CRITICAL: 0,
    }

    def decide(self, *, result: ToolResult, risk: RiskLevel, attempts: int) -> RetryDecision:
        """Return an evidence-based retry decision after a completed tool result."""
        if result.success:
            return RetryDecision(False, None, "Successful tool calls are not retried.")
        if not result.retryable:
            return RetryDecision(False, None, "Runtime marked the failure as non-retryable.")
        if result.error_code is None or result.error_code not in self._POLICIES:
            return RetryDecision(False, None, "No retry policy exists for this error code.")
        policy = self._POLICIES[result.error_code]
        maximum_attempts = min(policy.max_attempts, self._RISK_MAX_ATTEMPTS[risk])
        if attempts >= maximum_attempts:
            return RetryDecision(False, None, "The retry-attempt limit has been reached.")
        delay = policy.initial_delay_seconds * (policy.multiplier**attempts)
        return RetryDecision(True, delay, "Retry allowed by the error-code and risk policy.")
