"""Independent post-execution verification for registered tool operations."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from local_ai_agent.schemas.contracts import ToolResult, ToolStatus, VerificationResult

VerificationHandler = Callable[[dict[str, Any], ToolResult], Awaitable[VerificationResult]]


class VerificationEngine:
    """Runs the verification function registered for the specific operation at call time."""

    async def verify(
        self,
        *,
        arguments: dict[str, Any],
        result: ToolResult,
        verifier: VerificationHandler | None,
    ) -> ToolResult:
        if not result.success:
            return result.model_copy(
                update={
                    "verified": False,
                    "verification": VerificationResult(
                        verified=False,
                        strategy="execution-failed",
                        message="Verification is not attempted after a failed execution.",
                    ),
                }
            )
        if verifier is None:
            return result.model_copy(
                update={
                    "status": ToolStatus.PARTIAL,
                    "verified": False,
                    "verification": VerificationResult(
                        verified=False,
                        strategy="not-registered",
                        message="No verification handler is registered for this operation.",
                    ),
                }
            )
        try:
            verification = await verifier(arguments, result)
        except Exception as error:  # Verification failure must be returned as data, not hidden.
            verification = VerificationResult(
                verified=False,
                strategy="verification-exception",
                message=f"Verification handler failed: {type(error).__name__}",
            )
        status = result.status if verification.verified else ToolStatus.PARTIAL
        return result.model_copy(
            update={
                "status": status,
                "verified": verification.verified,
                "verification": verification,
            }
        )
