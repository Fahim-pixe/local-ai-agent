from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from local_ai_agent.config import Settings


class ProductionPromptError(RuntimeError):
    """Raised when the configured production system prompt is unavailable or unsafe to use."""


@dataclass(frozen=True, slots=True)
class ProductionSystemPrompt:
    """Versioned prompt content and its byte-level SHA-256 provenance."""

    content: str
    sha256: str
    source_path: Path


def load_production_prompt(settings: Settings) -> ProductionSystemPrompt:
    """Load the configured versioned system prompt and calculate its authoritative hash."""
    source_path = settings.system_prompt_path
    try:
        raw_content = source_path.read_bytes()
    except OSError as error:
        raise ProductionPromptError(
            f"Configured system prompt is unavailable: {source_path}"
        ) from error
    if not raw_content.strip():
        raise ProductionPromptError("Configured system prompt must not be empty.")
    try:
        content = raw_content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProductionPromptError("Configured system prompt must be UTF-8 text.") from error
    return ProductionSystemPrompt(
        content=content,
        sha256=hashlib.sha256(raw_content).hexdigest(),
        source_path=source_path,
    )
