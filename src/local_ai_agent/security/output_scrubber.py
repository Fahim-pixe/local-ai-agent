"""Runtime output redaction for secrets that must never leave the enforcement layer."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


class SecretScrubber:
    """Replace configured secret values in text and nested runtime result payloads."""

    _SECRET_NAME_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "PRIVATE_KEY")

    def __init__(self, secrets: Mapping[str, str] | None = None) -> None:
        source = secrets or self._environment_secrets()
        self._values = tuple(
            sorted({value for value in source.values() if len(value) >= 4}, key=len, reverse=True)
        )

    def scrub_text(self, value: str | None) -> str | None:
        if value is None:
            return None
        scrubbed = value
        for secret in self._values:
            scrubbed = scrubbed.replace(secret, "[REDACTED]")
        return scrubbed

    def scrub(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.scrub_text(value)
        if isinstance(value, Mapping):
            return {key: self.scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.scrub(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.scrub(item) for item in value)
        return value

    @classmethod
    def _environment_secrets(cls) -> dict[str, str]:
        return {
            name: value
            for name, value in os.environ.items()
            if any(marker in name.upper() for marker in cls._SECRET_NAME_MARKERS)
        }
