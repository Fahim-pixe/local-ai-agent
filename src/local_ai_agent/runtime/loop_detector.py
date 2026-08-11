"""Deterministic detection of repeated identical tool-call requests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter


class LoopDetector:
    """Flags a repeated `(tool_name, normalized_arguments)` tuple within one run."""

    def __init__(self, repeat_threshold: int = 2) -> None:
        self.repeat_threshold = repeat_threshold
        self._counts: Counter[tuple[str, str]] = Counter()

    @staticmethod
    def arguments_hash(arguments: dict[str, object]) -> str:
        normalized = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def record_and_check(self, tool_name: str, arguments: dict[str, object]) -> bool:
        key = (tool_name, self.arguments_hash(arguments))
        self._counts[key] += 1
        return self._counts[key] >= self.repeat_threshold
