"""Local Ollama client using native tool-call schemas rather than prompt-embedded tools."""

from __future__ import annotations

from typing import Any

import httpx


class OllamaError(RuntimeError):
    """Base class for local model runtime failures."""


class OllamaUnavailableError(OllamaError):
    """Raised when the Ollama service cannot be reached."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when the configured model is not present locally."""


class OllamaTimeoutError(OllamaError):
    """Raised when an Ollama request exceeds its configured deadline."""


class OllamaGenerationError(OllamaError):
    """Raised when Ollama responds but generation fails."""


class OllamaClient:
    def __init__(self, base_url: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def health_check(self, model: str) -> bool:
        """Confirm the service and configured model are available locally."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
        except httpx.TimeoutException as error:
            raise OllamaTimeoutError("Timed out while checking Ollama.") from error
        except httpx.HTTPError as error:
            raise OllamaUnavailableError(
                "Ollama is unavailable at the configured base URL."
            ) from error

        models = {entry.get("name") for entry in response.json().get("models", [])}
        if model not in models:
            raise OllamaModelNotFoundError(f"Configured Ollama model is not installed: {model}")
        return True

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Call Ollama's native `/api/chat` endpoint with a runtime-owned tool registry."""
        payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if tools:
            payload["tools"] = tools
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.TimeoutException as error:
            raise OllamaTimeoutError("Timed out while generating a model response.") from error
        except httpx.HTTPError as error:
            raise OllamaUnavailableError("Ollama chat request failed.") from error

        data = response.json()
        if data.get("error"):
            raise OllamaGenerationError(str(data["error"]))
        return data
