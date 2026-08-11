"""Executable entrypoint for the Local AI Agent API."""

from __future__ import annotations

import uvicorn

from local_ai_agent.api.app import create_app

app = create_app()


def main() -> None:
    """Run the API using settings loaded from the project configuration and environment."""
    uvicorn.run("local_ai_agent.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
