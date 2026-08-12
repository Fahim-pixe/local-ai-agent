"""Standalone process entry point for one local SQLite-backed dispatch worker."""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Sequence

from local_ai_agent.config import ensure_workspace, load_settings
from local_ai_agent.db.repository import RunRepository
from local_ai_agent.runtime.lifecycle import RunLifecycleService
from local_ai_agent.runtime.secure_run_runtime import build_secure_run_runtime
from local_ai_agent.runtime.worker_dispatch import LocalDispatchWorker


async def run_worker() -> None:
    """Run one worker until SIGINT or SIGTERM requests a graceful drain."""
    settings = load_settings()
    if not settings.dispatch_enabled:
        raise RuntimeError("Dispatch workers require DISPATCH_ENABLED=true.")
    ensure_workspace(settings)
    repository = RunRepository(settings.sqlite_path)
    repository.initialize()
    lifecycle = RunLifecycleService(repository)
    worker = LocalDispatchWorker(
        settings=settings,
        repository=repository,
        lifecycle=lifecycle,
        runtime_builder=build_secure_run_runtime,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_value in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signal_value, stop.set)
        except NotImplementedError:
            signal.signal(signal_value, lambda *_: stop.set())
    await worker.run(stop)


def main(argv: Sequence[str] | None = None) -> int:
    """Console-compatible worker main function; arguments are intentionally not accepted yet."""
    del argv
    asyncio.run(run_worker())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
