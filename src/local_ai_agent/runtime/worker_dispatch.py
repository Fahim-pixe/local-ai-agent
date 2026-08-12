"""Bounded local SQLite-backed worker dispatch with conservative recovery semantics."""

from __future__ import annotations

import asyncio
import os
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from local_ai_agent.config import Settings
from local_ai_agent.db.repository import DispatchClaim, RunRepository
from local_ai_agent.runtime.continuation import ContinuationError
from local_ai_agent.runtime.lifecycle import RunLifecycleService
from local_ai_agent.runtime.production_prompt import load_production_prompt
from local_ai_agent.schemas.contracts import AgentEvent

if TYPE_CHECKING:
    from collections.abc import Callable

    from local_ai_agent.runtime.secure_run_runtime import SecureRunRuntime


LOCAL_WORKER_CAPABILITIES: tuple[str, ...] = (
    "filesystem.list_directory",
    "filesystem.read_file",
    "filesystem.write_file",
    "filesystem.delete_file",
    "memory.store",
    "shell.execute",
    "python.execute",
)


@dataclass(slots=True)
class LocalDispatchWorker:
    """One local worker process model; it never reclaims a failed uncertain action."""

    settings: Settings
    repository: RunRepository
    lifecycle: RunLifecycleService
    runtime_builder: Callable[..., SecureRunRuntime]
    worker_id: str = field(
        default_factory=lambda: f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex}"
    )
    capabilities: tuple[str, ...] = LOCAL_WORKER_CAPABILITIES
    _last_recovery_sweep: float = field(default=0.0, init=False)

    async def start(self) -> None:
        self.repository.register_worker(
            worker_id=self.worker_id,
            hostname=socket.gethostname(),
            process_id=os.getpid(),
            capabilities=self.capabilities,
        )
        await self._recover_if_due(force=True)

    async def run_once(self) -> bool:
        """Claim and execute one approved action; returns whether work was claimed."""
        if not self.settings.dispatch_enabled:
            return False
        if not self.repository.heartbeat_worker(self.worker_id):
            await self.start()
        await self._recover_if_due()
        claim = self.repository.claim_next_dispatchable_action(
            worker_id=self.worker_id,
            capabilities=self.capabilities,
            lease_seconds=self.settings.worker_lease_seconds,
        )
        if claim is None:
            return False
        await self._execute_claim(claim)
        return True

    async def run(self, stop: asyncio.Event) -> None:
        """Poll with bounded delay until a caller requests graceful drain and shutdown."""
        await self.start()
        try:
            while not stop.is_set():
                claimed = await self.run_once()
                if claimed:
                    continue
                delay = self.settings.dispatch_claim_poll_seconds
                try:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
                except TimeoutError:
                    continue
        finally:
            self.repository.drain_worker(self.worker_id)
            self.repository.stop_worker(self.worker_id)

    async def _execute_claim(self, claim: DispatchClaim) -> None:
        action = claim.action
        if self.lifecycle.cancel_if_requested(action.run_id):
            self.repository.finish_pending_action(action.id, succeeded=False)
            return
        try:
            prompt = load_production_prompt(self.settings)
            runtime = self.runtime_builder(
                settings=self.settings,
                run_id=action.run_id,
                repository=self.repository,
                lifecycle=self.lifecycle,
                worker_id=self.worker_id,
            )
            await runtime.continuation.resume_claimed_action(
                action=action, system_prompt=prompt.content
            )
        except (ContinuationError, RuntimeError) as error:
            self.repository.finish_pending_action(action.id, succeeded=False)
            run = self.repository.get_run(action.run_id)
            if run is not None:
                self.repository.record_event(
                    AgentEvent(
                        run_id=run.id,
                        type="dispatch.action_failed",
                        state=run.state,
                        message="Worker-owned action failed without automatic re-claim.",
                        data={
                            "action_id": str(action.id),
                            "worker_id": self.worker_id,
                            "error_type": type(error).__name__,
                        },
                    )
                )

    async def _recover_if_due(self, *, force: bool = False) -> None:
        now = asyncio.get_running_loop().time()
        if (
            not force
            and now - self._last_recovery_sweep < self.settings.dispatch_recovery_sweep_seconds
        ):
            return
        self.lifecycle.recover_stale_executing_actions(
            lease_seconds=self.settings.worker_lease_seconds,
            reason="WORKER_CRASH_RECOVERY",
        )
        self._last_recovery_sweep = now


@dataclass(slots=True)
class LocalDispatchPool:
    """Bounded collection of independent local worker loops; disabled by configuration by default."""

    settings: Settings
    repository: RunRepository
    lifecycle: RunLifecycleService
    runtime_builder: Callable[..., SecureRunRuntime]
    _stop: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _tasks: list[asyncio.Task[None]] = field(default_factory=list, init=False)

    async def start(self) -> None:
        if not self.settings.dispatch_enabled or self._tasks:
            return
        for _ in range(self.settings.dispatch_max_workers):
            worker = LocalDispatchWorker(
                settings=self.settings,
                repository=self.repository,
                lifecycle=self.lifecycle,
                runtime_builder=self.runtime_builder,
            )
            self._tasks.append(asyncio.create_task(worker.run(self._stop)))

    async def stop(self) -> None:
        self._stop.set()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
