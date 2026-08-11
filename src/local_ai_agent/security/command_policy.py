"""Conservative allowlist and denylist checks for sandboxed command requests."""

from __future__ import annotations

import shlex
from dataclasses import dataclass


class CommandPolicyError(PermissionError):
    """Raised when a command does not satisfy the runtime command policy."""


@dataclass(frozen=True, slots=True)
class CommandPolicy:
    allowlist: frozenset[str]

    _FORBIDDEN_TOKENS = frozenset(
        {
            "&&",
            "||",
            ";",
            "|",
            ">",
            ">>",
            "<",
            "<<",
            "`",
            "$()",
            "rm",
            "sudo",
            "su",
            "shutdown",
            "reboot",
            "mount",
            "umount",
            "curl",
            "wget",
            "nc",
            "netcat",
        }
    )

    @classmethod
    def from_allowlist(cls, commands: tuple[str, ...]) -> CommandPolicy:
        return cls(frozenset(commands))

    def validate(self, command: str) -> str:
        if not command.strip():
            raise CommandPolicyError("Sandbox command must not be empty.")
        if any(marker in command for marker in ("$(`", "${", "\n", "\r")):
            raise CommandPolicyError(
                "Command contains forbidden shell interpolation or line control."
            )
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError as error:
            raise CommandPolicyError("Command could not be parsed safely.") from error
        if not tokens:
            raise CommandPolicyError("Sandbox command must not be empty.")
        if tokens[0] not in self.allowlist:
            raise CommandPolicyError(f"Command is not allowlisted: {tokens[0]}")
        if any(token in self._FORBIDDEN_TOKENS for token in tokens):
            raise CommandPolicyError("Command contains a forbidden token.")
        return command
