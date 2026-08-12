from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from local_ai_agent.config import Settings
from local_ai_agent.runtime.docker_sandbox import DockerSandboxExecutor, SandboxPolicyError
from local_ai_agent.schemas.contracts import ToolStatus


@pytest.fixture
def sandbox_settings(tmp_path: Path) -> Settings:
    workspace_root = tmp_path / "workspace"
    workspace_project = workspace_root / "project"
    workspace_project.mkdir(parents=True)
    return Settings(
        project_root=tmp_path,
        workspace_root=workspace_root,
        sqlite_path=workspace_root / ".agent" / "agent.db",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen3:8b",
        embedding_model="nomic-embed-text",
        model_context_tokens=32_768,
        agent_api_token=None,
        agent_name="Local AI Agent",
        agent_mission="Execute approved local tasks with evidence-based, policy-enforced tool use.",
        system_prompt_path=tmp_path / "config" / "system_prompt.md",
        default_max_tool_calls=30,
        default_max_runtime_seconds=60,
        default_max_shell_executions=10,
        default_max_retries=3,
        context_reserve_tokens=500,
        recent_tool_results=3,
        recent_conversation_messages=5,
        rag_chunk_tokens=512,
        rag_chunk_overlap_tokens=50,
        rag_top_k=5,
        context_chars_per_token=4,
        filesystem_max_read_bytes=65_536,
        filesystem_max_list_entries=1_000,
        shell_allowlist=(
            "python",
            "python3",
            "pytest",
            "pip",
            "git",
            "npm",
            "node",
            "npx",
            "ls",
            "cat",
            "echo",
            "grep",
            "find",
        ),
        docker_binary="docker",
        docker_sandbox_image="local-ai-agent-sandbox:latest",
        docker_sandbox_network="none",
        docker_sandbox_memory="256m",
        docker_sandbox_cpus=0.5,
        docker_sandbox_pids_limit=64,
        docker_sandbox_user="10001:10001",
        docker_sandbox_tmpfs_size="16m",
        docker_sandbox_read_only_root=True,
    )


def test_build_command_applies_all_mandatory_isolation_controls(sandbox_settings: Settings) -> None:
    executor = DockerSandboxExecutor(sandbox_settings)
    command = executor.build_command(
        command="echo sandbox-ready",
        workspace_path=sandbox_settings.workspace_project_path,
        container_name="verified-sandbox",
    )

    assert command[:5] == ["docker", "run", "--rm", "--name", "verified-sandbox"]
    assert command[command.index("--network") : command.index("--network") + 2] == [
        "--network",
        "none",
    ]
    assert command[command.index("--cap-drop") : command.index("--cap-drop") + 2] == [
        "--cap-drop",
        "ALL",
    ]
    assert "--read-only" in command
    assert command[command.index("--security-opt") : command.index("--security-opt") + 2] == [
        "--security-opt",
        "no-new-privileges:true",
    ]
    assert command[command.index("--pids-limit") : command.index("--pids-limit") + 2] == [
        "--pids-limit",
        "64",
    ]
    assert command[command.index("--memory") : command.index("--memory") + 2] == [
        "--memory",
        "256m",
    ]
    assert command[command.index("--cpus") : command.index("--cpus") + 2] == ["--cpus", "0.5"]
    assert command[command.index("--user") : command.index("--user") + 2] == [
        "--user",
        "10001:10001",
    ]
    assert command[command.index("--tmpfs") : command.index("--tmpfs") + 2] == [
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
    ]
    assert command[command.index("--workdir") : command.index("--workdir") + 2] == [
        "--workdir",
        "/workspace/project",
    ]
    mount = command[command.index("--mount") + 1]
    assert mount == (
        f"type=bind,src={sandbox_settings.workspace_project_path.resolve()},"
        "dst=/workspace/project,readonly=false"
    )
    assert command[-2:] == ["local-ai-agent-sandbox:latest", "echo sandbox-ready"]


def test_build_command_rejects_workspace_outside_project(
    sandbox_settings: Settings, tmp_path: Path
) -> None:
    executor = DockerSandboxExecutor(sandbox_settings)
    outside_workspace = tmp_path / "outside"
    outside_workspace.mkdir()

    with pytest.raises(SandboxPolicyError, match="approved project workspace"):
        executor.build_command(command="echo denied", workspace_path=outside_workspace)


@pytest.mark.parametrize(
    ("unsafe_settings", "expected_message"),
    [
        ({"docker_sandbox_network": "bridge"}, "network must be disabled"),
        ({"docker_sandbox_read_only_root": False}, "root filesystem must be read-only"),
        ({"docker_sandbox_user": "0:0"}, "must not run as root"),
        ({"docker_sandbox_pids_limit": 0}, "PID limit must be positive"),
    ],
)
def test_build_command_rejects_relaxed_policy(
    sandbox_settings: Settings, unsafe_settings: dict[str, object], expected_message: str
) -> None:
    executor = DockerSandboxExecutor(replace(sandbox_settings, **unsafe_settings))

    with pytest.raises(SandboxPolicyError, match=expected_message):
        executor.build_command(
            command="echo denied", workspace_path=sandbox_settings.workspace_project_path
        )


def test_execute_returns_typed_success_result(
    sandbox_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(command, 0, stdout="sandbox output", stderr="")

    monkeypatch.setattr("local_ai_agent.runtime.docker_sandbox.subprocess.run", fake_run)
    result = DockerSandboxExecutor(sandbox_settings).execute(
        tool_name="shell.execute",
        command="echo sandbox-ready",
    )

    assert result.status is ToolStatus.SUCCESS
    assert result.success is True
    assert result.verified is False
    assert result.data == {"exit_code": 0, "stdout": "sandbox output", "stderr": ""}
    assert observed["command"][0:3] == ["docker", "run", "--rm"]
    assert observed["kwargs"] == {
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": 60,
    }


def test_execute_returns_typed_command_failure(
    sandbox_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 17, stdout="", stderr="denied by sandbox")

    monkeypatch.setattr("local_ai_agent.runtime.docker_sandbox.subprocess.run", fake_run)
    result = DockerSandboxExecutor(sandbox_settings).execute(
        tool_name="shell.execute",
        command="exit 17",
    )

    assert result.status is ToolStatus.ERROR
    assert result.success is False
    assert result.error_code == "SANDBOX_COMMAND_FAILED"
    assert result.data == {"exit_code": 17, "stdout": "", "stderr": "denied by sandbox"}


def test_execute_cleans_up_after_timeout(
    sandbox_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[1] == "run":
            raise subprocess.TimeoutExpired(
                command, timeout=1, output=b"partial", stderr=b"timed out"
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("local_ai_agent.runtime.docker_sandbox.subprocess.run", fake_run)
    result = DockerSandboxExecutor(sandbox_settings).execute(
        tool_name="shell.execute",
        command="sleep 5",
        timeout_seconds=1,
    )

    assert result.status is ToolStatus.TIMEOUT
    assert result.error_code == "SANDBOX_TIMEOUT"
    assert result.retryable is True
    assert result.data == {"exit_code": None, "stdout": "partial", "stderr": "timed out"}
    assert calls[1][0:3] == ["docker", "rm", "--force"]


@pytest.mark.docker
@pytest.mark.skipif(
    os.getenv("RUN_DOCKER_INTEGRATION") != "1",
    reason="Set RUN_DOCKER_INTEGRATION=1 to run the real Docker isolation check.",
)
def test_executor_enforces_isolation_in_a_real_container(
    sandbox_settings: Settings, tmp_path: Path
) -> None:
    if shutil.which("docker") is None or shutil.which("sudo") is None:
        pytest.skip("Docker and sudo are required for the real-container check.")

    image_check = subprocess.run(
        ["sudo", "docker", "image", "inspect", sandbox_settings.docker_sandbox_image],
        check=False,
        capture_output=True,
        text=True,
    )
    if image_check.returncode != 0:
        pytest.skip("Build the configured sandbox image before the real-container check.")

    docker_wrapper = tmp_path / "docker-wrapper"
    docker_wrapper.write_text('#!/bin/sh\nexec sudo docker "$@"\n', encoding="utf-8")
    docker_wrapper.chmod(0o700)
    integration_workspace = sandbox_settings.workspace_project_path / ".docker-integration"
    subprocess.run(
        [
            "sudo",
            "install",
            "-d",
            "-m",
            "700",
            "-o",
            "10001",
            "-g",
            "10001",
            str(integration_workspace),
        ],
        check=True,
    )
    executor = DockerSandboxExecutor(replace(sandbox_settings, docker_binary=str(docker_wrapper)))
    probe = (
        "import os, pathlib; "
        "assert os.geteuid() == 10001; "
        "assert not os.path.exists('/sys/class/net/eth0'); "
        "assert os.statvfs('/').f_flag & os.ST_RDONLY; "
        "status = pathlib.Path('/proc/self/status').read_text(); "
        "assert 'NoNewPrivs:\\t1' in status; "
        "assert 'CapEff:\\t0000000000000000' in status; "
        "pathlib.Path('/workspace/project/executor-proof').write_text('isolated'); "
        "print('isolation-ok')"
    )
    try:
        result = executor.execute(
            tool_name="sandbox.probe",
            command=f"python3 -c {shlex.quote(probe)}",
            workspace_path=integration_workspace,
            timeout_seconds=30,
        )

        assert result.success is True
        assert result.data["stdout"].strip() == "isolation-ok"
        proof = subprocess.run(
            ["sudo", "cat", str(integration_workspace / "executor-proof")],
            check=True,
            capture_output=True,
            text=True,
        )
        assert proof.stdout == "isolated"
        container_check = subprocess.run(
            ["sudo", "docker", "container", "inspect", result.metadata["container_name"]],
            check=False,
            capture_output=True,
            text=True,
        )
        assert container_check.returncode != 0
    finally:
        subprocess.run(["sudo", "rm", "-rf", str(integration_workspace)], check=False)
