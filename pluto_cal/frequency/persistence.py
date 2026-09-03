"""Non-volatile XO correction persistence, intentionally separate from search."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import re
import socket
import subprocess
from typing import Protocol, runtime_checkable


class PersistenceError(RuntimeError):
    """Raised when an XO correction cannot be saved and verified."""


@runtime_checkable
class XOCorrectionPersistence(Protocol):
    def persist(
        self,
        value: int,
        *,
        before_write: Callable[[], None] | None = None,
    ) -> int: ...


def persistence_host_for_uri(uri: str | None) -> str:
    value = str(uri or "").strip()
    if value.lower().startswith("ip:"):
        host = value.split(":", 1)[1].strip().strip("/")
        if host:
            return host
    return "pluto.local"


def normalize_ssh_host(host: str) -> str:
    """Convert libiio's scoped IPv6 interface name for Python sockets."""

    value = str(host).strip()
    if "%" not in value:
        return value
    address, scope = value.rsplit("%", 1)
    if scope.isdecimal():
        return value
    try:
        scope_id = socket.if_nametoindex(scope)
    except OSError:
        return value
    return f"{address}%{scope_id}"


class SSHXOCorrectionPersistence:
    """Write U-Boot environment only after final verification succeeds."""

    def __init__(
        self,
        host: str,
        *,
        user: str = "root",
        expected_serial: str | None = None,
        password: str | None = "analog",
        timeout_s: float = 8.0,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.host = normalize_ssh_host(host)
        self.user = str(user).strip()
        self.expected_serial = str(expected_serial or "").strip() or None
        self.password = password
        self.timeout_s = float(timeout_s)
        self._runner = runner
        if not self.host or not self.user:
            raise ValueError("SSH persistence requires a host and user")

    def _ssh(self, remote_command: str) -> str:
        if self._runner is None:
            return self._paramiko_ssh(remote_command)
        command: Sequence[str] = (
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={max(1, int(round(self.timeout_s))) }",
            f"{self.user}@{self.host}",
            remote_command,
        )
        try:
            completed = self._runner(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_s + 2.0,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise PersistenceError(f"SSH persistence failed: {error}") from error
        if int(completed.returncode) != 0:
            detail = (completed.stderr or completed.stdout or "SSH command failed").strip()
            raise PersistenceError(detail)
        return str(completed.stdout).strip()

    def _paramiko_ssh(self, remote_command: str) -> str:
        try:
            import paramiko
        except ImportError as error:
            raise PersistenceError(
                "Paramiko is required for Pluto SSH persistence"
            ) from error
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        # Target identity is independently checked against the selected Pluto
        # serial before fw_setenv, so first use can safely avoid an interactive
        # host-key prompt while still refusing a cross-device write.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                self.host,
                username=self.user,
                password=self.password,
                timeout=self.timeout_s,
                auth_timeout=self.timeout_s,
                banner_timeout=self.timeout_s,
                allow_agent=False,
                look_for_keys=False,
            )
            _stdin, stdout, stderr = client.exec_command(
                remote_command, timeout=self.timeout_s
            )
            output = stdout.read().decode("utf-8", errors="replace").strip()
            error_output = stderr.read().decode("utf-8", errors="replace").strip()
            exit_code = int(stdout.channel.recv_exit_status())
        except Exception as error:
            raise PersistenceError(f"SSH persistence failed: {error}") from error
        finally:
            client.close()
        if exit_code != 0:
            raise PersistenceError(error_output or output or "SSH command failed")
        return output

    def read(self) -> int:
        output = self._ssh("fw_printenv -n xo_correction")
        values = re.findall(r"[-+]?\d+", output)
        if not values:
            raise PersistenceError(
                f"Invalid fw_printenv xo_correction output: {output!r}"
            )
        return int(values[-1])

    def verify_target(self) -> None:
        if self.expected_serial is None:
            return
        output = self._ssh("cat /etc/serial")
        remote_serial = output.strip().split("=", 1)[-1].strip()
        if remote_serial.casefold() != self.expected_serial.casefold():
            raise PersistenceError(
                "SSH target serial does not match the selected Pluto "
                f"({remote_serial or 'unknown'} != {self.expected_serial})"
            )

    def persist(
        self,
        value: int,
        *,
        before_write: Callable[[], None] | None = None,
    ) -> int:
        candidate = int(value)
        self.verify_target()
        if before_write is not None:
            before_write()
        self._ssh(f"fw_setenv xo_correction {candidate}")
        readback = self.read()
        if readback != candidate:
            raise PersistenceError(
                f"Persistent XO read-back {readback} does not match {candidate}"
            )
        return readback
