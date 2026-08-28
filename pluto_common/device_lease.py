"""Cross-process exclusive leases for physical ADALM-Pluto devices."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import BinaryIO, Mapping

from pluto_common.devices import pluto_identity


@dataclass(frozen=True)
class PlutoLeaseOwner:
    application: str
    role: str
    pid: int
    identity: str
    serial: str | None = None


class PlutoDeviceBusyError(RuntimeError):
    """Raised when another process already owns the selected Pluto."""

    def __init__(self, owner: PlutoLeaseOwner | None, identity: str) -> None:
        self.owner = owner
        self.identity = identity
        detail = (
            "another Pluto application"
            if owner is None
            else f"{owner.application} ({owner.role}, PID {owner.pid})"
        )
        serial = owner.serial if owner and owner.serial else identity
        super().__init__(f"ADALM-Pluto {serial} is already in use by {detail}")


def _lease_root() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    root = Path(base) if base else Path(tempfile.gettempdir())
    return root / "PlutoSpectrumApp" / "device-leases"


def _ensure_lease_root() -> Path:
    preferred = _lease_root()
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError:
        # Sandboxed or centrally managed PCs can expose LOCALAPPDATA while
        # denying writes there.  A per-user temporary directory still gives
        # all concurrently running Pluto applications a common lock location.
        fallback = Path(tempfile.gettempdir()) / "PlutoSpectrumApp-device-leases"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _lock_file(handle: BinaryIO) -> bool:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock_file(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _read_owner(path: Path) -> PlutoLeaseOwner | None:
    try:
        payload = json.loads(path.read_bytes()[1:].decode("utf-8").strip())
        return PlutoLeaseOwner(**payload)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return None


def _read_owner_handle(handle: BinaryIO) -> PlutoLeaseOwner | None:
    try:
        handle.seek(1)
        payload = json.loads(handle.read().decode("utf-8").strip())
        return PlutoLeaseOwner(**payload)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return None


class PlutoDeviceLease:
    """An OS-released exclusive lease held for one open libiio context."""

    def __init__(self, handle: BinaryIO, path: Path, owner: PlutoLeaseOwner) -> None:
        self._handle: BinaryIO | None = handle
        self.path = path
        self.owner = owner

    @classmethod
    def acquire(
        cls,
        configured_target: str | None,
        resolved_uri: str | None,
        contexts: Mapping[str, str],
        *,
        application: str,
        role: str,
    ) -> "PlutoDeviceLease":
        identity, serial = pluto_identity(configured_target, resolved_uri, contexts)
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        root = _ensure_lease_root()
        path = root / f"{digest}.lock"
        handle = path.open("a+b")
        if path.stat().st_size == 0:
            handle.write(b"\0")
            handle.flush()
        if not _lock_file(handle):
            owner = _read_owner_handle(handle)
            handle.close()
            raise PlutoDeviceBusyError(owner or _read_owner(path), identity)
        owner = PlutoLeaseOwner(
            application=str(application),
            role=str(role),
            pid=os.getpid(),
            identity=identity,
            serial=serial,
        )
        handle.seek(1)
        handle.truncate()
        handle.write(json.dumps(asdict(owner), ensure_ascii=True).encode("utf-8"))
        handle.flush()
        return cls(handle, path, owner)

    def release(self) -> None:
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        _unlock_file(handle)
        handle.close()

    def __enter__(self) -> "PlutoDeviceLease":
        return self

    def __exit__(self, *_args) -> None:
        self.release()

    def __del__(self) -> None:
        self.release()
