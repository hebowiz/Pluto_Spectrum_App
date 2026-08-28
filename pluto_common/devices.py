"""Stable Pluto identity selection on top of transient libiio URIs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping


_SERIAL_PATTERN = re.compile(
    r"\bserial\s*[=:]\s*([^\s,;)\]]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PlutoDeviceInfo:
    """One physical Pluto, preferably identified by its hardware serial."""

    uri: str
    description: str
    serial: str | None = None

    @property
    def selector(self) -> str:
        return f"serial:{self.serial}" if self.serial else self.uri

    @property
    def transport(self) -> str:
        return "USB" if self.uri.startswith("usb:") else "Network"

    @property
    def label(self) -> str:
        identity = f"Serial {self.serial}" if self.serial else self.uri
        return f"ADALM-Pluto - {identity} ({self.transport}: {self.uri})"


def serial_from_description(description: str) -> str | None:
    match = _SERIAL_PATTERN.search(str(description))
    return match.group(1) if match else None


def _is_pluto_context(uri: str, description: str) -> bool:
    uri_lower = str(uri).lower()
    description_lower = str(description).lower()
    return uri_lower.startswith("usb:") or (
        uri_lower.startswith("ip:") and "pluto" in description_lower
    )


def _transport_priority(uri: str) -> tuple[int, str]:
    if uri.startswith("usb:"):
        return (0, uri)
    if uri.startswith("ip:"):
        return (1, uri)
    return (2, uri)


def discover_pluto_devices(contexts: Mapping[str, str]) -> tuple[PlutoDeviceInfo, ...]:
    """Return physical Plutos, de-duplicating USB/IP paths by serial.

    A Pluto advertised through both direct USB and ``ip:pluto.local`` appears
    once. Direct USB is retained because it is the preferred application path.
    Contexts without a serial remain selectable by their explicit URI.
    """

    by_identity: dict[str, PlutoDeviceInfo] = {}
    for raw_uri, raw_description in contexts.items():
        uri = str(raw_uri)
        description = str(raw_description)
        if not _is_pluto_context(uri, description):
            continue
        serial = serial_from_description(description)
        identity = f"serial:{serial.lower()}" if serial else f"uri:{uri}"
        candidate = PlutoDeviceInfo(uri=uri, description=description, serial=serial)
        existing = by_identity.get(identity)
        if existing is None or _transport_priority(uri) < _transport_priority(
            existing.uri
        ):
            by_identity[identity] = candidate
    return tuple(
        sorted(
            by_identity.values(),
            key=lambda device: (
                "" if device.serial is None else device.serial.lower(),
                _transport_priority(device.uri),
            ),
        )
    )


def resolve_pluto_uri(
    configured_target: str | None,
    contexts: Mapping[str, str],
) -> str | None:
    """Resolve Auto, an explicit URI, or ``serial:<id>`` to a current URI."""

    target = "" if configured_target is None else str(configured_target).strip()
    devices = discover_pluto_devices(contexts)
    if target.lower().startswith("serial:"):
        serial = target.split(":", 1)[1].strip()
        if not serial:
            raise ValueError("Pluto serial selector must include a serial number")
        for device in devices:
            if device.serial and device.serial.lower() == serial.lower():
                return device.uri
        raise RuntimeError(f"Selected ADALM-Pluto serial {serial} is not connected")
    if target:
        return target
    if devices:
        return min(devices, key=lambda device: _transport_priority(device.uri)).uri
    return None


def pluto_identity(
    configured_target: str | None,
    resolved_uri: str | None,
    contexts: Mapping[str, str],
) -> tuple[str, str | None]:
    """Return a stable lock identity and optional hardware serial."""

    target = str(configured_target or "").strip()
    if target.lower().startswith("serial:"):
        serial = target.split(":", 1)[1].strip()
        if serial:
            return f"serial:{serial.casefold()}", serial
    uri = str(resolved_uri or target).strip()
    if uri:
        serial = serial_from_description(contexts.get(uri, ""))
        if serial:
            return f"serial:{serial.casefold()}", serial
        return f"uri:{uri.casefold()}", None
    return "auto:pluto", None


def short_pluto_identity(target: str | None, *, width: int = 4) -> str:
    """Format a compact serial/URI suffix suitable for a window title."""

    value = str(target or "").strip()
    if not value:
        return "No Device"
    if value.lower().startswith("serial:"):
        value = value.split(":", 1)[1].strip()
    compact = re.sub(r"[^0-9A-Za-z]", "", value) or value
    suffix = compact[-max(1, int(width)) :]
    return f"…{suffix}" if len(compact) > len(suffix) else suffix
