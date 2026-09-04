"""Pluto hardware adapter for frequency calibration."""

from __future__ import annotations

from collections.abc import Mapping
import re
import time
from typing import Protocol, runtime_checkable

import numpy as np

from pluto_cal.model import FrequencyCalibrationConfig
from pluto_common import PlutoDeviceLease, resolve_pluto_uri


class XOCorrectionRangeError(ValueError):
    """Raised when a requested runtime XO value is outside the device range."""


def parse_xo_correction_available(value: object) -> tuple[int, int]:
    """Parse AD936x range text such as ``[39999984 1 40000016]``."""

    numbers = [int(item) for item in re.findall(r"[-+]?\d+", str(value))]
    if len(numbers) < 2:
        raise RuntimeError(f"Invalid xo_correction_available value: {value!r}")
    lower, upper = numbers[0], numbers[-1]
    if lower > upper:
        lower, upper = upper, lower
    if lower == upper:
        raise RuntimeError("xo_correction_available contains an empty range")
    return lower, upper


@runtime_checkable
class FrequencyBackend(Protocol):
    @property
    def xo_correction_range(self) -> tuple[int, int]: ...

    def get_xo_correction(self) -> int: ...

    def set_xo_correction(self, value: int) -> None: ...

    def capture_iq(self) -> np.ndarray: ...

    def close(self) -> None: ...


class PlutoFrequencyBackend:
    """Own one Pluto RX and expose only operations needed by the optimizer."""

    def __init__(
        self,
        sdr: object,
        lease: PlutoDeviceLease,
        *,
        config: FrequencyCalibrationConfig,
        resolved_uri: str | None,
        contexts: Mapping[str, str] | None = None,
        device_serial: str | None = None,
        network_hosts: Mapping[str, str] | None = None,
    ) -> None:
        self._sdr = sdr
        self._lease = lease
        self.config = config
        self.resolved_uri = resolved_uri
        self.device_serial = device_serial or lease.owner.serial
        self._contexts = dict(contexts or {})
        self._network_hosts = {
            str(serial).casefold(): str(host)
            for serial, host in (network_hosts or {}).items()
        }
        self._closed = False
        self._xo_attribute, available = self._find_xo_attributes(sdr)
        self._xo_correction_range = parse_xo_correction_available(available.value)
        self._configure_receiver()

    @classmethod
    def open(
        cls,
        configured_target: str | None,
        config: FrequencyCalibrationConfig,
    ) -> "PlutoFrequencyBackend":
        import adi
        import iio

        try:
            contexts: Mapping[str, str] = iio.scan_contexts()
        except Exception:
            contexts = {}
        resolved_uri = resolve_pluto_uri(configured_target, contexts)
        if resolved_uri is None and configured_target:
            resolved_uri = str(configured_target)
        actual_serial, _actual_host = cls._probe_context_identity(
            iio, resolved_uri
        )
        network_hosts: dict[str, str] = {}
        for uri in contexts:
            if not str(uri).startswith("ip:"):
                continue
            serial, host = cls._probe_context_identity(iio, str(uri))
            if serial and host:
                network_hosts[serial.casefold()] = host
        lease_target = (
            f"serial:{actual_serial}" if actual_serial else configured_target
        )
        lease = PlutoDeviceLease.acquire(
            lease_target,
            resolved_uri,
            contexts,
            application="Pluto CAL",
            role="RX calibration",
        )
        try:
            sdr = adi.Pluto(uri=resolved_uri) if resolved_uri else adi.Pluto()
            return cls(
                sdr,
                lease,
                config=config,
                resolved_uri=resolved_uri,
                contexts=contexts,
                device_serial=actual_serial,
                network_hosts=network_hosts,
            )
        except Exception:
            lease.release()
            raise

    @staticmethod
    def _probe_context_identity(
        iio_module: object, uri: str | None
    ) -> tuple[str | None, str | None]:
        if not uri:
            return None, None
        try:
            context = iio_module.Context(uri)
            attrs = dict(getattr(context, "attrs", {}))
        except Exception:
            return None, None
        serial = str(attrs.get("hw_serial", "")).strip() or None
        host = str(attrs.get("ip,ip-addr", "")).strip() or None
        if host is None and str(uri).startswith("ip:"):
            host = str(uri).split(":", 1)[1].strip().strip("/") or None
        return serial, host

    @staticmethod
    def _find_xo_attributes(sdr: object) -> tuple[object, object]:
        context = getattr(sdr, "ctx", None) or getattr(sdr, "_ctx", None)
        control = None
        if context is not None:
            finder = getattr(context, "find_device", None)
            if callable(finder):
                control = finder("ad9361-phy")
        if control is None:
            control = getattr(sdr, "_ctrl", None)
        attrs = getattr(control, "attrs", {}) if control is not None else {}
        try:
            return attrs["xo_correction"], attrs["xo_correction_available"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                "Selected Pluto does not expose xo_correction attributes"
            ) from error

    @property
    def xo_correction_range(self) -> tuple[int, int]:
        return self._xo_correction_range

    @property
    def persistence_host(self) -> str | None:
        """Return an IP endpoint proven to represent this selected serial."""

        if self.device_serial:
            return self._network_hosts.get(self.device_serial.casefold())
        return None

    @property
    def persistence_hosts(self) -> tuple[str, ...]:
        """Return ordered SSH candidates for verified persistent storage."""

        candidates: list[str] = []
        if self.device_serial:
            matched = self._network_hosts.get(self.device_serial.casefold())
            if matched:
                candidates.append(matched)
        candidates.extend(("pluto.local", "192.168.2.1"))
        return tuple(dict.fromkeys(candidates))

    def _configure_receiver(self) -> None:
        sdr = self._sdr
        sdr.sample_rate = int(round(self.config.sample_rate_hz))
        sdr.rx_lo = int(round(self.config.rx_lo_hz))
        sdr.rx_rf_bandwidth = int(round(self.config.rx_bandwidth_hz))
        sdr.rx_buffer_size = int(self.config.rx_buffer_size)
        if hasattr(sdr, "rx_enabled_channels"):
            sdr.rx_enabled_channels = [0]
        if hasattr(sdr, "gain_control_mode_chan0"):
            sdr.gain_control_mode_chan0 = "manual"
        if hasattr(sdr, "rx_hardwaregain_chan0"):
            sdr.rx_hardwaregain_chan0 = float(self.config.rx_gain_db)
        self._discard_capture()

    def _discard_capture(self) -> None:
        try:
            self._sdr.rx()
        except Exception:
            # The next real capture reports any persistent transport failure.
            pass

    def get_xo_correction(self) -> int:
        return int(str(self._xo_attribute.value).strip())

    def set_xo_correction(self, value: int) -> None:
        candidate = int(value)
        lower, upper = self.xo_correction_range
        if not lower <= candidate <= upper:
            raise XOCorrectionRangeError(
                f"XO correction {candidate} is outside [{lower}, {upper}]"
            )
        self._xo_attribute.value = str(candidate)
        readback = self.get_xo_correction()
        if readback != candidate:
            raise RuntimeError(
                f"XO correction read-back {readback} does not match {candidate}"
            )
        time.sleep(self.config.settle_time_s)
        self._discard_capture()

    def capture_iq(self) -> np.ndarray:
        values = self._sdr.rx()
        if isinstance(values, (tuple, list)):
            if not values:
                raise RuntimeError("Pluto returned no RX channel data")
            values = values[0]
        result = np.asarray(values, dtype=np.complex128).reshape(-1)
        if result.size < 64:
            raise RuntimeError("Pluto returned an incomplete RX capture")
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            destroy = getattr(self._sdr, "rx_destroy_buffer", None)
            if callable(destroy):
                destroy()
        finally:
            self._lease.release()

    def __enter__(self) -> "PlutoFrequencyBackend":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
