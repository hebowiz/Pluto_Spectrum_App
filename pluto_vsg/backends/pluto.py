"""Finite ADALM-Pluto transmit backend using a non-cyclic DMA buffer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
import re
import threading
import time

import adi
import iio
import numpy as np

from pluto_common import (
    PlutoDeviceLease,
    discover_pluto_devices,
    resolve_pluto_uri,
)
from pluto_vsg.backends.base import BackendCapabilities
from pluto_vsg.engine import GenerationResult


_PLUTO_DAC_FULL_SCALE = 2**15 - 1
_PLUTO_MUTED_GAIN_DB = -89.75
_DEFAULT_STARTUP_DELAY_S = 0.010
_DEFAULT_DMA_PREROLL_S = 0.010
_DEFAULT_COMPLETION_MARGIN_S = 0.100
_MAX_BURST_COUNT = 1000
_NONCYCLIC_SUFFIX_GUARD_S = 0.002
_SERIAL_PATTERN = re.compile(r"\bserial\s*[=:]\s*([^\s,;]+)", re.IGNORECASE)

# Provisional conducted-power calibration measured at 2440 MHz with a
# constant-envelope FSK packet on 2026-08-27.  Keep this table isolated from
# the transmit mechanics so it can later be replaced by per-device and
# frequency-dependent calibration data without changing the UI contract.
_PLUTO_LEVEL_CALIBRATION_FREQUENCY_HZ = 2_440_000_000.0
_PLUTO_LEVEL_GAIN_POINTS_DB = np.asarray((-20.0, -10.0, -5.0, 0.0))
_PLUTO_LEVEL_OUTPUT_POINTS_DBM = np.asarray((-19.0, -9.4, -4.8, -0.2))


def _interpolate_with_linear_extrapolation(
    value: float,
    x_points: np.ndarray,
    y_points: np.ndarray,
) -> float:
    """Interpolate a monotonic calibration table and extend its end slopes."""

    x = float(value)
    if x < float(x_points[0]):
        start = 0
    elif x > float(x_points[-1]):
        start = x_points.size - 2
    else:
        return float(np.interp(x, x_points, y_points))
    slope = float(
        (y_points[start + 1] - y_points[start])
        / (x_points[start + 1] - x_points[start])
    )
    return float(y_points[start] + (x - x_points[start]) * slope)


def estimate_pluto_output_power_dbm(
    hardware_gain_db: float,
    digital_backoff_db: float,
    center_frequency_hz: float = _PLUTO_LEVEL_CALIBRATION_FREQUENCY_HZ,
) -> float:
    """Estimate active FSK packet power using the provisional 2440 MHz data."""

    del center_frequency_hz  # Reserved for the future frequency correction.
    full_scale_power_dbm = _interpolate_with_linear_extrapolation(
        hardware_gain_db,
        _PLUTO_LEVEL_GAIN_POINTS_DB,
        _PLUTO_LEVEL_OUTPUT_POINTS_DBM,
    )
    return full_scale_power_dbm + float(digital_backoff_db)


def pluto_hardware_gain_for_output_power_dbm(
    output_power_dbm: float,
    digital_backoff_db: float,
    center_frequency_hz: float = _PLUTO_LEVEL_CALIBRATION_FREQUENCY_HZ,
) -> float:
    """Return Tx Gain needed for a provisional conducted RF level target."""

    del center_frequency_hz  # Reserved for the future frequency correction.
    full_scale_target_dbm = float(output_power_dbm) - float(digital_backoff_db)
    return _interpolate_with_linear_extrapolation(
        full_scale_target_dbm,
        _PLUTO_LEVEL_OUTPUT_POINTS_DBM,
        _PLUTO_LEVEL_GAIN_POINTS_DB,
    )


def pluto_output_power_range_dbm(
    digital_backoff_db: float,
    center_frequency_hz: float = _PLUTO_LEVEL_CALIBRATION_FREQUENCY_HZ,
) -> tuple[float, float]:
    """Return the provisional RF-level range reachable by Pluto Tx Gain."""

    return (
        estimate_pluto_output_power_dbm(
            _PLUTO_MUTED_GAIN_DB,
            digital_backoff_db,
            center_frequency_hz,
        ),
        estimate_pluto_output_power_dbm(
            0.0,
            digital_backoff_db,
            center_frequency_hz,
        ),
    )


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "<not installed>"


def _libiio_version() -> str:
    value = getattr(iio, "version", None)
    if isinstance(value, tuple):
        return ".".join(str(part) for part in value[:2])
    return str(value or "<unavailable>")


@dataclass(frozen=True)
class PlutoTransmitSettings:
    center_frequency_hz: float
    sample_rate_hz: float
    rf_bandwidth_hz: float
    hardware_gain_db: float = -30.0
    digital_backoff_db: float = 0.0
    connection_uri: str | None = None
    lead_in_guard_s: float = _DEFAULT_STARTUP_DELAY_S
    dma_preroll_s: float = _DEFAULT_DMA_PREROLL_S
    stop_guard_s: float = _DEFAULT_COMPLETION_MARGIN_S
    burst_count: int = 1
    output_power_dbm: float | None = None

    @property
    def resolved_hardware_gain_db(self) -> float:
        """Tx Gain used by hardware, including provisional dBm conversion."""

        if self.output_power_dbm is None:
            return float(self.hardware_gain_db)
        return pluto_hardware_gain_for_output_power_dbm(
            self.output_power_dbm,
            self.digital_backoff_db,
            self.center_frequency_hz,
        )


class PlutoOutputBackend:
    """Transmit a finite schedule once through a non-cyclic DMA buffer.

    The requested packet count is already represented in ``GenerationResult``.
    A short zero prefix absorbs the DAC source transition and a trailing zero
    guard leaves the converter at zero after the one-shot DMA transfer. No
    host-timed cleanup is used to determine the packet count.
    """

    def __init__(self, settings: PlutoTransmitSettings) -> None:
        self.settings = settings
        self._validate_settings(settings)
        self._buffer: np.ndarray | None = None
        self._superframe: np.ndarray | None = None
        self._sample_count = 0
        self._frame_sample_count = 0
        self._waveform_peak = 0.0
        self._dac_peak_code = 0.0
        self._preload_guard_s = 0.0
        self._stop_event = threading.Event()
        self._sdr = None
        self._events: list[tuple[str, float]] = []
        self._observations: list[dict[str, object]] = []
        self._opened_uri: str | None = None
        self._firmware: tuple[int, int] | None = None
        self._calibration_mode: str | None = None
        self._calibration_mode_raw: str | None = None
        self._state = "MUTED"
        self._trace_started_utc = datetime.now(timezone.utc).isoformat()

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_name="ADALM-Pluto Non-Cyclic TX",
            supports_rf_output=True,
            maximum_sample_rate_hz=61_440_000.0,
        )

    @property
    def event_log(self) -> tuple[tuple[str, float], ...]:
        """Monotonic timestamps for correlating API activity with RF captures."""

        return tuple(self._events)

    def _record_event(self, name: str) -> None:
        self._events.append((name, time.monotonic()))

    def _set_numeric_if_changed(
        self,
        owner,
        name: str,
        target: int,
        *,
        tolerance: float,
        event_prefix: str,
    ) -> bool:
        """Avoid needless AD9361 calibrations caused by equivalent rewrites."""

        try:
            current = float(getattr(owner, name))
        except Exception:
            current = float("nan")
        if np.isfinite(current) and np.isclose(
            current, float(target), rtol=0.0, atol=float(tolerance)
        ):
            self._record_event(f"{event_prefix}_unchanged")
            return False
        setattr(owner, name, int(target))
        self._record_event(f"{event_prefix}_configured")
        return True

    @staticmethod
    def _safe_read(owner, name: str):
        try:
            value = getattr(owner, name)
        except Exception as error:
            return f"<read failed: {error}>"
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _observe_hardware(self, stage: str, sdr=None) -> None:
        """Capture non-fatal read-backs for correlating host work with RF output."""

        observation: dict[str, object] = {
            "stage": stage,
            "monotonic_s": time.monotonic(),
        }
        if sdr is not None:
            observation["tx_gain_db"] = self._safe_read(
                sdr, "tx_hardwaregain_chan0"
            )
            observation["tx_lo_hz"] = self._safe_read(sdr, "tx_lo")
            observation["sample_rate_hz"] = self._safe_read(sdr, "sample_rate")
            observation["tx_rf_bandwidth_hz"] = self._safe_read(
                sdr, "tx_rf_bandwidth"
            )
            observation["tx_lo_powerdown"] = self._read_tx_lo_powerdown(sdr)
        self._observations.append(observation)

    def diagnostic_report(self) -> dict[str, object]:
        """Return a JSON-safe trace of one finite-transmit attempt."""

        origin = self._events[0][1] if self._events else time.monotonic()
        previous = origin
        events = []
        for name, timestamp in self._events:
            events.append(
                {
                    "name": name,
                    "relative_ms": round((timestamp - origin) * 1e3, 3),
                    "delta_ms": round((timestamp - previous) * 1e3, 3),
                }
            )
            previous = timestamp
        observations = []
        for item in self._observations:
            converted = dict(item)
            timestamp = float(converted.pop("monotonic_s"))
            converted["relative_ms"] = round((timestamp - origin) * 1e3, 3)
            observations.append(converted)
        return {
            "started_utc": self._trace_started_utc,
            "connection_uri": self._opened_uri,
            "libiio_version": _libiio_version(),
            "pyadi_iio_version": _package_version("pyadi-iio"),
            "tx_dma_mode": "non-cyclic finite buffer",
            "tdd_policy": "not accessed; power-cycle after any prior TDD experiment",
            "state": self._state,
            "calibration_mode": self._calibration_mode,
            "calibration_mode_raw": self._calibration_mode_raw,
            "firmware": None
            if self._firmware is None
            else f"v{self._firmware[0]}.{self._firmware[1]}",
            "settings": asdict(self.settings),
            "resolved_hardware_gain_db": self.settings.resolved_hardware_gain_db,
            "sample_count": self._sample_count,
            "frame_sample_count": self._frame_sample_count,
            "waveform_peak": self._waveform_peak,
            "dac_full_scale": _PLUTO_DAC_FULL_SCALE,
            "dac_peak_code": self._dac_peak_code,
            "superframe_sample_count": 0
            if self._superframe is None
            else int(self._superframe.size),
            "preload_guard_ms": self._preload_guard_s * 1e3,
            "waveform_duration_ms": self.waveform_duration_s * 1e3,
            "frame_duration_ms": self.frame_duration_s * 1e3,
            "dma_buffer_duration_ms": self.dma_buffer_duration_s * 1e3,
            "events": events,
            "hardware_observations": observations,
        }

    @staticmethod
    def discover() -> dict[str, str]:
        try:
            contexts = iio.scan_contexts()
        except Exception:
            return {}
        return {
            str(uri): str(description)
            for uri, description in contexts.items()
            if uri.startswith("usb:")
            or (uri.startswith("ip:") and "pluto" in description.lower())
        }

    @staticmethod
    def discover_devices():
        return discover_pluto_devices(PlutoOutputBackend.discover())

    @staticmethod
    def _validate_settings(settings: PlutoTransmitSettings) -> None:
        if not 520_833.0 <= settings.sample_rate_hz <= 61_440_000.0:
            raise ValueError("Pluto sample rate must be between 520.833 kS/s and 61.44 MS/s")
        if settings.center_frequency_hz <= 0.0:
            raise ValueError("Pluto TX center frequency must be positive")
        if not 200_000.0 <= settings.rf_bandwidth_hz <= 56_000_000.0:
            raise ValueError("Pluto TX RF bandwidth must be between 200 kHz and 56 MHz")
        if settings.output_power_dbm is not None and not np.isfinite(
            settings.output_power_dbm
        ):
            raise ValueError("Pluto RF output level must be finite")
        if not -60.0 <= settings.digital_backoff_db <= 0.0:
            raise ValueError("Pluto digital backoff must be between -60 dB and 0 dB")
        resolved_gain_db = settings.resolved_hardware_gain_db
        if not -89.75 <= resolved_gain_db <= 0.0:
            minimum_dbm, maximum_dbm = pluto_output_power_range_dbm(
                settings.digital_backoff_db,
                settings.center_frequency_hz,
            )
            if settings.output_power_dbm is not None:
                raise ValueError(
                    "Pluto RF output level must be between "
                    f"{minimum_dbm:.2f} dBm and {maximum_dbm:.2f} dBm "
                    f"at {settings.digital_backoff_db:+.0f} dB backoff"
                )
            raise ValueError("Pluto TX attenuation must be between -89.75 dB and 0 dB")
        if settings.lead_in_guard_s < 0.0:
            raise ValueError("Pluto TX lead guard must not be negative")
        if settings.dma_preroll_s < 0.0:
            raise ValueError("Pluto TX DMA pre-roll must not be negative")
        if settings.stop_guard_s < 0.010:
            raise ValueError("Pluto TX completion guard must be at least 10 ms")
        if not 1 <= int(settings.burst_count) <= _MAX_BURST_COUNT:
            raise ValueError("Pluto packet count must be between 1 and 1000")
        if int(settings.burst_count) != settings.burst_count:
            raise ValueError("Pluto packet count must be an integer")

    @staticmethod
    def _resolve_connection_uri(configured_uri: str | None) -> str | None:
        return resolve_pluto_uri(configured_uri, PlutoOutputBackend.discover())

    @classmethod
    def _open_pluto(cls, configured_uri: str | None):
        contexts = cls.discover()
        primary_uri = resolve_pluto_uri(configured_uri, contexts)
        lease = PlutoDeviceLease.acquire(
            configured_uri,
            primary_uri,
            contexts,
            application="Pluto VSG",
            role="TX",
        )
        candidates = [primary_uri]
        target = str(configured_uri or "").strip()
        if target.lower().startswith("serial:"):
            serial = target.split(":", 1)[1].strip().lower()
            for uri, description in contexts.items():
                match = _SERIAL_PATTERN.search(description)
                if match and match.group(1).lower() == serial and uri not in candidates:
                    candidates.append(uri)

        errors: list[str] = []
        for uri in candidates:
            try:
                sdr = adi.Pluto(uri=uri) if uri is not None else adi.Pluto()
                return sdr, uri, lease
            except Exception as error:
                errors.append(f"{uri or 'auto'}: {error}")
        lease.release()
        detail = "; ".join(errors) or "no usable IIO context"
        raise RuntimeError(f"Unable to open selected ADALM-Pluto ({detail})")

    def transfer(self, result: GenerationResult) -> None:
        iq = np.asarray(result.iq, dtype=np.complex128).reshape(-1)
        if not iq.size:
            raise ValueError("Cannot transmit an empty waveform")
        if not np.all(np.isfinite(iq)):
            raise ValueError("Pluto transmission requires finite IQ samples")
        if float(np.max(np.abs(iq))) > 1.0 + 1e-6:
            raise ValueError("Pluto transmission requires normalized IQ magnitude <= 1.0")
        if not np.isclose(result.sample_rate_hz, self.settings.sample_rate_hz):
            raise ValueError("Transferred waveform sample rate differs from Pluto TX settings")

        burst_count = int(self.settings.burst_count)
        if iq.size % burst_count:
            raise ValueError("Generated IQ cannot be divided into equal Pluto frames")
        self._waveform_peak = float(np.max(np.abs(iq)))
        digital_scale = 10.0 ** (self.settings.digital_backoff_db / 20.0)
        dac_scale = _PLUTO_DAC_FULL_SCALE * digital_scale
        real = np.clip(iq.real, -1.0, 1.0) * dac_scale
        imag = np.clip(iq.imag, -1.0, 1.0) * dac_scale
        self._buffer = (real + 1j * imag).astype(np.complex64)
        self._dac_peak_code = float(np.max(np.abs(self._buffer)))
        self._sample_count = int(iq.size)
        self._frame_sample_count = int(iq.size // burst_count)
        self._preload_guard_s = self.settings.dma_preroll_s
        prefix_count = int(
            round(self._preload_guard_s * self.settings.sample_rate_hz)
        )
        suffix_count = int(
            round(_NONCYCLIC_SUFFIX_GUARD_S * self.settings.sample_rate_hz)
        )
        self._superframe = np.concatenate(
            (
                np.zeros(prefix_count, dtype=np.complex64),
                self._buffer,
                np.zeros(suffix_count, dtype=np.complex64),
            )
        )
        self._record_event("host_frame_prepared")

    @property
    def waveform_duration_s(self) -> float:
        return self._sample_count / self.settings.sample_rate_hz

    @property
    def frame_duration_s(self) -> float:
        return self._frame_sample_count / self.settings.sample_rate_hz

    @property
    def superframe_duration_s(self) -> float:
        """Compatibility alias for the complete finite transmission duration."""

        return self.waveform_duration_s

    @property
    def dma_buffer_duration_s(self) -> float:
        if self._superframe is None:
            return 0.0
        return self._superframe.size / self.settings.sample_rate_hz

    @staticmethod
    def _firmware_version(sdr) -> tuple[int, int] | None:
        attrs = getattr(getattr(sdr, "_ctx", None), "attrs", {})
        value = attrs.get("fw_version") if hasattr(attrs, "get") else None
        match = re.search(r"(\d+)\.(\d+)", str(value or ""))
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _tx_lo_powerdown_attribute(sdr):
        context = getattr(sdr, "_ctx", None)
        if context is None or not hasattr(context, "find_device"):
            return None
        phy = context.find_device("ad9361-phy")
        if phy is None:
            return None
        for channel in getattr(phy, "channels", ()):
            if (
                bool(getattr(channel, "output", False))
                and str(getattr(channel, "id", "")) == "altvoltage1"
                and "powerdown" in getattr(channel, "attrs", {})
            ):
                return channel.attrs["powerdown"]
        return None

    @staticmethod
    def _phy_device(sdr):
        context = getattr(sdr, "_ctx", None)
        if context is None or not hasattr(context, "find_device"):
            return None
        return context.find_device("ad9361-phy")

    @classmethod
    def _phy_attribute(cls, sdr, name: str):
        phy = cls._phy_device(sdr)
        if phy is None:
            return None
        attributes = getattr(phy, "attrs", {})
        return attributes.get(name) if hasattr(attributes, "get") else None

    @classmethod
    def _read_calibration_mode(cls, sdr) -> str:
        attribute = cls._phy_attribute(sdr, "calib_mode")
        if attribute is None:
            raise RuntimeError("Pluto calib_mode control is unavailable")
        raw = str(attribute.value).strip()
        # Pluto firmware may append calibration status/counter information,
        # for example ``manual_tx_quad 21``. Only the first token is the
        # writable policy name listed by calib_mode_available.
        return raw.split(maxsplit=1)[0] if raw else ""

    def _read_calibration_mode_with_diagnostics(self, sdr) -> str:
        attribute = self._phy_attribute(sdr, "calib_mode")
        if attribute is None:
            raise RuntimeError("Pluto calib_mode control is unavailable")
        self._calibration_mode_raw = str(attribute.value).strip()
        return (
            self._calibration_mode_raw.split(maxsplit=1)[0]
            if self._calibration_mode_raw
            else ""
        )

    @classmethod
    def _calibration_modes_available(cls, sdr) -> tuple[str, ...]:
        attribute = cls._phy_attribute(sdr, "calib_mode_available")
        if attribute is None:
            raise RuntimeError("Pluto calib_mode_available is unavailable")
        return tuple(str(attribute.value).split())

    @classmethod
    def _write_calibration_mode(cls, sdr, mode: str) -> None:
        available = cls._calibration_modes_available(sdr)
        if mode not in available:
            raise RuntimeError(
                f"Pluto calibration mode {mode!r} is unavailable "
                f"({', '.join(available)})"
            )
        attribute = cls._phy_attribute(sdr, "calib_mode")
        attribute.value = mode

    def _verify_prepared_configuration(self, sdr) -> None:
        checks = (
            (
                "sample rate",
                float(sdr.sample_rate),
                self.settings.sample_rate_hz,
                max(2.0, self.settings.sample_rate_hz * 1e-6),
            ),
            (
                "TX center frequency",
                float(sdr.tx_lo),
                self.settings.center_frequency_hz,
                10.0,
            ),
            (
                "TX RF bandwidth",
                float(sdr.tx_rf_bandwidth),
                self.settings.rf_bandwidth_hz,
                max(2.0, self.settings.rf_bandwidth_hz * 1e-6),
            ),
        )
        mismatches = [
            f"{label}: device={current:g}, requested={target:g}"
            for label, current, target, tolerance in checks
            if not np.isclose(current, target, rtol=0.0, atol=tolerance)
        ]
        mode = self._read_calibration_mode_with_diagnostics(sdr)
        self._calibration_mode = mode
        if mode != "manual_tx_quad":
            mismatches.append(
                f"calib_mode: device={mode!r}, requested='manual_tx_quad'"
            )
        if mismatches:
            raise RuntimeError(
                "ADALM-Pluto is not READY. Run Prepare after changing RF/baseband "
                "settings. " + "; ".join(mismatches)
            )

    def prepare(self) -> None:
        """Apply RF/baseband settings and run one explicit TX quad calibration.

        Automatic TX quadrature calibration is disabled before configuration.
        The potentially radiating calibration is therefore confined to this
        explicit preparation step and never occurs inside finite transmission.
        """

        sdr, uri, lease = self._open_pluto(self.settings.connection_uri)
        self._opened_uri = uri
        self._sdr = sdr
        self._state = "CONFIGURE"
        try:
            self._set_tx_lo_powerdown(sdr, True)
            sdr.tx_hardwaregain_chan0 = _PLUTO_MUTED_GAIN_DB
            self._record_event("prepare_muted")
            self._firmware = self._firmware_version(sdr)
            available = self._calibration_modes_available(sdr)
            self._record_event("calibration_modes_read")
            if "manual_tx_quad" not in available or "tx_quad" not in available:
                raise RuntimeError(
                    "Pluto firmware does not expose manual_tx_quad and tx_quad "
                    f"({', '.join(available)})"
                )

            self._write_calibration_mode(sdr, "manual_tx_quad")
            self._calibration_mode = self._read_calibration_mode_with_diagnostics(sdr)
            if self._calibration_mode != "manual_tx_quad":
                raise RuntimeError(
                    "Pluto failed to enter manual_tx_quad calibration mode"
                )
            self._record_event("automatic_tx_quad_disabled")

            sdr.tx_enabled_channels = [0]
            self._set_numeric_if_changed(
                sdr,
                "sample_rate",
                int(round(self.settings.sample_rate_hz)),
                tolerance=max(2.0, self.settings.sample_rate_hz * 1e-6),
                event_prefix="sample_rate",
            )
            self._set_numeric_if_changed(
                sdr,
                "tx_lo",
                int(round(self.settings.center_frequency_hz)),
                tolerance=10.0,
                event_prefix="tx_frequency",
            )
            self._set_numeric_if_changed(
                sdr,
                "tx_rf_bandwidth",
                int(round(self.settings.rf_bandwidth_hz)),
                tolerance=max(2.0, self.settings.rf_bandwidth_hz * 1e-6),
                event_prefix="tx_bandwidth",
            )
            self._state = "CALIBRATING"
            self._record_event("explicit_tx_quad_started")
            self._write_calibration_mode(sdr, "tx_quad")
            self._record_event("explicit_tx_quad_completed")
            # Treat tx_quad as a command, not a persistent policy. Reassert
            # manual mode so later writes cannot launch an implicit TX tone.
            self._write_calibration_mode(sdr, "manual_tx_quad")
            self._calibration_mode = self._read_calibration_mode_with_diagnostics(sdr)
            self._verify_prepared_configuration(sdr)
            self._state = "READY"
            self._record_event("pluto_ready")
        except Exception:
            self._state = "ERROR"
            raise
        finally:
            self._mute_and_stop(sdr)
            self._record_event("prepare_cleanup_completed")
            self._sdr = None
            lease.release()

    @classmethod
    def _read_tx_lo_powerdown(cls, sdr):
        attribute = cls._tx_lo_powerdown_attribute(sdr)
        if attribute is None:
            return "<unavailable>"
        try:
            return bool(int(str(attribute.value).strip()))
        except Exception as error:
            return f"<read failed: {error}>"

    @classmethod
    def _set_tx_lo_powerdown(cls, sdr, powerdown: bool) -> None:
        attribute = cls._tx_lo_powerdown_attribute(sdr)
        if attribute is None:
            raise RuntimeError("Pluto TX LO powerdown control is unavailable")
        attribute.value = "1" if powerdown else "0"
        readback = cls._read_tx_lo_powerdown(sdr)
        if readback != bool(powerdown):
            raise RuntimeError(
                "Pluto TX LO powerdown read-back failed "
                f"(requested {bool(powerdown)}, found {readback})"
            )

    @classmethod
    def _mute_and_stop(cls, sdr) -> None:
        """Mute RF first, then remove DMA and select the DAC zero source."""

        try:
            sdr.tx_hardwaregain_chan0 = _PLUTO_MUTED_GAIN_DB
        except Exception:
            pass
        try:
            cls._set_tx_lo_powerdown(sdr, True)
        except Exception:
            pass
        try:
            sdr.tx_destroy_buffer()
        except Exception:
            pass
        try:
            sdr.tx_enabled_channels = []
            sdr.tx()
        except Exception:
            pass
    def _wait_precise(self, duration_s: float) -> None:
        """Wait for a short RF window without Windows timer-quantum overshoot."""

        deadline = time.perf_counter() + max(0.0, float(duration_s))
        while not self._stop_event.is_set():
            remaining = deadline - time.perf_counter()
            if remaining <= 0.0:
                return
            if remaining > 0.004:
                self._stop_event.wait(remaining - 0.002)
            else:
                # Sub-millisecond packet windows need a short busy wait;
                # Sleep/Event waits commonly overshoot by 15.6 ms on Windows.
                pass

    @staticmethod
    def _commit_noncyclic_stream(sdr) -> bool:
        """Enqueue the first libiio v1 TX stream block.

        pyadi-iio's libiio v1 compatibility layer obtains and writes the first
        stream block on ``tx(data)``, but that block is not enqueued until the
        stream advances. Repeated streaming calls do this implicitly. A
        one-shot VSG transfer must advance it explicitly or cleanup discards
        the only written block before it ever reaches the DMA.
        """

        stream = getattr(sdr, "_tx_stream", None)
        if stream is None:
            return False
        next(stream)
        return True

    def start(self) -> None:
        if self._buffer is None or self._superframe is None:
            raise RuntimeError("Transfer a waveform before starting Pluto TX")
        sdr, uri, lease = self._open_pluto(self.settings.connection_uri)
        self._opened_uri = uri
        self._sdr = sdr
        try:
            # TX LO powerdown is the hard mute. Gain attenuation remains a
            # second layer, but is not relied upon to suppress preparation.
            self._set_tx_lo_powerdown(sdr, True)
            self._record_event("tx_lo_powered_down")
            sdr.tx_hardwaregain_chan0 = _PLUTO_MUTED_GAIN_DB
            self._record_event("gain_muted")
            version = self._firmware_version(sdr)
            self._firmware = version
            self._observe_hardware("after_initial_mute", sdr)
            # No RF/baseband property may be written from this point. A
            # mismatch means Prepare was skipped or another application
            # changed the device, so fail muted instead of auto-calibrating.
            self._verify_prepared_configuration(sdr)
            self._state = "TRANSMITTING"
            self._record_event("prepared_configuration_verified")
            sdr.tx_enabled_channels = [0]
            sdr.tx_cyclic_buffer = False
            self._record_event("noncyclic_mode_selected")

            if self._stop_event.is_set():
                return
            # RF reconfiguration can rewrite AD936x gain state. Reassert and
            # verify mute before enabling the LO; the actual non-cyclic push
            # below is the software start operation.
            sdr.tx_hardwaregain_chan0 = _PLUTO_MUTED_GAIN_DB
            muted_readback = float(sdr.tx_hardwaregain_chan0)
            if not np.isclose(muted_readback, _PLUTO_MUTED_GAIN_DB, atol=0.01):
                raise RuntimeError(
                    "Pluto TX mute read-back failed before DMA buffer transfer "
                    f"({muted_readback:+.2f} dB)"
                )
            self._record_event("gain_remuted_before_transfer")
            # Power the LO while gain remains at the hardware minimum. The
            # user-configured lead guard is a muted LO-settling interval.
            self._set_tx_lo_powerdown(sdr, False)
            self._record_event("tx_lo_powered_up")
            self._observe_hardware("after_tx_lo_powerup", sdr)
            if self._stop_event.wait(self.settings.lead_in_guard_s):
                return
            self._record_event("muted_lo_settled")

            sdr.tx_hardwaregain_chan0 = self.settings.resolved_hardware_gain_db
            self._record_event("requested_gain_applied")
            if self._stop_event.is_set():
                return

            # Non-cyclic push starts from sample zero exactly once. The short
            # zero prefix is intentionally part of the buffer so any DMA/DAC
            # source transition cannot truncate the first packet.
            self._observe_hardware("before_noncyclic_push", sdr)
            self._record_event("noncyclic_push_started")
            sdr.tx(self._superframe)
            self._record_event("noncyclic_push_completed")
            if self._commit_noncyclic_stream(sdr):
                self._record_event("noncyclic_stream_committed")
            self._record_event("packet_schedule_submitted")
            # libiio implementations differ on whether push returns after
            # queueing or after playback. Keep the user completion margin as
            # a post-submit hold, but do not inflate the DMA buffer with a long
            # zero tail: large one-shot USB buffers proved unreliable.
            self._wait_precise(
                max(self.dma_buffer_duration_s, self.settings.stop_guard_s)
            )
            self._record_event("finite_buffer_elapsed")
        except Exception:
            self._state = "ERROR"
            raise
        finally:
            # Do not perform IIO read-backs while RF is unmuted: each USB
            # transaction can lengthen the on-air window by a scheduler tick.
            self._record_event("cleanup_started")
            self._mute_and_stop(sdr)
            self._record_event("cleanup_completed")
            self._observe_hardware("after_cleanup", sdr)
            if self._state != "ERROR":
                self._state = "READY"
            self._sdr = None
            lease.release()

    def stop(self) -> None:
        """Request cancellation; cleanup is performed by the TX owner thread."""

        self._stop_event.set()
