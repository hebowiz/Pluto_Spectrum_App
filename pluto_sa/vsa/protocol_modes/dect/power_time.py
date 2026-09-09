"""ETSI DECT physical-packet power-time template measurement.

The verdict path deliberately uses unsmoothed per-sample RF power.  Display
smoothing, burst detection, and modulation measurements are separate users of
the IQ recording and must not alter these samples.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve, firwin

from pluto_sa.vsa.model import IQRecording


@dataclass(frozen=True)
class DectPowerTimeTemplate:
    """EN 300 176-1 clause 9 limits in SI units."""

    attack_release_limit_s: float = 10e-6
    maintenance_time_s: float = 0.5e-6
    idle_guard_time_s: float = 27e-6
    minimum_relative_db: float = -1.0
    maximum_relative_db: float = 1.0
    attack_relative_db: float = 4.0
    maintenance_relative_db: float = -6.0
    attack_release_threshold_w: float = 25e-6
    absolute_peak_limit_w: float = 315e-3
    idle_limit_w: float = 20e-9
    measurement_bandwidth_hz: float = 3e6
    idle_measurement_bandwidth_hz: float = 1e6
    timing_accuracy_target_s: float = 0.1e-6


@dataclass(frozen=True)
class DectPowerTimeCriterion:
    name: str
    value: str
    limit: str
    status: str
    failure_samples: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        samples = np.array(self.failure_samples, dtype=np.float64, copy=True)
        samples.setflags(write=False)
        object.__setattr__(self, "failure_samples", samples)


@dataclass(frozen=True)
class DectPowerTimeResult:
    template: DectPowerTimeTemplate
    p0_sample: float
    power_time_start_sample: float
    packet_end_sample: float
    next_power_time_start_sample: float | None
    reference_power_linear: float
    reference_power_db: float
    power_unit: str
    attack_time_s: float | None
    release_time_s: float | None
    measurement_bandwidth_hz: float
    idle_measurement_bandwidth_hz: float
    amplitude_calibrated: bool
    timing_resolution_s: float
    criteria: tuple[DectPowerTimeCriterion, ...]
    overall_status: str
    incomplete_reasons: tuple[str, ...] = ()

    @property
    def criterion_map(self) -> dict[str, DectPowerTimeCriterion]:
        return {criterion.name: criterion for criterion in self.criteria}


@dataclass(frozen=True)
class DectPowerMeasurementPaths:
    """Parallel ETSI measurement-receiver outputs from one common IQ block."""

    raw_power: np.ndarray
    power_time_power: np.ndarray
    one_mhz_power: np.ndarray
    power_time_available: bool
    ntp_available: bool
    amplitude_calibrated: bool
    power_unit: str
    input_usable_bandwidth_hz: float
    power_time_measurement_bandwidth_hz: float
    ntp_measurement_bandwidth_hz: float
    idle_measurement_bandwidth_hz: float
    power_time_filter_type: str
    ntp_filter_type: str
    power_time_filter_group_delay_samples: float
    ntp_filter_group_delay_samples: float
    filter_delay_compensated: bool = True

    def __post_init__(self) -> None:
        for name in ("raw_power", "power_time_power", "one_mhz_power"):
            values = np.array(getattr(self, name), dtype=np.float64, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class DectNTPResult:
    power_linear: float
    power_db: float
    power_unit: str
    measurement_bandwidth_hz: float
    available: bool
    start_sample: float
    stop_sample: float
    reason: str = ""


def _weighted_mean(values: np.ndarray, start: float, stop: float) -> float:
    """Mean a sample-and-hold sequence over fractional sample boundaries."""

    lo = max(0, int(np.floor(start)))
    hi = min(values.size, int(np.ceil(stop)))
    if hi <= lo or stop <= start:
        return float("nan")
    indices = np.arange(lo, hi, dtype=np.float64)
    weights = np.maximum(0.0, np.minimum(indices + 1.0, stop) - np.maximum(indices, start))
    total = float(np.sum(weights))
    return float(np.sum(values[lo:hi] * weights) / total) if total > 0.0 else float("nan")


def _region(
    values: np.ndarray, start: float, stop: float
) -> tuple[np.ndarray, np.ndarray, bool]:
    complete = bool(start >= 0.0 and stop <= values.size and stop > start)
    lo = max(0, int(np.ceil(start - 0.5)))
    hi = min(values.size, int(np.ceil(stop - 0.5)))
    indices = np.arange(lo, hi, dtype=np.float64)
    return values[lo:hi], indices, complete


def _crossing_sample(
    values: np.ndarray,
    threshold: float,
    start: float,
    stop: float,
    *,
    rising: bool,
) -> float | None:
    lo = max(0, int(np.floor(start)) - 1)
    hi = min(values.size - 1, int(np.ceil(stop)))
    if hi <= lo:
        return None
    left = values[lo:hi]
    right = values[lo + 1 : hi + 1]
    mask = ((left < threshold) & (right >= threshold)) if rising else (
        (left >= threshold) & (right < threshold)
    )
    edges = np.flatnonzero(mask)
    if not edges.size:
        return None
    crossings: list[float] = []
    for local_edge in edges:
        edge = int(local_edge) + lo
        delta = float(values[edge + 1] - values[edge])
        fraction = 0.0 if delta == 0.0 else float((threshold - values[edge]) / delta)
        crossing = float(edge + np.clip(fraction, 0.0, 1.0))
        if start <= crossing <= stop:
            crossings.append(crossing)
    if not crossings:
        return None
    return crossings[-1] if rising else crossings[0]


def _db(value: float) -> float:
    return float(10.0 * np.log10(max(float(value), np.finfo(float).tiny)))


def _criterion(
    name: str,
    value: str,
    limit: str,
    status: str,
    failures: np.ndarray | None = None,
) -> DectPowerTimeCriterion:
    return DectPowerTimeCriterion(
        name, value, limit, status,
        np.empty(0, dtype=np.float64) if failures is None else failures,
    )


def _complex_measurement_receiver(
    iq: np.ndarray,
    sample_rate_hz: float,
    bandwidth_hz: float,
) -> tuple[np.ndarray, str, float]:
    """Apply an explicit unity-DC-gain, linear-phase measurement receiver.

    ``fftconvolve(..., mode='same')`` centers the symmetric FIR result, so the
    nominal group delay is removed from the returned time axis.  The filter's
    transition response remains present; this is not a ramp correction.
    """

    values = np.asarray(iq, dtype=np.complex128)
    if bandwidth_hz >= sample_rate_hz:
        return np.array(values, copy=True), "Nyquist-limited (no additional FIR)", 0.0
    count = int(np.ceil(8.0 * sample_rate_hz / bandwidth_hz))
    count = max(65, min(2049, count | 1))
    taps = firwin(
        count,
        cutoff=0.5 * bandwidth_hz,
        fs=sample_rate_hz,
        window=("kaiser", 8.0),
        scale=True,
    )
    filtered = fftconvolve(values, taps, mode="same")
    return filtered, f"{count}-tap Kaiser FIR, unity DC gain", 0.5 * (count - 1)


def build_dect_power_measurement_paths(
    recording: IQRecording,
    *,
    power_time_bandwidth_hz: float = 3e6,
    one_mhz_bandwidth_hz: float = 1e6,
) -> DectPowerMeasurementPaths:
    """Branch 3 MHz and 1 MHz receivers from the same common DECT IQ."""

    rate = float(recording.sample_rate_hz)
    usable = min(rate, float(recording.usable_bandwidth_hz or rate))
    normalized_iq = np.asarray(recording.iq, dtype=np.complex128) / recording.full_scale
    scale_mw = 10.0 ** (recording.dbfs_to_dbm_offset_db / 10.0)
    scale = scale_mw / 1000.0 if recording.amplitude_calibrated else 1.0
    power_time_iq, power_time_filter, power_time_delay = _complex_measurement_receiver(
        normalized_iq, rate, float(power_time_bandwidth_hz)
    )
    one_mhz_iq, ntp_filter, ntp_delay = _complex_measurement_receiver(
        normalized_iq, rate, float(one_mhz_bandwidth_hz)
    )
    return DectPowerMeasurementPaths(
        raw_power=np.abs(normalized_iq) ** 2 * scale,
        power_time_power=np.abs(power_time_iq) ** 2 * scale,
        one_mhz_power=np.abs(one_mhz_iq) ** 2 * scale,
        power_time_available=usable >= float(power_time_bandwidth_hz),
        ntp_available=usable >= float(one_mhz_bandwidth_hz),
        amplitude_calibrated=bool(recording.amplitude_calibrated),
        power_unit="dBm" if recording.amplitude_calibrated else "dBFS",
        input_usable_bandwidth_hz=usable,
        power_time_measurement_bandwidth_hz=float(power_time_bandwidth_hz),
        ntp_measurement_bandwidth_hz=float(one_mhz_bandwidth_hz),
        idle_measurement_bandwidth_hz=float(one_mhz_bandwidth_hz),
        power_time_filter_type=power_time_filter,
        ntp_filter_type=ntp_filter,
        power_time_filter_group_delay_samples=power_time_delay,
        ntp_filter_group_delay_samples=ntp_delay,
    )


def measure_dect_ntp(
    paths: DectPowerMeasurementPaths,
    *,
    p0_sample: float,
    packet_end_sample: float,
) -> DectNTPResult:
    """Measure clause 10 NTP with the 1 MHz path over p0..packet end."""

    complete = p0_sample >= 0.0 and packet_end_sample <= paths.one_mhz_power.size
    available = bool(paths.ntp_available and complete)
    value = _weighted_mean(paths.one_mhz_power, p0_sample, packet_end_sample)
    value_db = _db(value) + (30.0 if paths.amplitude_calibrated else 0.0)
    reason = ""
    if not paths.ntp_available:
        reason = "input usable bandwidth is below 1 MHz"
    elif not complete:
        reason = "p0..physical packet end is not fully captured"
    elif not paths.amplitude_calibrated:
        reason = "absolute amplitude calibration is required for formal NTP"
    return DectNTPResult(
        power_linear=value,
        power_db=value_db,
        power_unit=paths.power_unit,
        measurement_bandwidth_hz=paths.ntp_measurement_bandwidth_hz,
        available=available and paths.amplitude_calibrated,
        start_sample=float(p0_sample),
        stop_sample=float(packet_end_sample),
        reason=reason,
    )


def measure_dect_power_time(
    recording: IQRecording,
    *,
    p0_sample: float,
    power_time_start_sample: float,
    packet_end_sample: float,
    next_power_time_start_sample: float | None = None,
    measurement_bandwidth_hz: float | None = None,
    measurement_paths: DectPowerMeasurementPaths | None = None,
    template: DectPowerTimeTemplate | None = None,
) -> DectPowerTimeResult:
    """Measure one physical packet against EN 300 176-1 clause 9."""

    limits = template or DectPowerTimeTemplate()
    rate = float(recording.sample_rate_hz)
    paths = measurement_paths or build_dect_power_measurement_paths(recording)
    bandwidth = float(paths.power_time_measurement_bandwidth_hz)
    power = paths.power_time_power
    calibrated = bool(paths.amplitude_calibrated)
    unit = "dBm" if calibrated else "dBFS"
    reference_power = _weighted_mean(power, power_time_start_sample, packet_end_sample)
    reference_power_db = _db(reference_power) + (30.0 if calibrated else 0.0)
    criteria: list[DectPowerTimeCriterion] = []
    reasons: list[str] = []

    bandwidth_ok = bool(
        paths.power_time_available
        and (
            measurement_bandwidth_hz is None
            or float(measurement_bandwidth_hz) >= limits.measurement_bandwidth_hz
        )
    )
    if not bandwidth_ok:
        reasons.append("at least 3 MHz usable measurement bandwidth is required")
    if not calibrated:
        reasons.append("absolute amplitude calibration is required")
    timing_resolution = 1.0 / rate
    if timing_resolution > limits.timing_accuracy_target_s:
        reasons.append("sample interval exceeds the 0.1 us timing target")

    span = limits.attack_release_limit_s * rate
    threshold = limits.attack_release_threshold_w
    attack_cross = (
        _crossing_sample(power, threshold, power_time_start_sample - span, power_time_start_sample, rising=True)
        if calibrated and bandwidth_ok else None
    )
    release_cross = (
        _crossing_sample(power, threshold, packet_end_sample, packet_end_sample + span, rising=False)
        if calibrated and bandwidth_ok else None
    )
    attack_time = None if attack_cross is None else (power_time_start_sample - attack_cross) / rate
    release_time = None if release_cross is None else (release_cross - packet_end_sample) / rate
    attack_window_complete = power_time_start_sample - span >= 0.0
    release_window_complete = packet_end_sample + span <= power.size
    attack_missing_status = (
        "FAIL" if calibrated and bandwidth_ok and attack_window_complete
        else "INCOMPLETE"
    )
    release_missing_status = (
        "FAIL" if calibrated and bandwidth_ok and release_window_complete
        else "INCOMPLETE"
    )
    attack_status = attack_missing_status if attack_time is None else (
        "PASS" if power_time_start_sample - float(attack_cross) < span - 1e-6 else "FAIL"
    )
    attack_failures = (
        np.asarray([attack_cross], dtype=np.float64)
        if attack_cross is not None and attack_status == "FAIL"
        else np.arange(max(0.0, power_time_start_sample - span), power_time_start_sample)
        if attack_cross is None and attack_status == "FAIL"
        else None
    )
    criteria.append(_criterion(
        "Attack Time",
        "N/A" if attack_time is None else f"{attack_time * 1e6:.3f} us",
        "< 10 us from 25 uW crossing to packet start",
        attack_status,
        attack_failures,
    ))
    release_status = release_missing_status if release_time is None else (
        "PASS" if float(release_cross) - packet_end_sample < span - 1e-6 else "FAIL"
    )
    release_failures = (
        np.asarray([release_cross], dtype=np.float64)
        if release_cross is not None and release_status == "FAIL"
        else np.arange(packet_end_sample, min(float(power.size), packet_end_sample + span))
        if release_cross is None and release_status == "FAIL"
        else None
    )
    criteria.append(_criterion(
        "Release Time",
        "N/A" if release_time is None else f"{release_time * 1e6:.3f} us",
        "< 10 us from packet end to 25 uW crossing",
        release_status,
        release_failures,
    ))

    def relative_region(name: str, start: float, stop: float, offset_db: float, *, maximum: bool) -> None:
        values, indices, complete = _region(power, start, stop)
        threshold_value = reference_power * 10.0 ** (offset_db / 10.0)
        if not bandwidth_ok or not complete or not values.size:
            criteria.append(_criterion(name, "N/A", f"{'<' if maximum else '>'} reference {offset_db:+.0f} dB", "INCOMPLETE"))
            if not complete:
                reasons.append(f"{name} interval is not fully captured")
            return
        failed = values >= threshold_value if maximum else values <= threshold_value
        measured = float(np.max(values) if maximum else np.min(values))
        criteria.append(_criterion(
            name,
            f"{_db(measured / reference_power):+.3f} dB rel. reference",
            f"{'<' if maximum else '>'} reference {offset_db:+.0f} dB",
            "FAIL" if np.any(failed) else "PASS",
            indices[failed],
        ))

    relative_region("Minimum Packet Power", power_time_start_sample, packet_end_sample, -1.0, maximum=False)
    relative_region(
        "Maximum Packet Power",
        power_time_start_sample + limits.attack_release_limit_s * rate,
        packet_end_sample + limits.attack_release_limit_s * rate,
        1.0,
        maximum=True,
    )

    attack_values, attack_indices, attack_complete = _region(
        power,
        power_time_start_sample - span,
        power_time_start_sample + span,
    )
    attack_relative_limit = reference_power * 10.0 ** (limits.attack_relative_db / 10.0)
    if not bandwidth_ok or not attack_complete or not attack_values.size:
        criteria.append(_criterion("Attack Region Maximum", "N/A", "< min(reference +4 dB, 315 mW)", "INCOMPLETE"))
        if not attack_complete:
            reasons.append("Attack Region Maximum interval is not fully captured")
    else:
        relative_fail = attack_values >= attack_relative_limit
        absolute_fail = attack_values >= limits.absolute_peak_limit_w if calibrated else np.zeros(attack_values.size, dtype=bool)
        status = "FAIL" if np.any(relative_fail | absolute_fail) else "PASS" if calibrated else "INCOMPLETE"
        criteria.append(_criterion(
            "Attack Region Maximum",
            f"{_db(float(np.max(attack_values)) / reference_power):+.3f} dB rel. reference" + (
                f" / {_db(float(np.max(attack_values))) + 30.0:.3f} dBm" if calibrated else ""
            ),
            "< min(reference +4 dB, 315 mW)",
            status,
            attack_indices[relative_fail | absolute_fail],
        ))

    relative_region(
        "Post-Packet Maintenance",
        packet_end_sample,
        packet_end_sample + limits.maintenance_time_s * rate,
        -6.0,
        maximum=False,
    )

    idle_status = "INCOMPLETE"
    idle_value = "N/A"
    idle_failures = np.empty(0, dtype=np.float64)
    if next_power_time_start_sample is not None:
        gap = next_power_time_start_sample - packet_end_sample
        if gap < 2.0 * limits.idle_guard_time_s * rate:
            idle_status = "NOT APPLICABLE"
            idle_value = "Next packet starts < 54 us after packet end"
        elif calibrated:
            idle_start = packet_end_sample + limits.idle_guard_time_s * rate
            idle_stop = next_power_time_start_sample - limits.idle_guard_time_s * rate
            idle_values, idle_indices, idle_complete = _region(
                paths.one_mhz_power, idle_start, idle_stop
            )
            if idle_complete and idle_values.size:
                failed = idle_values >= limits.idle_limit_w
                idle_failures = idle_indices[failed]
                idle_value = f"{_db(float(np.max(idle_values))) + 30.0:.3f} dBm max"
                idle_status = "FAIL" if np.any(failed) else "PASS"
            else:
                reasons.append("Idle Power interval is not fully captured")
        else:
            reasons.append("Idle Power requires absolute amplitude calibration")
    else:
        reasons.append("next packet start is not captured; Idle Power is not evaluable")
    criteria.append(_criterion("Idle Power", idle_value, "< 20 nW (-46.990 dBm), 1 MHz BW", idle_status, idle_failures))

    statuses = {criterion.status for criterion in criteria}
    overall = "FAIL" if "FAIL" in statuses else "INCOMPLETE" if (
        reasons or "INCOMPLETE" in statuses
    ) else "PASS"
    return DectPowerTimeResult(
        template=limits,
        p0_sample=float(p0_sample),
        power_time_start_sample=float(power_time_start_sample),
        packet_end_sample=float(packet_end_sample),
        next_power_time_start_sample=(None if next_power_time_start_sample is None else float(next_power_time_start_sample)),
        reference_power_linear=reference_power,
        reference_power_db=reference_power_db,
        power_unit=unit,
        attack_time_s=attack_time,
        release_time_s=release_time,
        measurement_bandwidth_hz=bandwidth,
        idle_measurement_bandwidth_hz=limits.idle_measurement_bandwidth_hz,
        amplitude_calibrated=calibrated,
        timing_resolution_s=timing_resolution,
        criteria=tuple(criteria),
        overall_status=overall,
        incomplete_reasons=tuple(dict.fromkeys(reasons)),
    )
