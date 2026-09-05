"""Shared VSA display-DSP used by generic and dedicated analyzers.

Protocol modes choose signal settings and result ranges. Filtering,
normalization, and constellation reference transforms live here so a PHY
preset cannot silently change measurement presentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pluto_sa.vsa.model import ModulationKind
from pluto_sa.vsa.demod.gfsk import prepare_fsk_frequency
from pluto_sa.vsa.pattern import prepare_psk_iq


@dataclass(frozen=True)
class FSKDisplayData:
    """One display-only FSK series shared by line and symbol views."""

    time_s: np.ndarray
    corrected_frequency_hz: np.ndarray
    symbol_time_s: np.ndarray
    symbol_frequency_hz: np.ndarray


def format_evm(percent: float) -> str:
    value = float(percent)
    if not np.isfinite(value) or value < 0.0:
        return "--"
    db_text = "-inf" if value == 0.0 else f"{20.0 * np.log10(value / 100.0):.1f}"
    return f"{value:.2f} % / {db_text} dB"


def prepare_psk_display_waveform(
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    tx_filter: str,
    filter_parameter: float | None,
    apply_measurement_filter: bool = True,
    result_start_time_s: float | None = None,
    result_stop_time_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(iq)
    start_sample = 0
    stop_sample = values.size
    if result_start_time_s is not None and result_stop_time_s is not None:
        guard_s = 16.0 / float(symbol_rate_hz)
        start_sample = max(0, int(np.floor((float(result_start_time_s) - guard_s) * sample_rate_hz)))
        stop_sample = min(values.size, int(np.ceil((float(result_stop_time_s) + guard_s) * sample_rate_hz)))
    prepared, prepared_rate_hz = prepare_psk_iq(
        values[start_sample:stop_sample],
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        tx_filter=tx_filter,
        filter_parameter=filter_parameter,
        apply_measurement_filter=apply_measurement_filter,
    )
    time_offset_s = start_sample / float(sample_rate_hz)
    time_s = time_offset_s + np.arange(prepared.size, dtype=np.float64) / float(prepared_rate_hz)
    return prepared, time_s


def prepare_fsk_display_frequency(
    iq: np.ndarray,
    *,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    gaussian_bt: float | None,
    result_start_time_s: float | None = None,
    result_stop_time_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare the common Measured FSK trace used by every VSA mode."""
    values = np.asarray(iq)
    start_sample = 0
    stop_sample = values.size
    if result_start_time_s is not None and result_stop_time_s is not None:
        guard_s = 16.0 / float(symbol_rate_hz)
        start_sample = max(
            0,
            int(
                np.floor(
                    (float(result_start_time_s) - guard_s) * sample_rate_hz
                )
            ),
        )
        stop_sample = min(
            values.size,
            int(
                np.ceil(
                    (float(result_stop_time_s) + guard_s) * sample_rate_hz
                )
            ),
        )
    frequency_hz, prepared_rate_hz = prepare_fsk_frequency(
        values[start_sample:stop_sample],
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        gaussian_bt=gaussian_bt,
    )
    time_offset_s = start_sample / float(sample_rate_hz)
    time_s = time_offset_s + np.arange(
        frequency_hz.size, dtype=np.float64
    ) / float(prepared_rate_hz)
    return frequency_hz, time_s


def sample_fsk_display_trace(
    frequency_hz: np.ndarray,
    time_s: np.ndarray,
    symbol_time_s: np.ndarray,
) -> np.ndarray:
    """Sample the displayed FSK trace at recovered symbol-center times.

    This function is display-only.  Decoder decisions and RF-PHY measurement
    values retain their protocol-specific filtering and symbol-window rules.
    """

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    trace_time = np.asarray(time_s, dtype=np.float64)
    symbol_time = np.asarray(symbol_time_s, dtype=np.float64)
    count = min(frequency.size, trace_time.size)
    if count == 0 or symbol_time.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.interp(symbol_time, trace_time[:count], frequency[:count])


def build_fsk_display_data(
    frequency_hz: np.ndarray,
    time_s: np.ndarray,
    symbol_time_s: np.ndarray,
    *,
    frequency_offset_hz: float = 0.0,
    frequency_drift_hz_per_s: float = 0.0,
    reference_time_s: float = 0.0,
) -> FSKDisplayData:
    """Build the corrected frequency series used by every FSK display."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    trace_time = np.asarray(time_s, dtype=np.float64)
    symbol_time = np.asarray(symbol_time_s, dtype=np.float64)
    count = min(frequency.size, trace_time.size)
    trace_time = np.array(trace_time[:count], copy=True)
    correction = float(frequency_offset_hz) + float(
        frequency_drift_hz_per_s
    ) * (trace_time - float(reference_time_s))
    corrected = np.array(frequency[:count] - correction, copy=True)
    sampled = sample_fsk_display_trace(corrected, trace_time, symbol_time)
    symbol_time = np.array(symbol_time, copy=True)
    for values in (trace_time, corrected, symbol_time, sampled):
        values.setflags(write=False)
    return FSKDisplayData(
        time_s=trace_time,
        corrected_frequency_hz=corrected,
        symbol_time_s=symbol_time,
        symbol_frequency_hz=sampled,
    )


def fit_binary_fsk_display_drift(
    symbol_time_s: np.ndarray,
    symbol_frequency_hz: np.ndarray,
    symbols: np.ndarray,
) -> tuple[float, float]:
    """Estimate display-only carrier drift while fitting both FSK levels."""

    time_s = np.asarray(symbol_time_s, dtype=np.float64)
    frequency_hz = np.asarray(symbol_frequency_hz, dtype=np.float64)
    bits = np.asarray(symbols, dtype=np.float64)
    count = min(time_s.size, frequency_hz.size, bits.size)
    if count < 3:
        return 0.0, float(time_s[0]) if count else 0.0
    time_s = time_s[:count]
    frequency_hz = frequency_hz[:count]
    levels = 2.0 * bits[:count] - 1.0
    finite = np.isfinite(time_s) & np.isfinite(frequency_hz) & np.isfinite(levels)
    if np.count_nonzero(finite) < 3:
        return 0.0, float(np.nanmean(time_s))
    time_s = time_s[finite]
    frequency_hz = frequency_hz[finite]
    levels = levels[finite]
    reference_time_s = float(np.mean(time_s))
    design = np.column_stack(
        (np.ones(time_s.size), levels, time_s - reference_time_s)
    )
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        design, frequency_hz, rcond=None
    )
    drift_hz_per_s = float(coefficients[2]) if rank == 3 else 0.0
    return drift_hz_per_s, reference_time_s


def constellation_display_symbols(modulation: ModulationKind, symbols: np.ndarray) -> np.ndarray:
    values = np.asarray(symbols, dtype=np.complex128)
    if modulation in {ModulationKind.QPSK, ModulationKind.OQPSK, ModulationKind.PI4_DQPSK}:
        values = values * np.exp(-1j * np.pi / 4.0)
    return values


def physical_constellation_display_symbols(modulation: ModulationKind, symbols: np.ndarray) -> np.ndarray:
    values = np.asarray(symbols, dtype=np.complex128)
    if modulation is ModulationKind.PI4_DQPSK:
        rotation = (np.arange(values.size, dtype=np.float64) + 1.0) * np.pi / 4.0
        return values * np.exp(-1j * rotation)
    if modulation in {ModulationKind.QPSK, ModulationKind.OQPSK}:
        return values * np.exp(-1j * np.pi / 4.0)
    return values


def normalized_psk_display(
    iq: np.ndarray,
    time_s: np.ndarray,
    symbol_time_s: np.ndarray,
    *,
    modulation: ModulationKind,
    differential_symbols: np.ndarray,
    physical: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized trajectory, sampled physical IQ, and Symbol Plot data."""
    values = np.asarray(iq, dtype=np.complex128)
    times = np.asarray(time_s, dtype=np.float64)
    symbol_times = np.asarray(symbol_time_s, dtype=np.float64)
    count = min(values.size, times.size)
    sampled = np.empty(0, dtype=np.complex128)
    if count and symbol_times.size:
        valid = (symbol_times >= times[0]) & (symbol_times <= times[count - 1])
        symbol_times = symbol_times[valid]
        sampled = np.interp(symbol_times, times[:count], values[:count].real) + 1j * np.interp(symbol_times, times[:count], values[:count].imag)
    rms = float(np.sqrt(np.mean(np.abs(sampled) ** 2))) if sampled.size else 1.0
    if np.isfinite(rms) and rms > 1e-12:
        values = values / rms
        sampled = sampled / rms
    plotted = (
        physical_constellation_display_symbols(modulation, sampled)
        if physical
        else constellation_display_symbols(modulation, differential_symbols)
    )
    return values, sampled, plotted
