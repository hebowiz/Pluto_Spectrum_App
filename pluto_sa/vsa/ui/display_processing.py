"""Shared VSA display-DSP used by generic and dedicated analyzers.

Protocol modes choose signal settings and result ranges. Filtering,
normalization, and constellation reference transforms live here so a PHY
preset cannot silently change measurement presentation.
"""

from __future__ import annotations

import numpy as np

from pluto_sa.vsa.model import ModulationKind
from pluto_sa.vsa.demod.gfsk import prepare_fsk_frequency
from pluto_sa.vsa.pattern import prepare_psk_iq


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
