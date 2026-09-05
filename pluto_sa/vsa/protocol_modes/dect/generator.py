"""Deterministic Classic DECT GFSK packets for analyzer and future VSG use."""

from __future__ import annotations

import numpy as np

from pluto_protocol.dect.common import r_crc_bits, x_crc_bits
from pluto_sa.vsa.demod.fsk_reference import fsk_reference_frequency_levels
from pluto_sa.vsa.model import IQRecording


DECT_SYMBOL_RATE_HZ = 1_152_000.0
DECT_FREQUENCY_DEVIATION_HZ = 288_000.0
RFP_SYNC_BITS = np.array(
    [int(bit) for bit in "10101010101010101110100110001010"], dtype=np.uint8
)
PP_SYNC_BITS = 1 - RFP_SYNC_BITS
PACKET_SYMBOL_COUNTS = {"P00": 96, "P32": 420, "P32Z": 424, "P80": 900, "P80Z": 904}


def _payload_bits(pattern: str, count: int, seed: int) -> np.ndarray:
    normalized = str(pattern).strip().lower().replace("_", "")
    if normalized in {"00001111", "casea", "case_a"}:
        source = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
    elif normalized in {"00110011", "0011"}:
        source = np.array([0, 0, 1, 1], dtype=np.uint8)
    elif normalized in {"0101", "alternating", "caseb", "case_b"}:
        source = np.array([0, 1], dtype=np.uint8)
    elif normalized in {"prbs9", "prbs-9"}:
        state = 0x1FF
        values = []
        for _ in range(count):
            values.append(state & 1)
            feedback = ((state >> 4) ^ (state >> 8)) & 1
            state = (state >> 1) | (feedback << 8)
        return np.asarray(values, dtype=np.uint8)
    elif normalized == "random":
        return np.random.default_rng(seed).integers(0, 2, count, dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported DECT payload pattern: {pattern}")
    return np.resize(source, count).astype(np.uint8)


def _apply_error_control(bits: np.ndarray, packet_type: str) -> np.ndarray:
    """Fill the A-field R-CRC and format-specific 2-level X/Z fields."""

    result = np.array(bits, dtype=np.uint8, copy=True)
    if result.size >= 96:
        result[80:96] = r_crc_bits(result[32:80])
    layout = {"P32": (320, False), "P32Z": (320, True), "P80": (800, False), "P80Z": (800, True)}.get(packet_type)
    if layout is None:
        return result
    b_size, has_z = layout
    b_start = 96
    b_bits = result[b_start:b_start + b_size]
    if b_size == 320:
        indices = [index + 48 * (1 + index // 16) for index in range(80)]
    else:
        indices = [index + 64 * (1 + index // 16) for index in range(160)]
    x_start = b_start + b_size
    result[x_start:x_start + 4] = x_crc_bits(b_bits[indices])
    if has_z:
        result[x_start + 4:x_start + 8] = result[x_start:x_start + 4]
    return result


def generate_dect_packet(
    *,
    direction: str = "RFP",
    packet_type: str = "P32",
    payload_pattern: str = "00001111",
    center_frequency_hz: float = 1_888_704_000.0,
    samples_per_symbol: int = 8,
    frequency_error_hz: float = 0.0,
    symbol_rate_error_ppm: float = 0.0,
    frequency_deviation_hz: float = DECT_FREQUENCY_DEVIATION_HZ,
    power_dbm: float = -10.0,
    attack_time_s: float = 4e-6,
    release_time_s: float = 4e-6,
    padding_symbols: int = 24,
    prolonged_preamble: bool = False,
    seed: int = 1,
) -> IQRecording:
    """Generate one burst with power edges outside the p0..packet-end range."""

    packet_key = str(packet_type).upper()
    if packet_key not in PACKET_SYMBOL_COUNTS:
        raise ValueError(f"Unsupported DECT packet type: {packet_type}")
    sps = int(samples_per_symbol)
    if sps < 4:
        raise ValueError("samples_per_symbol must be at least 4")
    direction_key = str(direction).upper()
    if direction_key not in {"RFP", "PP"}:
        raise ValueError("direction must be RFP or PP")
    symbol_count = PACKET_SYMBOL_COUNTS[packet_key]
    sync = RFP_SYNC_BITS if direction_key == "RFP" else PP_SYNC_BITS
    prolonged = sync[:16] if prolonged_preamble else np.empty(0, dtype=np.uint8)
    physical_bits = _apply_error_control(
        np.concatenate(
            (sync, _payload_bits(payload_pattern, symbol_count - sync.size, seed))
        ),
        packet_key,
    )
    bits = np.concatenate((prolonged, physical_bits))
    actual_symbol_rate_hz = DECT_SYMBOL_RATE_HZ * (
        1.0 + float(symbol_rate_error_ppm) * 1e-6
    )
    sample_rate_hz = DECT_SYMBOL_RATE_HZ * sps
    effective_sps = sample_rate_hz / actual_symbol_rate_hz
    packet_samples = max(1, int(round(bits.size * effective_sps)))
    # Generate on a fine symbol-time grid and sample it with the requested
    # timing-rate error so sub-sample clock offsets remain testable.
    reference_sps = max(64, 8 * sps)
    reference_levels = fsk_reference_frequency_levels(
        bits,
        samples_per_symbol=reference_sps,
        transmit_gaussian_bt=0.5,
    )
    source_axis = np.arange(reference_levels.size, dtype=np.float64) / reference_sps
    target_axis = np.arange(packet_samples, dtype=np.float64) / effective_sps
    shaped_levels = np.interp(target_axis, source_axis, reference_levels)

    padding = int(padding_symbols) * sps
    total = padding + packet_samples + padding
    instantaneous_frequency = np.zeros(total, dtype=np.float64)
    instantaneous_frequency[padding : padding + packet_samples] = (
        float(frequency_error_hz) + float(frequency_deviation_hz) * shaped_levels
    )
    instantaneous_frequency[:padding] = float(frequency_error_hz)
    instantaneous_frequency[padding + packet_samples :] = float(frequency_error_hz)
    phase = 2.0 * np.pi * np.concatenate(
        (np.zeros(1), np.cumsum(instantaneous_frequency[:-1]))
    ) / sample_rate_hz

    amplitude = np.zeros(total, dtype=np.float64)
    amplitude[padding : padding + packet_samples] = 1.0
    attack_samples = max(1, int(round(float(attack_time_s) * sample_rate_hz)))
    release_samples = max(1, int(round(float(release_time_s) * sample_rate_hz)))
    attack_start = max(0, padding - attack_samples)
    amplitude[attack_start:padding] = np.linspace(
        0.0, 1.0, padding - attack_start, endpoint=False
    )
    release_stop = min(total, padding + packet_samples + release_samples)
    amplitude[padding + packet_samples : release_stop] = np.linspace(
        1.0, 0.0, release_stop - padding - packet_samples, endpoint=False
    )
    amplitude *= 10.0 ** (float(power_dbm) / 20.0)
    iq = (amplitude * np.exp(1j * phase)).astype(np.complex64)
    return IQRecording(
        iq=iq,
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=float(center_frequency_hz),
        usable_bandwidth_hz=0.8 * sample_rate_hz,
        source="Generated DECT",
        full_scale=1.0,
        amplitude_calibrated=True,
        metadata={
            "generated_protocol": "DECT",
            "generated_bits": bits,
            "direction": direction_key,
            "packet_type": packet_key,
            "payload_pattern": payload_pattern,
            "preamble_mode": "Prolonged" if prolonged_preamble else "Normal",
            "expected_p0_sample": padding + prolonged.size * effective_sps,
            "expected_packet_stop_sample": padding + packet_samples,
            "dc_removal_recommended": False,
        },
    )
