"""Deterministic Bluetooth EDR DH1 waveforms for VSA development."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pluto_sa.vsa.mapping import BLUETOOTH_EDR_MAPPING, phase_indices_to_logical_symbols
from pluto_sa.vsa.model import IQRecording, ModulationKind
from pluto_sa.vsa.profiles.bluetooth_br import (
    access_code_bits,
    fec13_encode,
    header_error_check,
    modulate_packet_bits,
    payload_crc_bytes,
    prbs9_period,
    whitening_sequence,
)


EDR_SYNC_BITS_2MBPS = np.asarray(
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    dtype=np.uint8,
)
EDR_SYNC_BITS_3MBPS = np.asarray(
    [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1,
     1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class BluetoothEDRWaveform:
    recording: IQRecording
    packet_name: str
    modulation: ModulationKind
    packet_type: int
    payload_length_bytes: int
    access_bits: np.ndarray
    header_air_bits: np.ndarray
    sync_bits: np.ndarray
    payload_header_bits: np.ndarray
    payload_body_bits: np.ndarray
    payload_crc_bits: np.ndarray
    payload_air_bits: np.ndarray
    trailer_bits: np.ndarray
    differential_phase_indices: np.ndarray
    logical_symbols: np.ndarray
    packet_start_sample: int
    gfsk_stop_sample: int
    edr_start_sample: int
    packet_stop_sample: int


def _bits_lsb(value: int, width: int) -> np.ndarray:
    return np.asarray([(int(value) >> index) & 1 for index in range(width)], dtype=np.uint8)


def _bytes_to_air_bits(values: bytes) -> np.ndarray:
    return np.asarray(
        [(byte >> index) & 1 for byte in values for index in range(8)],
        dtype=np.uint8,
    )


def _header_air_bits(
    *, packet_type: int, clock_6_1: int, uap: int, lt_addr: int = 1
) -> np.ndarray:
    packed = int(lt_addr) | (int(packet_type) << 3) | (1 << 7)
    data = _bits_lsb(packed, 10)
    hec = header_error_check(data, int(uap))
    hec_bits = np.asarray([(hec >> shift) & 1 for shift in range(7, -1, -1)], dtype=np.uint8)
    header = np.concatenate((data, hec_bits))
    whitened = header ^ whitening_sequence(int(clock_6_1), header.size)
    return fec13_encode(whitened)


def _payload_header_bits(payload_length_bytes: int) -> np.ndarray:
    # LLID=2 (ACL-U start/no fragmentation), FLOW=1, 10-bit length, reserved=0.
    packed = 0b10 | (1 << 2) | (int(payload_length_bytes) << 3)
    return _bits_lsb(packed, 16)


def _phase_indices(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    groups = np.asarray(bits, dtype=np.uint8).reshape(-1, int(bits_per_symbol))
    if int(bits_per_symbol) == 2:
        mapping = {
            (0, 0): 0,
            (0, 1): 1,
            (1, 1): 2,
            (1, 0): 3,
        }
    else:
        mapping = {
            (0, 0, 0): 0,
            (0, 0, 1): 1,
            (0, 1, 1): 2,
            (0, 1, 0): 3,
            (1, 1, 0): 4,
            (1, 1, 1): 5,
            (1, 0, 1): 6,
            (1, 0, 0): 7,
        }
    return np.asarray([mapping[tuple(map(int, group))] for group in groups], dtype=np.int16)


def edr_sync_symbols(bits_per_symbol: int) -> np.ndarray:
    """Return Bluetooth EDR sync phase indices in transmitted symbol order."""

    width = int(bits_per_symbol)
    if width == 2:
        bits = EDR_SYNC_BITS_2MBPS
    elif width == 3:
        bits = EDR_SYNC_BITS_3MBPS
    else:
        raise ValueError("bits_per_symbol must be 2 or 3")
    return _phase_indices(bits, width)


def _rrc_taps(samples_per_symbol: int, beta: float = 0.4, span_symbols: int = 10) -> np.ndarray:
    sps = int(samples_per_symbol)
    time = np.arange(-span_symbols * sps / 2, span_symbols * sps / 2 + 1) / sps
    taps = np.empty(time.size, dtype=np.float64)
    for index, value in enumerate(time):
        if np.isclose(value, 0.0):
            taps[index] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif np.isclose(abs(value), 1.0 / (4.0 * beta)):
            taps[index] = beta / np.sqrt(2.0) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            numerator = (
                np.sin(np.pi * value * (1.0 - beta))
                + 4.0 * beta * value * np.cos(np.pi * value * (1.0 + beta))
            )
            denominator = np.pi * value * (1.0 - (4.0 * beta * value) ** 2)
            taps[index] = numerator / denominator
    return taps / np.sqrt(np.sum(taps**2))


def _modulate_edr_symbols(
    phase_indices: np.ndarray, *, order: int, samples_per_symbol: int
) -> np.ndarray:
    phase_step = 2.0 * np.pi / int(order)
    phase_offset = np.pi / 4.0 if int(order) == 4 else 0.0
    changes = np.exp(
        1j
        * (
            phase_offset
            + phase_step * np.asarray(phase_indices, dtype=np.float64)
        )
    )
    symbols = np.concatenate(
        (np.ones(1, dtype=np.complex128), np.cumprod(changes))
    )
    span_symbols = 10
    padded = np.pad(symbols, (span_symbols, span_symbols), mode="edge")
    impulses = np.zeros(padded.size * int(samples_per_symbol), dtype=np.complex128)
    impulses[
        np.arange(padded.size) * int(samples_per_symbol) + int(samples_per_symbol) // 2
    ] = padded
    shaped = np.convolve(impulses, _rrc_taps(samples_per_symbol), mode="same")
    start = span_symbols * int(samples_per_symbol)
    result = shaped[start : start + symbols.size * int(samples_per_symbol)]
    result /= np.sqrt(np.mean(np.abs(result) ** 2))
    return result


def generate_edr_dh1(
    packet_name: str,
    *,
    sample_rate_hz: float = 16_000_000.0,
    center_frequency_hz: float = 2_441_000_000.0,
    lap: int = 0xC6967E,
    uap: int = 0x6B,
    clock_6_1: int = 0x2B,
    carrier_frequency_offset_hz: float = 20_000.0,
    duration_s: float = 0.003,
    packet_start_s: float = 0.002,
    snr_db: float = 35.0,
    seed: int = 1,
) -> BluetoothEDRWaveform:
    """Generate a maximum-length 2-DH1 or 3-DH1 packet at 1 MSym/s."""
    normalized = str(packet_name).upper()
    if normalized == "2-DH1":
        modulation = ModulationKind.PI4_DQPSK
        packet_type = 0x4
        payload_length = 54
        bits_per_symbol = 2
        sync_bits = EDR_SYNC_BITS_2MBPS
    elif normalized == "3-DH1":
        modulation = ModulationKind.DPSK8
        packet_type = 0x8
        payload_length = 83
        bits_per_symbol = 3
        sync_bits = EDR_SYNC_BITS_3MBPS
    else:
        raise ValueError("packet_name must be 2-DH1 or 3-DH1")
    samples_per_symbol = int(round(float(sample_rate_hz) / 1_000_000.0))
    if samples_per_symbol < 4 or not np.isclose(
        samples_per_symbol * 1_000_000.0, float(sample_rate_hz)
    ):
        raise ValueError("sample_rate_hz must be an integer multiple of 1 MHz")

    access = access_code_bits(int(lap))
    header_air = _header_air_bits(
        packet_type=packet_type,
        clock_6_1=int(clock_6_1),
        uap=int(uap),
    )
    gfsk = modulate_packet_bits(
        np.concatenate((access, header_air)),
        sample_rate_hz=float(sample_rate_hz),
    ).astype(np.complex128)

    payload_header = _payload_header_bits(payload_length)
    period = prbs9_period()
    body = period[np.arange(payload_length * 8) % period.size]
    crc = _bytes_to_air_bits(
        payload_crc_bytes(np.concatenate((payload_header, body)), int(uap))
    )
    payload = np.concatenate((payload_header, body, crc))
    whitening = whitening_sequence(int(clock_6_1), 18 + payload.size)
    payload_air = payload ^ whitening[18:]
    trailer = np.zeros(2 * bits_per_symbol, dtype=np.uint8)
    psk_bits = np.concatenate((sync_bits, payload_air, trailer))
    phase_indices = _phase_indices(psk_bits, bits_per_symbol)
    psk = _modulate_edr_symbols(
        phase_indices,
        order=2**bits_per_symbol,
        samples_per_symbol=samples_per_symbol,
    )
    psk *= np.exp(1j * (np.angle(gfsk[-1]) - np.angle(psk[0])))

    guard = np.full(5 * samples_per_symbol, gfsk[-1], dtype=np.complex128)
    packet = np.concatenate((gfsk, guard, psk))
    total_samples = int(round(float(duration_s) * float(sample_rate_hz)))
    packet_start = int(round(float(packet_start_s) * float(sample_rate_hz)))
    if packet_start < 0 or packet_start + packet.size > total_samples:
        raise ValueError("packet does not fit inside the requested recording")
    baseband = np.zeros(total_samples, dtype=np.complex128)
    baseband[packet_start : packet_start + packet.size] = packet
    sample_index = np.arange(total_samples, dtype=np.float64)
    carrier = np.exp(
        2j * np.pi * float(carrier_frequency_offset_hz) * sample_index / float(sample_rate_hz)
    )
    amplitude = 0.25
    iq = amplitude * baseband * carrier
    rng = np.random.default_rng(int(seed))
    noise_sigma = amplitude * np.sqrt(0.5 / 10.0 ** (float(snr_db) / 10.0))
    iq += noise_sigma * (
        rng.standard_normal(total_samples) + 1j * rng.standard_normal(total_samples)
    )
    gfsk_stop = packet_start + gfsk.size
    edr_start = gfsk_stop + guard.size
    packet_stop = packet_start + packet.size
    recording = IQRecording(
        iq=iq.astype(np.complex64),
        sample_rate_hz=float(sample_rate_hz),
        center_frequency_hz=float(center_frequency_hz),
        usable_bandwidth_hz=float(sample_rate_hz),
        source=f"Generated Bluetooth {normalized}",
        amplitude_calibrated=True,
        metadata={
            "packet_name": normalized,
            "lap": int(lap),
            "uap": int(uap),
            "clock_6_1": int(clock_6_1),
            "carrier_frequency_offset_hz": float(carrier_frequency_offset_hz),
            "tx_filter": "Root Raised Cosine",
            "rolloff": 0.4,
            "seed": int(seed),
        },
    )
    return BluetoothEDRWaveform(
        recording=recording,
        packet_name=normalized,
        modulation=modulation,
        packet_type=packet_type,
        payload_length_bytes=payload_length,
        access_bits=access,
        header_air_bits=header_air,
        sync_bits=np.array(sync_bits, copy=True),
        payload_header_bits=payload_header,
        payload_body_bits=body,
        payload_crc_bits=crc,
        payload_air_bits=payload_air,
        trailer_bits=trailer,
        differential_phase_indices=phase_indices,
        logical_symbols=phase_indices_to_logical_symbols(
            modulation, BLUETOOTH_EDR_MAPPING, phase_indices
        ),
        packet_start_sample=packet_start,
        gfsk_stop_sample=gfsk_stop,
        edr_start_sample=edr_start,
        packet_stop_sample=packet_stop,
    )
