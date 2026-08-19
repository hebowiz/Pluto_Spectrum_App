"""Deterministic maximum-length Bluetooth Basic Rate DH1 waveform."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pluto_sa.vsa.model import IQRecording, ModulationKind
from pluto_sa.vsa.profiles.bluetooth_br import (
    access_code_bits,
    build_packet_bits,
    modulate_packet_bits,
    payload_crc_bytes,
    prbs9_period,
)


@dataclass(frozen=True)
class BluetoothBRDH1Waveform:
    recording: IQRecording
    packet_name: str
    modulation: ModulationKind
    packet_type: int
    payload_length_bytes: int
    packet_bits: np.ndarray
    access_bits: np.ndarray
    header_air_bits: np.ndarray
    payload_header_bits: np.ndarray
    payload_body_bits: np.ndarray
    payload_crc_bits: np.ndarray
    payload_air_bits: np.ndarray
    packet_start_sample: int
    packet_stop_sample: int


def _bits_lsb(value: int, width: int) -> np.ndarray:
    return np.asarray(
        [(int(value) >> index) & 1 for index in range(int(width))],
        dtype=np.uint8,
    )


def _bytes_to_air_bits(values: bytes) -> np.ndarray:
    return np.asarray(
        [(byte >> index) & 1 for byte in values for index in range(8)],
        dtype=np.uint8,
    )


def generate_br_dh1(
    *,
    sample_rate_hz: float = 16_000_000.0,
    center_frequency_hz: float = 2_441_000_000.0,
    lap: int = 0xC6967E,
    uap: int = 0x6B,
    clock_6_1: int = 0x2B,
    carrier_frequency_offset_hz: float = 20_000.0,
    frequency_deviation_hz: float = 160_000.0,
    duration_s: float = 0.003,
    packet_start_s: float = 0.002,
    snr_db: float = 35.0,
    seed: int = 11,
) -> BluetoothBRDH1Waveform:
    """Generate a standards-structured, maximum-payload DH1 at 1 MSym/s."""
    samples_per_symbol = int(round(float(sample_rate_hz) / 1_000_000.0))
    if samples_per_symbol < 4 or not np.isclose(
        samples_per_symbol * 1_000_000.0, float(sample_rate_hz)
    ):
        raise ValueError("sample_rate_hz must be an integer multiple of 1 MHz")

    payload_length = 27
    # LLID=2 (ACL-U start/no fragmentation), FLOW=1, 5-bit length=27.
    payload_header = _bits_lsb(0b10 | (1 << 2) | (payload_length << 3), 8)
    period = prbs9_period()
    payload_body = period[np.arange(payload_length * 8) % period.size]
    payload_crc = _bytes_to_air_bits(
        payload_crc_bytes(
            np.concatenate((payload_header, payload_body)), int(uap)
        )
    )
    payload = np.concatenate((payload_header, payload_body, payload_crc))
    packet_bits = build_packet_bits(
        clock_6_1=int(clock_6_1),
        uap=int(uap),
        payload_bits=payload,
        lap=int(lap),
        packet_type=0x4,
    )
    access = access_code_bits(int(lap))
    header_stop = access.size + 54
    payload_air = packet_bits[header_stop:]

    total_samples = int(round(float(duration_s) * float(sample_rate_hz)))
    packet_start = int(round(float(packet_start_s) * float(sample_rate_hz)))
    packet_samples = packet_bits.size * samples_per_symbol
    packet_stop = packet_start + packet_samples
    if packet_start < 0 or packet_stop > total_samples:
        raise ValueError("packet does not fit inside the requested recording")
    baseband = modulate_packet_bits(
        packet_bits,
        sample_rate_hz=float(sample_rate_hz),
        frequency_deviation_hz=float(frequency_deviation_hz),
        carrier_frequency_offset_hz=float(carrier_frequency_offset_hz),
        prefix_samples=packet_start,
        suffix_samples=total_samples - packet_stop,
    ).astype(np.complex128)
    amplitude = 0.25
    rng = np.random.default_rng(int(seed))
    noise_sigma = amplitude * np.sqrt(0.5 / 10.0 ** (float(snr_db) / 10.0))
    iq = amplitude * baseband + noise_sigma * (
        rng.standard_normal(total_samples) + 1j * rng.standard_normal(total_samples)
    )
    recording = IQRecording(
        iq=iq.astype(np.complex64),
        sample_rate_hz=float(sample_rate_hz),
        center_frequency_hz=float(center_frequency_hz),
        usable_bandwidth_hz=float(sample_rate_hz),
        source="Generated Bluetooth DH1",
        amplitude_calibrated=True,
        metadata={
            "packet_name": "DH1",
            "lap": int(lap),
            "uap": int(uap),
            "clock_6_1": int(clock_6_1),
            "carrier_frequency_offset_hz": float(carrier_frequency_offset_hz),
            "frequency_deviation_hz": float(frequency_deviation_hz),
            "tx_filter": "Gaussian",
            "bt": 0.5,
            "seed": int(seed),
        },
    )
    return BluetoothBRDH1Waveform(
        recording=recording,
        packet_name="DH1",
        modulation=ModulationKind.FSK,
        packet_type=0x4,
        payload_length_bytes=payload_length,
        packet_bits=packet_bits,
        access_bits=access,
        header_air_bits=packet_bits[access.size:header_stop],
        payload_header_bits=payload_header,
        payload_body_bits=payload_body,
        payload_crc_bits=payload_crc,
        payload_air_bits=payload_air,
        packet_start_sample=packet_start,
        packet_stop_sample=packet_stop,
    )
