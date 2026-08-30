"""Bluetooth LE 1M/2M uncoded RF test packet waveform generation."""

from __future__ import annotations

import math

import numpy as np

from pluto_protocol.model import GeneratedPacketBits
from pluto_sa.vsa.profiles.bluetooth_br import prbs9_period
from pluto_vsg.engine.base import FieldBoundary, GenerationResult
from pluto_vsg.engine.bluetooth_br import (
    _append_field_boundaries,
    _extend_edge_phase,
    _modulate_gfsk,
    _placed_power_envelope,
)
from pluto_vsg.model import (
    BluetoothLEPayloadSourceKind,
    BluetoothLEPayloadType,
    BluetoothLEPhy,
    WaveformProject,
    bluetooth_le_payload_code,
    waveform_timing_samples,
    validate_project,
)


_SYNC_WORD = "10010100100000100110111010001110"
_FIXED_PAYLOADS = {
    BluetoothLEPayloadType.F0: "11110000",
    BluetoothLEPayloadType.AA: "10101010",
    BluetoothLEPayloadType.FF: "11111111",
    BluetoothLEPayloadType.ZERO: "00000000",
    BluetoothLEPayloadType.OF: "00001111",
    BluetoothLEPayloadType.FIVE: "01010101",
}


def _bits_lsb(value: int, width: int) -> np.ndarray:
    return np.asarray([(int(value) >> index) & 1 for index in range(width)], dtype=np.uint8)


def _prbs15_period() -> np.ndarray:
    register = np.ones(15, dtype=np.uint8)
    sequence = np.empty((1 << 15) - 1, dtype=np.uint8)
    for index in range(sequence.size):
        sequence[index] = register[-1]
        feedback = register[-2] ^ register[-1]
        register[1:] = register[:-1]
        register[0] = feedback
    return sequence


def le_test_payload_bits(
    payload_type: BluetoothLEPayloadType | str, length_bytes: int
) -> np.ndarray:
    payload_type = BluetoothLEPayloadType(payload_type)
    count = int(length_bytes) * 8
    if count == 0:
        return np.empty(0, dtype=np.uint8)
    if payload_type == BluetoothLEPayloadType.PRBS9:
        sequence = prbs9_period()
    elif payload_type == BluetoothLEPayloadType.PRBS15:
        sequence = _prbs15_period()
    else:
        sequence = np.asarray(
            [int(bit) for bit in _FIXED_PAYLOADS[payload_type]], dtype=np.uint8
        )
    return sequence[np.arange(count) % sequence.size]


def le_payload_bits(project: WaveformProject) -> np.ndarray:
    settings = project.bluetooth_le
    if settings is None:
        raise ValueError("Bluetooth LE settings are required")
    count = int(settings.payload_length_bytes) * 8
    if count == 0:
        return np.empty(0, dtype=np.uint8)
    source = BluetoothLEPayloadSourceKind(settings.payload_source)
    if source == BluetoothLEPayloadSourceKind.PRBS9:
        sequence = prbs9_period()
    elif source == BluetoothLEPayloadSourceKind.PRBS15:
        sequence = _prbs15_period()
    else:
        pattern = settings.payload_pattern.strip().replace(" ", "")
        if source == BluetoothLEPayloadSourceKind.FIXED:
            pattern = pattern[:1]
        sequence = np.asarray([int(bit) for bit in pattern], dtype=np.uint8)
    return sequence[np.arange(count) % sequence.size]


def le_whitening_sequence(channel_index: int, count: int) -> np.ndarray:
    """Generate LE whitening bits using Core register positions 0 through 6."""

    channel = int(channel_index)
    register = np.asarray(
        [1] + [(channel >> index) & 1 for index in range(5, -1, -1)],
        dtype=np.uint8,
    )
    output = np.empty(int(count), dtype=np.uint8)
    for index in range(output.size):
        feedback = int(register[6])
        output[index] = feedback
        previous = register.copy()
        register[0] = feedback
        register[1] = previous[0]
        register[2] = previous[1]
        register[3] = previous[2]
        register[4] = previous[3] ^ feedback
        register[5] = previous[4]
        register[6] = previous[5]
    return output


def le_crc24_bits(bits: np.ndarray, init: int = 0x555555) -> np.ndarray:
    """Return the LE CRC in transmission order."""

    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("bits must be a one-dimensional binary array")
    if not 0 <= int(init) <= 0xFFFFFF:
        raise ValueError("init must be a 24-bit value")

    # Core Vol 6, Part B, 3.1.1: process PDU bits in transmitted order with
    # x^24 + x^10 + x^9 + x^6 + x^4 + x^3 + x + 1.  Position 23 supplies
    # feedback, the register shifts toward the MSB and 0x00065B represents
    # the polynomial terms below x^24.
    register = int(init)
    for bit in values:
        feedback = int(bit) ^ ((register >> 23) & 1)
        register = (register << 1) & 0xFFFFFF
        if feedback:
            register ^= 0x00065B

    # CRC register positions 23 down to 0 are transmitted in this order.
    return np.asarray(
        [(register >> position) & 1 for position in range(23, -1, -1)],
        dtype=np.uint8,
    )


class BluetoothLEWaveformEngine:
    """Generate normalized LE 1M/2M Direct Test Mode packets."""

    def generate(self, project: WaveformProject) -> GenerationResult:
        issues = validate_project(project)
        if issues:
            details = "; ".join(f"{item.path}: {item.message}" for item in issues)
            raise ValueError(f"Invalid waveform project: {details}")
        settings = project.bluetooth_le
        if settings is None:
            raise ValueError("Bluetooth LE settings are required")

        phy = BluetoothLEPhy(settings.phy)
        symbol_rate_hz = 1_000_000.0 if phy == BluetoothLEPhy.LE_1M else 2_000_000.0
        expected_rate = symbol_rate_hz * int(project.samples_per_symbol)
        if not np.isclose(project.sample_rate_hz, expected_rate):
            raise ValueError("LE sample rate must equal symbol rate times samples per symbol")

        preamble = settings.preamble_bits.strip().replace(" ", "")
        preamble_bits = np.asarray([int(bit) for bit in preamble], dtype=np.uint8)
        sync_bits = np.asarray(
            [int(bit) for bit in settings.sync_word_bits.strip().replace(" ", "")],
            dtype=np.uint8,
        )
        header_bits = np.asarray(
            [int(bit) for bit in settings.pdu_header_bits.strip().replace(" ", "")],
            dtype=np.uint8,
        )
        length_bits = _bits_lsb(settings.payload_length_bytes, 8)
        payload_bits = le_payload_bits(project)
        pdu_bits = np.concatenate((header_bits, length_bits, payload_bits))
        crc_bits = (
            le_crc24_bits(pdu_bits, settings.crc_init)
            if settings.crc_enabled
            else np.empty(0, dtype=np.uint8)
        )
        pdu_crc_bits = np.concatenate((pdu_bits, crc_bits))
        whitening_bits = (
            le_whitening_sequence(settings.whitening_channel_index, pdu_crc_bits.size)
            if settings.whitening_enabled
            else np.zeros(pdu_crc_bits.size, dtype=np.uint8)
        )
        pdu_crc_air_bits = pdu_crc_bits ^ whitening_bits
        packet_bits = np.concatenate((preamble_bits, sync_bits, pdu_crc_air_bits))

        sps = int(project.samples_per_symbol)
        data_iq = _modulate_gfsk(
            packet_bits,
            samples_per_symbol=sps,
            sample_rate_hz=project.sample_rate_hz,
            deviation_hz=settings.frequency_deviation_hz,
            gaussian_bt=settings.gaussian_bt,
        )
        data_sample_count = int(data_iq.size)
        envelope = project.power_envelope
        if envelope.enabled:
            rise_count = round(envelope.rise_symbols * sps)
            fall_count = round(envelope.fall_symbols * sps)
            rise_start = round(envelope.rise_delay_symbols * sps)
            fall_start = data_sample_count + round(envelope.fall_delay_symbols * sps)
            active_start = min(0, rise_start)
            active_stop = max(data_sample_count, fall_start + fall_count)
        else:
            rise_count = fall_count = 0
            rise_start = 0
            fall_start = data_sample_count
            active_start = 0
            active_stop = data_sample_count
        positions = np.arange(active_start, active_stop, dtype=np.int64)
        active_iq = _extend_edge_phase(data_iq, positions)
        if envelope.enabled:
            active_iq *= _placed_power_envelope(
                positions,
                rise_start=rise_start,
                rise_samples=rise_count,
                fall_start=fall_start,
                fall_samples=fall_count,
                shape=envelope.shape,
            )

        prefix_count = int(settings.pre_idle_symbols) * sps
        _, _, minimum_period_count, period_count = waveform_timing_samples(project)
        if period_count < minimum_period_count:
            raise ValueError("Packet period is shorter than the generated burst")
        nominal_packet_us = packet_bits.size / symbol_rate_hz * 1e6
        interval_us = None
        if settings.rf_test_interval_enabled and project.period_symbols is None:
            interval_us = math.ceil((nominal_packet_us + 249.0) / 625.0) * 625.0
            target_count = round(interval_us * 1e-6 * project.sample_rate_hz)
            period_count = max(period_count, target_count)
        suffix_count = period_count - prefix_count - active_iq.size
        single = np.concatenate(
            (
                np.zeros(prefix_count, dtype=np.complex128),
                active_iq,
                np.zeros(suffix_count, dtype=np.complex128),
            )
        )
        iq = np.tile(single, int(project.repeat_count)).astype(np.complex64)
        data_start = prefix_count - active_start

        boundaries: list[FieldBoundary] = []
        packet_ranges: list[tuple[int, int]] = []
        for repeat in range(int(project.repeat_count)):
            offset = repeat * single.size + data_start
            packet_ranges.append((offset, offset + data_sample_count))
            _append_field_boundaries(
                boundaries,
                project.fields,
                repeat_offset=offset,
                samples_per_symbol=sps,
                repeat_suffix=("" if project.repeat_count == 1 else f" [{repeat + 1}]"),
            )

        return GenerationResult(
            iq=iq,
            sample_rate_hz=project.sample_rate_hz,
            field_boundaries=tuple(boundaries),
            packet_bits=GeneratedPacketBits(
                bits=packet_bits,
                protocol_id="bluetooth.le",
                phy_name=phy.value,
                context={
                    "phy": phy.value,
                    "whitening_enabled": bool(settings.whitening_enabled),
                    "whitening_channel_index": int(settings.whitening_channel_index),
                    "crc_enabled": bool(settings.crc_enabled),
                    "crc_init": int(settings.crc_init),
                },
            ),
            metadata={
                "project_name": project.name,
                "standard": project.standard.value,
                "center_frequency_hz": project.center_frequency_hz,
                "packet_name": (
                    f"{phy.value} RF Test Packet"
                    if settings.rf_test_interval_enabled
                    else f"{phy.value} Packet"
                ),
                "phy": phy.value,
                "payload_type": BluetoothLEPayloadType(settings.payload_type).value,
                "payload_length_bytes": int(settings.payload_length_bytes),
                "preamble_bits": preamble_bits,
                "sync_word_bits": sync_bits,
                "pdu_header_bits": header_bits,
                "pdu_length_bits": length_bits,
                "payload_bits": payload_bits,
                "crc_bits": crc_bits,
                "pdu_crc_air_bits": pdu_crc_air_bits,
                "whitening_bits": whitening_bits,
                "packet_bits": packet_bits,
                "packet_sample_count": data_sample_count,
                "period_sample_count": single.size,
                "period_symbols": single.size / sps,
                "post_idle_sample_count": suffix_count,
                "packet_ranges_samples": tuple(packet_ranges),
                "data_start_sample": data_start,
                "data_stop_sample": data_start + data_sample_count,
                "samples_per_symbol": sps,
                "symbol_rate_hz": symbol_rate_hz,
                "frequency_deviation_hz": settings.frequency_deviation_hz,
                "gaussian_bt": settings.gaussian_bt,
                "test_packet_interval_us": interval_us,
                "whitening_enabled": bool(settings.whitening_enabled),
                "whitening_channel_index": int(settings.whitening_channel_index),
                "crc_enabled": bool(settings.crc_enabled),
                "crc_init": int(settings.crc_init),
            },
        )
