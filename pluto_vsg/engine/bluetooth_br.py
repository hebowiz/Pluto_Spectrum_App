"""Bluetooth Basic Rate DH1 waveform generation for Pluto VSG."""

from __future__ import annotations

import numpy as np

from pluto_sa.vsa.demod.fsk_reference import fsk_reference_frequency_levels
from pluto_sa.vsa.profiles.bluetooth_br import (
    access_code_bits,
    build_packet_bits,
    fec13_encode,
    header_error_check,
    payload_crc_bytes,
    prbs9_period,
)
from pluto_vsg.engine.base import FieldBoundary, GenerationResult
from pluto_vsg.model import (
    FieldDefinition,
    PayloadSourceKind,
    WaveformProject,
    validate_project,
)


_BR_SYMBOL_RATE_HZ = 1_000_000.0


def _append_field_boundaries(
    destination: list[FieldBoundary],
    fields: tuple[FieldDefinition, ...],
    *,
    repeat_offset: int,
    samples_per_symbol: int,
    start_symbol: int = 0,
    level: int = 0,
    parent_name: str | None = None,
    repeat_suffix: str = "",
) -> int:
    """Expand one logical field tree into transmitted-symbol/sample spans."""

    cursor = int(start_symbol)
    for packet_field in fields:
        stop = cursor + int(packet_field.symbol_count)
        destination.append(
            FieldBoundary(
                name=f"{packet_field.name}{repeat_suffix}",
                start_sample=repeat_offset + cursor * samples_per_symbol,
                stop_sample=repeat_offset + stop * samples_per_symbol,
                start_symbol=cursor,
                stop_symbol=stop,
                logical_bit_count=packet_field.logical_bit_count,
                level=level,
                parent_name=parent_name,
            )
        )
        if packet_field.children:
            child_stop = _append_field_boundaries(
                destination,
                packet_field.children,
                repeat_offset=repeat_offset,
                samples_per_symbol=samples_per_symbol,
                start_symbol=cursor,
                level=level + 1,
                parent_name=packet_field.name,
                repeat_suffix=repeat_suffix,
            )
            if child_stop != stop:
                raise ValueError(
                    f"Field hierarchy for {packet_field.name} does not fill its span"
                )
        cursor = stop
    return cursor


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


def _payload_body(project: WaveformProject) -> np.ndarray:
    settings = project.bluetooth_br
    assert settings is not None
    count = settings.payload_length_bytes * 8
    if count == 0:
        return np.empty(0, dtype=np.uint8)
    source = PayloadSourceKind(settings.payload_source)
    if source == PayloadSourceKind.PRBS9:
        sequence = prbs9_period()
    else:
        pattern = settings.payload_pattern.strip().replace(" ", "")
        sequence = np.asarray([int(bit) for bit in pattern], dtype=np.uint8)
    if source == PayloadSourceKind.FIXED:
        return np.full(count, int(sequence[0]), dtype=np.uint8)
    return sequence[np.arange(count) % sequence.size]


def _header_data_bits(project: WaveformProject) -> np.ndarray:
    settings = project.bluetooth_br
    assert settings is not None
    packed = (
        int(settings.lt_addr)
        | (0x4 << 3)
        | (int(settings.flow) << 7)
        | (int(settings.arqn) << 8)
        | (int(settings.seqn) << 9)
    )
    return _bits_lsb(packed, 10)


def _unwhitened_packet_bits(
    project: WaveformProject, payload_bits: np.ndarray
) -> np.ndarray:
    settings = project.bluetooth_br
    assert settings is not None
    header_data = _header_data_bits(project)
    hec = header_error_check(header_data, settings.uap)
    hec_bits_msb = np.asarray([int(bit) for bit in f"{hec:08b}"], dtype=np.uint8)
    header_air = fec13_encode(np.concatenate((header_data, hec_bits_msb)))
    return np.concatenate(
        (access_code_bits(settings.lap), header_air, payload_bits)
    )


def _ramp_curve(sample_count: int, *, rising: bool, shape: str) -> np.ndarray:
    count = max(0, int(sample_count))
    if count == 0:
        return np.empty(0, dtype=np.float64)
    if count == 1:
        return np.asarray([1.0 if rising else 0.0])
    position = np.linspace(0.0, 1.0, count)
    curve = (
        position
        if shape.lower() == "linear"
        else 0.5 - 0.5 * np.cos(np.pi * position)
    )
    return curve if rising else curve[::-1]


def _placed_power_envelope(
    positions: np.ndarray,
    *,
    rise_start: int,
    rise_samples: int,
    fall_start: int,
    fall_samples: int,
    shape: str,
) -> np.ndarray:
    coordinates = np.asarray(positions, dtype=np.int64)
    envelope = np.ones(coordinates.size, dtype=np.float64)
    rise_count = max(0, int(rise_samples))
    fall_count = max(0, int(fall_samples))
    if rise_count:
        rise_response = np.zeros(coordinates.size, dtype=np.float64)
        rise_response[coordinates >= rise_start + rise_count] = 1.0
        inside = (coordinates >= rise_start) & (
            coordinates < rise_start + rise_count
        )
        curve = _ramp_curve(rise_count, rising=True, shape=shape)
        rise_response[inside] = curve[coordinates[inside] - rise_start]
        envelope *= rise_response
    if fall_count:
        fall_response = np.ones(coordinates.size, dtype=np.float64)
        fall_response[coordinates >= fall_start + fall_count] = 0.0
        inside = (coordinates >= fall_start) & (
            coordinates < fall_start + fall_count
        )
        curve = _ramp_curve(fall_count, rising=False, shape=shape)
        fall_response[inside] = curve[coordinates[inside] - fall_start]
        envelope *= fall_response
    return envelope


class BluetoothBRWaveformEngine:
    """Generate a normalized, standards-structured BR DH1 waveform."""

    def generate(self, project: WaveformProject) -> GenerationResult:
        issues = validate_project(project)
        if issues:
            details = "; ".join(f"{item.path}: {item.message}" for item in issues)
            raise ValueError(f"Invalid waveform project: {details}")
        settings = project.bluetooth_br
        if settings is None:
            raise ValueError("Bluetooth BR settings are required")
        samples_per_symbol = int(project.samples_per_symbol)
        expected_rate = _BR_SYMBOL_RATE_HZ * samples_per_symbol
        if not np.isclose(project.sample_rate_hz, expected_rate):
            raise ValueError(
                "Bluetooth BR sample rate must equal 1 MSym/s multiplied by "
                "samples per symbol"
            )

        body = _payload_body(project)
        payload_header = _bits_lsb(
            0b10 | (1 << 2) | (settings.payload_length_bytes << 3), 8
        )
        payload_crc = _bytes_to_air_bits(
            payload_crc_bytes(np.concatenate((payload_header, body)), settings.uap)
        )
        payload = np.concatenate((payload_header, body, payload_crc))
        if settings.whitening_enabled:
            packet_bits = build_packet_bits(
                clock_6_1=settings.clock_6_1,
                uap=settings.uap,
                payload_bits=payload,
                lt_addr=settings.lt_addr,
                packet_type=0x4,
                flow=settings.flow,
                arqn=settings.arqn,
                seqn=settings.seqn,
                lap=settings.lap,
            )
        else:
            packet_bits = _unwhitened_packet_bits(project, payload)

        frequency_levels = fsk_reference_frequency_levels(
            packet_bits,
            samples_per_symbol=samples_per_symbol,
            transmit_gaussian_bt=settings.gaussian_bt,
        )
        data_sample_count = frequency_levels.size
        if project.power_envelope.enabled:
            rise_count = round(
                project.power_envelope.rise_symbols * samples_per_symbol
            )
            fall_count = round(
                project.power_envelope.fall_symbols * samples_per_symbol
            )
            rise_start = round(
                project.power_envelope.rise_delay_symbols * samples_per_symbol
            )
            fall_start = data_sample_count + round(
                project.power_envelope.fall_delay_symbols * samples_per_symbol
            )
            active_start = min(0, rise_start)
            active_stop = max(data_sample_count, fall_start + fall_count)
        else:
            rise_count = fall_count = 0
            rise_start = 0
            fall_start = data_sample_count
            active_start = 0
            active_stop = data_sample_count
        active_positions = np.arange(active_start, active_stop, dtype=np.int64)
        # The Gaussian reference uses nearest-edge extension. Apply the same
        # explicit rule while the burst envelope exists outside packet data.
        active_levels = frequency_levels[
            np.clip(active_positions, 0, data_sample_count - 1)
        ]
        packet_time_s = np.arange(active_levels.size) / project.sample_rate_hz
        frequency_hz = (
            active_levels * settings.frequency_deviation_hz
            + settings.carrier_frequency_offset_hz
        )
        phase = 2.0 * np.pi * np.cumsum(frequency_hz) / project.sample_rate_hz
        active_iq = np.exp(1j * phase)
        if project.power_envelope.enabled:
            active_iq *= _placed_power_envelope(
                active_positions,
                rise_start=rise_start,
                rise_samples=rise_count,
                fall_start=fall_start,
                fall_samples=fall_count,
                shape=project.power_envelope.shape,
            )

        prefix_count = settings.pre_idle_symbols * samples_per_symbol
        suffix_count = settings.post_idle_symbols * samples_per_symbol
        prefix = np.zeros(prefix_count, dtype=np.complex128)
        suffix = np.zeros(suffix_count, dtype=np.complex128)
        single = np.concatenate((prefix, active_iq, suffix))
        iq = np.tile(single, project.repeat_count).astype(np.complex64)
        data_start_in_single = prefix_count - active_start

        project_symbol_count = sum(
            packet_field.symbol_count for packet_field in project.fields
        )
        if project_symbol_count != packet_bits.size:
            raise ValueError(
                "Composer field symbols do not match the generated packet "
                f"({project_symbol_count} != {packet_bits.size})"
            )
        boundaries: list[FieldBoundary] = []
        for repeat in range(project.repeat_count):
            repeat_offset = repeat * single.size + data_start_in_single
            suffix_name = "" if project.repeat_count == 1 else f" [{repeat + 1}]"
            stop_symbol = _append_field_boundaries(
                boundaries,
                project.fields,
                repeat_offset=repeat_offset,
                samples_per_symbol=samples_per_symbol,
                repeat_suffix=suffix_name,
            )
            if stop_symbol != packet_bits.size:
                raise ValueError("Field hierarchy does not fill the generated packet")

        return GenerationResult(
            iq=iq,
            sample_rate_hz=project.sample_rate_hz,
            field_boundaries=tuple(boundaries),
            metadata={
                "project_name": project.name,
                "standard": project.standard.value,
                "center_frequency_hz": project.center_frequency_hz,
                "packet_name": "DH1",
                "packet_bits": packet_bits,
                "payload_header_bits": payload_header,
                "payload_body_bits": body,
                "payload_crc_bits": payload_crc,
                "samples_per_symbol": samples_per_symbol,
                "symbol_rate_hz": _BR_SYMBOL_RATE_HZ,
                "frequency_deviation_hz": settings.frequency_deviation_hz,
                "gaussian_bt": settings.gaussian_bt,
                "packet_sample_count": int(data_sample_count),
                "active_sample_count": int(active_iq.size),
                "data_start_sample": int(data_start_in_single),
                "data_stop_sample": int(data_start_in_single + data_sample_count),
                "ramp_rise_start_relative_samples": int(rise_start),
                "ramp_fall_start_relative_samples": int(
                    fall_start - data_sample_count
                ),
                "edge_frequency_mode": "Hold first / last symbol",
            },
        )
