"""Classic DECT 2-level GFSK waveform generation."""

from __future__ import annotations

import numpy as np

from pluto_protocol.dect.common import r_crc_bits, x_crc_bits
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
    DectPacketType,
    PayloadSourceKind,
    WaveformProject,
    waveform_timing_samples,
    validate_project,
)
from pluto_vsg.profiles.dect import (
    DECT_B_FIELD_BITS,
    DECT_SYMBOL_RATE_HZ,
    dect_s_field_text,
)
from pluto_vsg.rf_level import iq_level_metadata, measure_iq_levels


def _bits(text: str) -> np.ndarray:
    normalized = str(text).replace(" ", "").replace("_", "")
    return np.asarray([int(bit) for bit in normalized], dtype=np.uint8)


def _b_field_bits(project: WaveformProject, count: int) -> np.ndarray:
    settings = project.dect
    assert settings is not None
    source = PayloadSourceKind(settings.b_field_source)
    if source is PayloadSourceKind.PRBS9:
        sequence = prbs9_period()
    else:
        sequence = _bits(settings.b_field_pattern)
    if source is PayloadSourceKind.FIXED:
        return np.full(count, int(sequence[0]), dtype=np.uint8)
    return sequence[np.arange(count) % sequence.size]


def _x_test_data(packet_type: DectPacketType, b_bits: np.ndarray) -> np.ndarray:
    if packet_type in {DectPacketType.P32, DectPacketType.P32Z}:
        indices = [index + 48 * (1 + index // 16) for index in range(80)]
    else:
        indices = [index + 64 * (1 + index // 16) for index in range(160)]
    return np.asarray(b_bits[indices], dtype=np.uint8)


def dect_packet_bits(project: WaveformProject) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compile editable DECT settings into the exact transmitted bit stream."""

    settings = project.dect
    if settings is None:
        raise ValueError("DECT settings are required")
    packet_type = DectPacketType(settings.packet_type)
    s_field = dect_s_field_text(settings)
    preamble = _bits(s_field[:16])
    sync_word = _bits(s_field[16:])
    header = _bits(settings.a_header_bits)
    tail = _bits(settings.a_tail_bits)
    a_information = np.concatenate((header, tail))
    r_crc = r_crc_bits(a_information) if settings.r_crc_auto else _bits(settings.r_crc_bits)
    a_field = np.concatenate((a_information, r_crc))
    parts = [preamble, sync_word, a_field]
    prolonged = preamble if settings.prolonged_preamble else np.empty(0, dtype=np.uint8)
    if prolonged.size:
        parts.insert(0, prolonged)

    b_count = DECT_B_FIELD_BITS.get(packet_type)
    b_field = np.empty(0, dtype=np.uint8)
    x_field = np.empty(0, dtype=np.uint8)
    z_field = np.empty(0, dtype=np.uint8)
    if b_count is not None:
        b_field = _b_field_bits(project, b_count)
        x_field = (
            x_crc_bits(_x_test_data(packet_type, b_field))
            if settings.x_crc_auto
            else _bits(settings.x_field_bits)
        )
        parts.extend((b_field, x_field))
        if packet_type in {DectPacketType.P32Z, DectPacketType.P80Z}:
            z_field = x_field.copy() if settings.z_repeat_auto else _bits(settings.z_field_bits)
            parts.append(z_field)
    return np.concatenate(parts), {
        "prolonged_preamble_bits": prolonged,
        "preamble_bits": preamble,
        "sync_word_bits": sync_word,
        "a_header_bits": header,
        "a_tail_bits": tail,
        "r_crc_bits": r_crc,
        "a_field_bits": a_field,
        "b_field_bits": b_field,
        "x_field_bits": x_field,
        "z_field_bits": z_field,
    }


class DectWaveformEngine:
    """Generate normalized Classic DECT packets without VSA synchronization."""

    def generate(self, project: WaveformProject) -> GenerationResult:
        issues = validate_project(project)
        if issues:
            details = "; ".join(f"{item.path}: {item.message}" for item in issues)
            raise ValueError(f"Invalid waveform project: {details}")
        settings = project.dect
        if settings is None:
            raise ValueError("DECT settings are required")

        packet_bits, field_bits = dect_packet_bits(project)
        sps = int(project.samples_per_symbol)
        data_iq = _modulate_gfsk(
            packet_bits,
            samples_per_symbol=sps,
            sample_rate_hz=project.sample_rate_hz,
            deviation_hz=settings.frequency_deviation_hz,
            gaussian_bt=settings.gaussian_bt,
        )
        if settings.carrier_frequency_offset_hz:
            sample_index = np.arange(data_iq.size, dtype=np.float64)
            data_iq *= np.exp(
                2j
                * np.pi
                * float(settings.carrier_frequency_offset_hz)
                * sample_index
                / project.sample_rate_hz
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
            raise ValueError("Packet period is shorter than the generated DECT burst")
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
        active_ranges: list[tuple[int, int]] = []
        for repeat in range(int(project.repeat_count)):
            active_offset = repeat * single.size + prefix_count
            active_ranges.append((active_offset, active_offset + active_iq.size))
            offset = repeat * single.size + data_start
            packet_ranges.append((offset, offset + data_sample_count))
            _append_field_boundaries(
                boundaries,
                project.fields,
                repeat_offset=offset,
                samples_per_symbol=sps,
                repeat_suffix=("" if project.repeat_count == 1 else f" [{repeat + 1}]"),
            )

        packet_type = DectPacketType(settings.packet_type)
        level_metrics = measure_iq_levels(iq, active_ranges)
        return GenerationResult(
            iq=iq,
            sample_rate_hz=project.sample_rate_hz,
            field_boundaries=tuple(boundaries),
            packet_bits=GeneratedPacketBits(
                bits=packet_bits,
                protocol_id="dect.classic",
                phy_name="2-level GFSK",
                context={
                    "direction": settings.direction.value,
                    "packet_type": packet_type.value,
                    "preamble_mode": "Prolonged" if settings.prolonged_preamble else "Normal",
                    "p0_internal_bit": 16 if settings.prolonged_preamble else 0,
                },
            ),
            metadata={
                "project_name": project.name,
                "standard": project.standard.value,
                "center_frequency_hz": project.center_frequency_hz,
                "actual_rf_frequency_hz": (
                    project.center_frequency_hz + settings.carrier_frequency_offset_hz
                ),
                "carrier_plan_id": settings.carrier_plan_id,
                "carrier_channel": settings.carrier_channel,
                "carrier_frequency_offset_hz": settings.carrier_frequency_offset_hz,
                "packet_name": f"DECT {packet_type.value} Packet",
                "packet_type": packet_type.value,
                "direction": settings.direction.value,
                "packet_bits": packet_bits,
                **field_bits,
                "packet_sample_count": data_sample_count,
                "period_sample_count": single.size,
                "period_symbols": single.size / sps,
                "post_idle_sample_count": suffix_count,
                "packet_ranges_samples": tuple(packet_ranges),
                "active_ranges_samples": tuple(active_ranges),
                "data_start_sample": data_start,
                "data_stop_sample": data_start + data_sample_count,
                "samples_per_symbol": sps,
                "symbol_rate_hz": DECT_SYMBOL_RATE_HZ,
                "frequency_deviation_hz": settings.frequency_deviation_hz,
                "gaussian_bt": settings.gaussian_bt,
                **iq_level_metadata(level_metrics),
            },
        )
