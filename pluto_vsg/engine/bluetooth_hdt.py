"""Bluetooth-derived HDT packet waveform generation."""

from __future__ import annotations

import numpy as np

from pluto_protocol.bluetooth.hdt import convolutional_encode, hdt_definition, map_hdt_symbols, puncture
from pluto_protocol.model import GeneratedPacketBits
from pluto_sa.vsa.profiles.bluetooth_br import prbs9_period
from pluto_vsg.engine.base import FieldBoundary, GenerationResult
from pluto_vsg.engine.bluetooth_br import (
    _append_field_boundaries, _extend_edge_phase, _placed_power_envelope, _srrc_taps,
)
from pluto_vsg.model import PayloadSourceKind, WaveformProject, waveform_timing_samples, validate_project


def hdt_payload_bits(project: WaveformProject) -> np.ndarray:
    settings = project.bluetooth_hdt
    if settings is None:
        raise ValueError("Bluetooth HDT settings are required")
    count = int(settings.payload_length_bytes) * 8
    if not count:
        return np.empty(0, dtype=np.uint8)
    if settings.payload_source == PayloadSourceKind.PRBS9:
        source = prbs9_period()
    else:
        pattern = settings.payload_pattern.strip().replace(" ", "")
        if settings.payload_source == PayloadSourceKind.FIXED:
            pattern = pattern[:1]
        source = np.asarray([int(bit) for bit in pattern], dtype=np.uint8)
    return source[np.arange(count) % source.size]


def _shape_symbols(symbols: np.ndarray, sps: int, rolloff: float) -> np.ndarray:
    span = 10
    padded = np.pad(np.asarray(symbols, dtype=np.complex128), (span, span), mode="edge")
    impulses = np.zeros(padded.size * sps, dtype=np.complex128)
    impulses[np.arange(padded.size) * sps + sps // 2] = padded
    shaped = np.convolve(impulses, _srrc_taps(sps, rolloff, span), mode="same")
    result = shaped[span * sps : (span + symbols.size) * sps]
    return result / np.sqrt(np.mean(np.abs(result) ** 2))


class BluetoothHDTWaveformEngine:
    def generate(self, project: WaveformProject) -> GenerationResult:
        issues = validate_project(project)
        if issues:
            raise ValueError("Invalid waveform project: " + "; ".join(f"{i.path}: {i.message}" for i in issues))
        settings = project.bluetooth_hdt
        if settings is None:
            raise ValueError("Bluetooth HDT settings are required")
        definition = hdt_definition(settings.rate)
        sps = int(project.samples_per_symbol)
        if not np.isclose(project.sample_rate_hz, 2_000_000.0 * sps):
            raise ValueError("HDT sample rate must equal 2 Msym/s times samples per symbol")

        payload = hdt_payload_bits(project)
        coded = puncture(convolutional_encode(payload), definition.payload_code_rate)
        payload_symbols = map_hdt_symbols(coded, settings.rate)
        # Deterministic reference training and a compact RI/length control header.
        training_bits = np.resize(np.asarray([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8), 74 * 2)
        training = map_hdt_symbols(training_bits, "HDT2")
        control_data = np.asarray(
            [((definition.rate_indicator >> (2 - i)) & 1) for i in range(3)]
            + [((int(settings.payload_length_bytes) >> i) & 1) for i in range(12)], dtype=np.uint8,
        )
        control_bits = np.resize(convolutional_encode(control_data), 62 * 2)
        control = map_hdt_symbols(control_bits, "HDT2")
        symbols = np.concatenate((training, control, payload_symbols))
        data_iq = _shape_symbols(symbols, sps, float(settings.rrc_rolloff))
        data_count = data_iq.size

        envelope = project.power_envelope
        rise_count = round(envelope.rise_symbols * sps) if envelope.enabled else 0
        fall_count = round(envelope.fall_symbols * sps) if envelope.enabled else 0
        rise_start = round(envelope.rise_delay_symbols * sps) if envelope.enabled else 0
        fall_start = data_count + round(envelope.fall_delay_symbols * sps) if envelope.enabled else data_count
        active_start = min(0, rise_start)
        active_stop = max(data_count, fall_start + fall_count)
        positions = np.arange(active_start, active_stop)
        active_iq = _extend_edge_phase(data_iq, positions)
        if envelope.enabled:
            active_iq *= _placed_power_envelope(positions, rise_start=rise_start, rise_samples=rise_count, fall_start=fall_start, fall_samples=fall_count, shape=envelope.shape)
        prefix = int(settings.pre_idle_symbols) * sps
        _, _, minimum, period = waveform_timing_samples(project)
        if period < minimum:
            raise ValueError("Packet period is shorter than the generated burst")
        suffix = period - prefix - active_iq.size
        single = np.concatenate((np.zeros(prefix), active_iq, np.zeros(suffix))).astype(np.complex64)
        iq = np.tile(single, int(project.repeat_count))
        data_start = prefix - active_start
        boundaries: list[FieldBoundary] = []
        ranges = []
        for repeat in range(int(project.repeat_count)):
            offset = repeat * single.size + data_start
            ranges.append((offset, offset + data_count))
            _append_field_boundaries(boundaries, project.fields, repeat_offset=offset, samples_per_symbol=sps, repeat_suffix="" if project.repeat_count == 1 else f" [{repeat + 1}]")
        return GenerationResult(
            iq=iq, sample_rate_hz=project.sample_rate_hz, field_boundaries=tuple(boundaries),
            packet_bits=GeneratedPacketBits(payload, "bluetooth.hdt", settings.rate.value, context={"rate_indicator": definition.rate_indicator}),
            metadata={
                "project_name": project.name, "standard": project.standard.value,
                "packet_name": f"{settings.rate.value} RF Test Packet", "phy": settings.rate.value,
                "modulation": definition.modulation, "payload_code_rate": definition.payload_code_rate,
                "payload_bits": payload, "coded_payload_bits": coded, "packet_sample_count": data_count,
                "period_sample_count": single.size, "packet_ranges_samples": tuple(ranges),
                "data_start_sample": data_start, "data_stop_sample": data_start + data_count,
                "samples_per_symbol": sps, "symbol_rate_hz": 2_000_000.0,
            },
        )
