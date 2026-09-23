import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import numpy as np
import pytest
from pluto_vsa.profiles.bluetooth_br import (
    BluetoothBRProfile,
    access_code_bits,
    decode_dh1_payload,
    payload_crc_bytes,
)
from pluto_vsa.profiles.bluetooth_edr import generate_edr_dh1
from pluto_vsa.mapping import (
    BLUETOOTH_EDR_MAPPING,
    phase_indices_to_logical_symbols,
    reverse_symbol_bits,
)
from pluto_vsa.model import IQRecording, ModulationKind, SignalDescription
from pluto_vsa.pattern import (
    KnownPattern,
    PatternAnalyzer,
    PatternSearchMode,
    PatternSearchSettings,
    ResultRangeSettings,
)
from pluto_vsg.engine import BluetoothBRWaveformEngine
from pluto_vsg.model import validate_project
from pluto_vsg.model import (
    BluetoothPacketKind,
    PayloadSourceKind,
    bluetooth_packet_is_edr,
    bluetooth_packet_properties,
)
from pluto_vsg.profiles import bluetooth_br_edr_project, bluetooth_br_fields
from pluto_vsg.ui.main_window import _instantaneous_frequency_khz


def test_vsg_dh1_generation_decodes_with_valid_hec_and_crc() -> None:
    base = bluetooth_br_edr_project()
    assert base.bluetooth_br is not None
    settings = replace(base.bluetooth_br, whitening_enabled=True)
    project = replace(
        base,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )

    result = BluetoothBRWaveformEngine().generate(project)
    packet_start, packet_stop = result.metadata["packet_ranges_samples"][0]
    recording = IQRecording(
        iq=result.iq[packet_start:packet_stop],
        sample_rate_hz=result.sample_rate_hz,
        center_frequency_hz=project.center_frequency_hz,
    )
    analyzed = BluetoothBRProfile(access_code_bits(settings.lap)).analyze(
        recording,
        clock_6_1=settings.clock_6_1,
        uap=settings.uap,
    )

    assert analyzed.header is not None
    assert analyzed.header.hec_valid is True
    assert analyzed.header.packet_type == 4
    payload = decode_dh1_payload(analyzed.payload_bits, uap=settings.uap)
    assert payload.length_bytes == 27
    assert payload.crc_valid is True


@pytest.mark.parametrize(
    ("packet_kind", "payload_length"),
    (
        (BluetoothPacketKind.DH1_2, 54),
        (BluetoothPacketKind.DH1_3, 83),
    ),
)
def test_vsg_edr_generation_matches_validated_phase_sequence(
    packet_kind, payload_length
) -> None:
    base = bluetooth_br_edr_project()
    assert base.bluetooth_br is not None
    settings = replace(
        base.bluetooth_br,
        packet_kind=packet_kind,
        payload_length_bytes=payload_length,
        whitening_enabled=True,
    )
    project = replace(
        base,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )

    result = BluetoothBRWaveformEngine().generate(project)
    reference = generate_edr_dh1(
        packet_kind.value,
        sample_rate_hz=project.sample_rate_hz,
        carrier_frequency_offset_hz=0.0,
        duration_s=0.001,
        packet_start_s=0.0001,
        snr_db=200.0,
    )

    assert result.metadata["packet_name"] == packet_kind.value
    np.testing.assert_array_equal(
        result.metadata["edr_phase_indices"],
        reference.differential_phase_indices,
    )
    assert result.metadata["edr_start_sample"] - result.metadata["gfsk_stop_sample"] == 5 * project.samples_per_symbol
    assert np.max(np.abs(result.iq)) <= 1.0 + 1e-6
    assert [field.name for field in project.fields] == [
        "Access Code",
        "Header",
        "Guard",
        "EDR Data",
    ]

    modulation = (
        ModulationKind.PI4_DQPSK
        if packet_kind == BluetoothPacketKind.DH1_2
        else ModulationKind.DPSK8
    )
    expected_symbols = phase_indices_to_logical_symbols(
        modulation,
        BLUETOOTH_EDR_MAPPING,
        np.asarray(result.metadata["edr_phase_indices"]),
    )
    pattern = reverse_symbol_bits(expected_symbols[:10], modulation.order)
    analyzed = PatternAnalyzer().search(
        IQRecording(
            iq=result.iq,
            sample_rate_hz=result.sample_rate_hz,
            center_frequency_hz=project.center_frequency_hz,
        ),
        SignalDescription(
            modulation=modulation,
            symbol_rate_hz=1_000_000.0,
            tx_filter="Root Raised Cosine",
            filter_parameter=settings.edr_rolloff,
            symbol_mapping="Bluetooth EDR",
        ),
        PatternSearchSettings(
            pattern=KnownPattern(tuple(map(int, pattern))),
            mode=PatternSearchMode.ON,
            correlation_threshold_auto=False,
            iq_correlation_threshold=0.9,
        ),
        ResultRangeSettings(result_length=244),
    )
    assert analyzed.pattern_symbol_errors == 0
    assert analyzed.correlation > 0.99
    np.testing.assert_array_equal(analyzed.decoded_symbols, expected_symbols)
    assert analyzed.evm_rms_percent < 5.0


def test_vsg_edr_guard_relative_power_changes_only_guard_amplitude() -> None:
    base = bluetooth_br_edr_project()
    assert base.bluetooth_br is not None
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1_2,
        payload_length_bytes=54,
        edr_guard_relative_power_db=-12.0,
        edr_guard_ramp_in_symbols=0.0,
        edr_guard_ramp_out_symbols=0.0,
    )
    project = replace(
        base,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )

    result = BluetoothBRWaveformEngine().generate(project)
    gfsk_stop = int(result.metadata["gfsk_stop_sample"])
    edr_start = int(result.metadata["edr_start_sample"])
    reference = np.median(
        np.abs(result.iq[gfsk_stop - 8 * project.samples_per_symbol : gfsk_stop])
    )
    guard = np.median(np.abs(result.iq[gfsk_stop:edr_start]))

    assert 20.0 * np.log10(guard / reference) == pytest.approx(-12.0, abs=0.02)
    assert result.metadata["edr_guard_relative_power_db"] == -12.0
    assert project.fields[2].name == "Guard"
    assert project.fields[2].relative_power_db == -12.0


def test_vsg_edr_guard_power_uses_smooth_edge_transitions() -> None:
    base = bluetooth_br_edr_project()
    assert base.bluetooth_br is not None
    settings = replace(
        base.bluetooth_br,
        packet_kind=BluetoothPacketKind.DH1_2,
        payload_length_bytes=54,
        edr_guard_relative_power_db=-18.0,
        edr_guard_ramp_in_symbols=1.0,
        edr_guard_ramp_out_symbols=1.0,
        edr_guard_ramp_shape="Cosine",
    )
    project = replace(
        base,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )

    result = BluetoothBRWaveformEngine().generate(project)
    gfsk_stop = int(result.metadata["gfsk_stop_sample"])
    edr_start = int(result.metadata["edr_start_sample"])
    sps = project.samples_per_symbol
    guard = result.iq[gfsk_stop:edr_start]
    guard_magnitude = np.abs(guard)

    assert guard[0] == pytest.approx(result.iq[gfsk_stop - 1])
    assert guard[-1] == pytest.approx(result.iq[edr_start])
    assert np.all(np.diff(guard_magnitude[:sps]) <= 1e-12)
    assert np.all(np.diff(guard_magnitude[-sps:]) >= -1e-12)
    assert result.metadata["edr_guard_ramp_in_symbols"] == 1.0
    assert result.metadata["edr_guard_ramp_out_symbols"] == 1.0
    assert result.metadata["edr_guard_ramp_shape"] == "Cosine"


@pytest.mark.parametrize(
    ("packet_kind", "payload_max", "packet_type", "bits_per_symbol", "slots"),
    (
        (BluetoothPacketKind.DH1, 27, 0x4, 1, 1),
        (BluetoothPacketKind.DH3, 183, 0xB, 1, 3),
        (BluetoothPacketKind.DH5, 339, 0xF, 1, 5),
        (BluetoothPacketKind.DH1_2, 54, 0x4, 2, 1),
        (BluetoothPacketKind.DH3_2, 367, 0xA, 2, 3),
        (BluetoothPacketKind.DH5_2, 679, 0xE, 2, 5),
        (BluetoothPacketKind.DH1_3, 83, 0x8, 3, 1),
        (BluetoothPacketKind.DH3_3, 552, 0xB, 3, 3),
        (BluetoothPacketKind.DH5_3, 1021, 0xF, 3, 5),
    ),
)
def test_vsg_all_dhx_packet_definitions_generate(
    packet_kind, payload_max, packet_type, bits_per_symbol, slots
) -> None:
    base = bluetooth_br_edr_project()
    assert base.bluetooth_br is not None
    settings = replace(
        base.bluetooth_br,
        packet_kind=packet_kind,
        payload_length_bytes=payload_max,
    )
    project = replace(
        base,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )

    assert bluetooth_packet_properties(packet_kind) == (
        payload_max,
        packet_type,
        bits_per_symbol,
        slots,
    )
    assert bluetooth_packet_is_edr(packet_kind) is (bits_per_symbol > 1)
    assert validate_project(project) == ()

    result = BluetoothBRWaveformEngine().generate(project)
    payload_header = np.asarray(result.metadata["payload_header_bits"])
    assert payload_header.size == (
        8 if packet_kind == BluetoothPacketKind.DH1 else 16
    )
    packed_payload_header = sum(
        int(bit) << index for index, bit in enumerate(payload_header)
    )
    assert (packed_payload_header >> 3) & (
        0x1F if packet_kind == BluetoothPacketKind.DH1 else 0x3FF
    ) == payload_max
    assert result.metadata["payload_body_bits"].size == payload_max * 8
    expected_crc = payload_crc_bytes(
        np.concatenate((payload_header, result.metadata["payload_body_bits"])),
        settings.uap,
    )
    expected_crc_bits = np.asarray(
        [(byte >> bit) & 1 for byte in expected_crc for bit in range(8)],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(
        result.metadata["payload_crc_bits"], expected_crc_bits
    )
    assert result.metadata["packet_name"] == packet_kind.value
    assert result.metadata["packet_sample_count"] > 0
    header_air = np.asarray(result.metadata["packet_bits"])[72:126]
    header_data = header_air.reshape(-1, 3)[:, 0]
    packed_header = sum(
        int(bit) << index for index, bit in enumerate(header_data[:10])
    )
    assert (packed_header >> 3) & 0xF == packet_type
    packet_duration_us = (
        result.metadata["packet_sample_count"] / result.sample_rate_hz * 1e6
    )
    assert packet_duration_us <= slots * 625.0


def test_vsg_generation_emits_hierarchical_sample_boundaries() -> None:
    project = bluetooth_br_edr_project()

    result = BluetoothBRWaveformEngine().generate(project)
    boundaries = result.field_boundaries
    access = next(item for item in boundaries if item.name == "Access Code")
    sync_word = next(item for item in boundaries if item.name == "Sync Word")
    header = next(item for item in boundaries if item.name == "Header")
    header_type = next(item for item in boundaries if item.name == "TYPE")
    payload_body = next(item for item in boundaries if item.name == "Payload Body")

    assert access.level == 0
    assert access.start_symbol == 0
    assert access.stop_symbol == 72
    assert sync_word.level == 1
    assert sync_word.parent_name == "Access Code"
    assert (sync_word.start_symbol, sync_word.stop_symbol) == (4, 68)
    assert header.logical_bit_count == 18
    assert header.stop_symbol - header.start_symbol == 54
    assert header_type.logical_bit_count == 4
    assert header_type.stop_symbol - header_type.start_symbol == 12
    assert payload_body.logical_bit_count == 27 * 8
    assert payload_body.stop_sample - payload_body.start_sample == 27 * 8 * 8
    packet_ranges = result.metadata["packet_ranges_samples"]
    expected_start = (
        project.bluetooth_br.pre_idle_symbols * project.samples_per_symbol
        - min(
            0,
            round(
                project.power_envelope.rise_delay_symbols
                * project.samples_per_symbol
            ),
        )
    )
    assert packet_ranges == ((
        expected_start,
        expected_start + result.metadata["packet_sample_count"],
    ),)


def test_vsg_payload_sources_generate_distinct_expected_bits() -> None:
    base = bluetooth_br_edr_project()
    assert base.bluetooth_br is not None
    engine = BluetoothBRWaveformEngine()
    fixed_settings = replace(
        base.bluetooth_br,
        payload_length_bytes=2,
        payload_source=PayloadSourceKind.FIXED,
        payload_pattern="0",
    )
    pattern_settings = replace(
        base.bluetooth_br,
        payload_length_bytes=2,
        payload_source=PayloadSourceKind.PATTERN,
        payload_pattern="10",
    )
    prbs_settings = replace(
        base.bluetooth_br,
        payload_length_bytes=2,
        payload_source=PayloadSourceKind.PRBS9,
        payload_pattern="0",
    )
    fixed = engine.generate(
        replace(
            base,
            bluetooth_br=fixed_settings,
            fields=bluetooth_br_fields(fixed_settings),
        )
    )
    pattern = engine.generate(
        replace(
            base,
            bluetooth_br=pattern_settings,
            fields=bluetooth_br_fields(pattern_settings),
        )
    )
    prbs = engine.generate(
        replace(
            base,
            bluetooth_br=prbs_settings,
            fields=bluetooth_br_fields(prbs_settings),
        )
    )

    np.testing.assert_array_equal(
        fixed.metadata["payload_body_bits"], np.zeros(16, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        pattern.metadata["payload_body_bits"], np.tile([1, 0], 8)
    )
    assert not np.array_equal(
        prbs.metadata["payload_body_bits"],
        fixed.metadata["payload_body_bits"],
    )
    assert not np.array_equal(
        prbs.metadata["payload_body_bits"],
        pattern.metadata["payload_body_bits"],
    )


def test_vsg_engine_coerces_ui_string_payload_source() -> None:
    base = bluetooth_br_edr_project()
    assert base.bluetooth_br is not None
    settings = replace(
        base.bluetooth_br,
        payload_length_bytes=2,
        payload_source=PayloadSourceKind.PRBS9.value,
        payload_pattern="0",
    )
    project = replace(
        base,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )

    result = BluetoothBRWaveformEngine().generate(project)

    assert np.count_nonzero(result.metadata["payload_body_bits"]) > 0


def test_vsg_delayed_ramp_down_holds_last_symbol_frequency() -> None:
    base = bluetooth_br_edr_project()
    project = replace(
        base,
        power_envelope=replace(
            base.power_envelope,
            fall_delay_symbols=2.0,
            fall_symbols=1.0,
        ),
    )

    result = BluetoothBRWaveformEngine().generate(project)
    data_stop = int(result.metadata["data_stop_sample"])
    hold_samples = 2 * project.samples_per_symbol
    frequency = _instantaneous_frequency_khz(result.iq, result.sample_rate_hz)
    packet_bits = np.asarray(result.metadata["packet_bits"])
    expected_khz = (
        (2.0 * float(packet_bits[-1]) - 1.0)
        * project.bluetooth_br.frequency_deviation_hz
        + project.bluetooth_br.carrier_frequency_offset_hz
    ) / 1e3

    np.testing.assert_allclose(
        frequency[data_stop : data_stop + hold_samples - 1],
        expected_khz,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.abs(result.iq[data_stop : data_stop + hold_samples]),
        1.0,
        atol=1e-6,
    )
    assert result.metadata["edge_frequency_mode"] == "Hold first / last symbol"
