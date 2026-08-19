from pathlib import Path

import numpy as np
import pytest

from pluto_sa.vsa.model import ModulationKind, SignalDescription
from pluto_sa.vsa.pattern import (
    DemodulationSettings,
    KnownPattern,
    MatchSelectionPolicy,
    PatternAnalyzer,
    PatternSearchMode,
    PatternSearchSettings,
    ResultRangeSettings,
)
from pluto_sa.vsa.profiles.bluetooth_br import decode_header_air_bits, prbs9_period
from pluto_sa.vsa.profiles.bluetooth_edr import generate_edr_dh1
from pluto_sa.vsa.sources import FileIQSource


@pytest.mark.parametrize(
    ("packet_name", "packet_type", "payload_length", "bit_width"),
    (("2-DH1", 0x4, 54, 2), ("3-DH1", 0x8, 83, 3)),
)
def test_generated_edr_dh1_has_spec_packet_fields(
    packet_name, packet_type, payload_length, bit_width
):
    waveform = generate_edr_dh1(packet_name)

    header = decode_header_air_bits(
        waveform.header_air_bits,
        uap=0x6B,
        clock_6_1=0x2B,
    )
    assert header.hec_valid is True
    assert header.packet_type == packet_type
    assert waveform.payload_length_bytes == payload_length
    packed_payload_header = sum(
        int(bit) << index for index, bit in enumerate(waveform.payload_header_bits)
    )
    assert (packed_payload_header >> 3) & 0x3FF == payload_length
    np.testing.assert_array_equal(
        waveform.payload_body_bits,
        prbs9_period()[np.arange(payload_length * 8) % 511],
    )
    assert waveform.trailer_bits.size == 2 * bit_width
    assert not np.any(waveform.trailer_bits)
    assert waveform.differential_phase_indices.size == 244
    assert waveform.recording.sample_count == 48_000
    assert waveform.edr_start_sample - waveform.gfsk_stop_sample == 80
    assert waveform.packet_stop_sample <= waveform.recording.sample_count


def test_generated_edr_waveform_is_repeatable():
    first = generate_edr_dh1("2-DH1", seed=77)
    second = generate_edr_dh1("2-DH1", seed=77)

    np.testing.assert_array_equal(first.recording.iq, second.recording.iq)


@pytest.mark.parametrize("packet_name", ("2-DH1", "3-DH1"))
def test_generic_vsa_recovers_edr_sync_pattern(packet_name):
    waveform = generate_edr_dh1(packet_name)
    sync_symbols = waveform.logical_symbols[:10]
    signal = SignalDescription(
        modulation=waveform.modulation,
        symbol_rate_hz=1_000_000.0,
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
        symbol_mapping="Bluetooth EDR",
    )

    result = PatternAnalyzer().search(
        waveform.recording,
        signal,
        PatternSearchSettings(
            pattern=KnownPattern(tuple(map(int, sync_symbols))),
            mode=PatternSearchMode.ON,
            match_selection=MatchSelectionPolicy.STRONGEST,
            correlation_threshold_auto=False,
            iq_correlation_threshold=0.9,
        ),
        ResultRangeSettings(result_length=244),
    )

    assert result.correlation > 0.97
    assert result.pattern_symbol_errors == 0
    assert result.pattern_start_sample == pytest.approx(
        waveform.edr_start_sample + 16, abs=4
    )
    assert result.carrier_frequency_offset_hz == pytest.approx(20_000.0, abs=5_000.0)
    np.testing.assert_array_equal(
        result.decoded_symbols, waveform.logical_symbols
    )
    assert result.metadata["physical_evm_rms_percent"] < 5.0
    assert result.metadata["differential_symbol_evm_rms_percent"] == pytest.approx(
        result.evm_rms_percent
    )
    assert result.metadata["bluetooth_devm_rms_percent"] < 5.0
    assert np.median(np.abs(result.measured_symbols)) == pytest.approx(1.0, abs=0.02)
    assert np.max(np.abs(result.measured_symbols)) < 1.05


@pytest.mark.parametrize("packet_name", ("2-DH1", "3-DH1"))
def test_psk_carrier_is_centered_before_matched_filter(packet_name):
    waveform = generate_edr_dh1(
        packet_name,
        carrier_frequency_offset_hz=100_000.0,
        snr_db=80.0,
        seed=9,
    )
    signal = SignalDescription(
        modulation=waveform.modulation,
        symbol_rate_hz=1_000_000.0,
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
        symbol_mapping="Bluetooth EDR",
    )
    search = PatternSearchSettings(
        pattern=KnownPattern(
            tuple(map(int, waveform.logical_symbols[:10]))
        ),
        mode=PatternSearchMode.ON,
        correlation_threshold_auto=False,
        iq_correlation_threshold=0.8,
    )
    result_range = ResultRangeSettings(result_length=244)
    demodulation = DemodulationSettings()
    analyzer = PatternAnalyzer()

    filter_first = analyzer._search_psk_pass(
        waveform.recording,
        signal,
        search,
        result_range,
        demodulation,
        prefilter_carrier_frequency_offset_hz=0.0,
    )
    centered_first = analyzer.search(
        waveform.recording,
        signal,
        search,
        result_range,
        demodulation,
    )

    assert centered_first.metadata["prefilter_cfo_correction_applied"] is True
    assert centered_first.metadata["prefilter_coarse_cfo_hz"] == pytest.approx(
        100_000.0, abs=500.0
    )
    assert centered_first.metadata["postfilter_residual_cfo_hz"] == pytest.approx(
        0.0, abs=500.0
    )
    assert centered_first.carrier_frequency_offset_hz == pytest.approx(
        100_000.0, abs=500.0
    )
    assert centered_first.evm_rms_percent < 1.0
    assert centered_first.evm_rms_percent < filter_first.evm_rms_percent / 5.0
    np.testing.assert_array_equal(
        centered_first.decoded_symbols,
        waveform.logical_symbols,
    )


@pytest.mark.parametrize(
    ("filename", "packet_name"),
    (
        ("bluetooth_2dh1_prbs9_16msps.npz", "2-DH1"),
        ("bluetooth_3dh1_prbs9_16msps.npz", "3-DH1"),
    ),
)
def test_checked_in_edr_fixture_matches_generator(filename, packet_name):
    path = Path(__file__).with_name("fixtures") / filename
    expected = generate_edr_dh1(
        packet_name,
        seed=21 if packet_name == "2-DH1" else 31,
    )

    with np.load(path, allow_pickle=False) as fixture:
        np.testing.assert_array_equal(fixture["iq"], expected.recording.iq)
        np.testing.assert_array_equal(
            fixture["differential_phase_indices"],
            expected.differential_phase_indices,
        )
        assert int(fixture["payload_length_bytes"]) == expected.payload_length_bytes


def test_high_rate_edr_iqtar_reports_three_distinct_evm_definitions():
    recording = FileIQSource.load(
        Path(__file__).with_name("fixtures") / "bt_6DH1_capture.iq.tar"
    )
    signal = SignalDescription(
        modulation=ModulationKind.DPSK8,
        symbol_rate_hz=2_000_000.0,
        tx_filter="Root Raised Cosine",
        filter_parameter=0.4,
        symbol_mapping="Bluetooth EDR",
    )
    result = PatternAnalyzer().search(
        recording,
        signal,
        PatternSearchSettings(
            pattern=KnownPattern((2, 7, 2, 7, 2, 7, 7, 2, 2, 2)),
            mode=PatternSearchMode.ON,
            correlation_threshold_auto=False,
            iq_correlation_threshold=0.9,
        ),
        ResultRangeSettings(result_length=600),
    )

    assert result.metadata["physical_evm_rms_percent"] == pytest.approx(
        9.06, abs=0.1
    )
    assert result.metadata["differential_symbol_evm_rms_percent"] == pytest.approx(
        5.95, abs=0.1
    )
    assert result.metadata["bluetooth_devm_rms_percent"] == pytest.approx(
        4.94, abs=0.1
    )
