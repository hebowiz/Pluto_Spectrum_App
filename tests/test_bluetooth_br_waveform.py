from pathlib import Path

import numpy as np
import pytest

from pluto_sa.vsa.model import ModulationKind, SignalDescription
from pluto_sa.vsa.pattern import (
    KnownPattern,
    PatternAnalyzer,
    PatternSearchMode,
    PatternSearchSettings,
    ResultRangeSettings,
)
from pluto_sa.vsa.profiles.bluetooth_br import (
    BluetoothBRProfile,
    access_code_bits,
    decode_dh1_payload,
    prbs9_period,
)
from pluto_sa.vsa.profiles.bluetooth_br_waveform import generate_br_dh1


def test_generated_br_dh1_has_valid_header_payload_and_crc() -> None:
    waveform = generate_br_dh1()
    result = BluetoothBRProfile(access_bits=waveform.access_bits).analyze(
        waveform.recording,
        clock_6_1=0x2B,
        uap=0x6B,
    )

    assert result.header is not None
    assert result.header.hec_valid is True
    assert result.header.packet_type == 0x4
    payload = decode_dh1_payload(result.payload_bits, uap=0x6B)
    assert payload.length_bytes == 27
    assert payload.crc_valid is True
    recovered_body = np.asarray(
        [(byte >> bit) & 1 for byte in payload.body for bit in range(8)],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(
        recovered_body, prbs9_period()[np.arange(27 * 8) % 511]
    )
    assert result.demodulation.access_bit_errors == 0
    assert waveform.recording.sample_count == 48_000
    assert waveform.packet_start_sample == 32_000
    assert waveform.packet_stop_sample == 37_856


def test_generated_br_dh1_is_repeatable() -> None:
    first = generate_br_dh1(seed=77)
    second = generate_br_dh1(seed=77)
    np.testing.assert_array_equal(first.recording.iq, second.recording.iq)


def test_generic_vsa_recovers_complete_br_dh1_symbol_stream() -> None:
    waveform = generate_br_dh1()
    result = PatternAnalyzer().search(
        waveform.recording,
        SignalDescription(
            modulation=ModulationKind.GFSK,
            symbol_rate_hz=1_000_000.0,
            frequency_deviation_hz=160_000.0,
            tx_filter="Gaussian",
            filter_parameter=0.5,
        ),
        PatternSearchSettings(
            pattern=KnownPattern(tuple(map(int, waveform.access_bits[:32]))),
            mode=PatternSearchMode.ON,
            correlation_threshold_auto=False,
            iq_correlation_threshold=0.9,
        ),
        ResultRangeSettings(result_length=waveform.packet_bits.size),
    )

    assert result.correlation > 0.97
    assert result.pattern_symbol_errors == 0
    assert result.pattern_start_sample == pytest.approx(
        waveform.packet_start_sample, abs=1
    )
    assert result.metadata["timing_correction_accepted"]
    assert not result.metadata["drift_model_accepted"]
    assert result.carrier_frequency_drift_hz_per_s == 0.0
    assert result.metadata["drift_rejection_reason"] != "Accepted"
    assert result.carrier_frequency_offset_hz == pytest.approx(20_000.0, abs=2_000.0)
    np.testing.assert_array_equal(result.decoded_symbols, waveform.packet_bits)


def test_checked_in_br_dh1_fixture_matches_generator() -> None:
    path = Path(__file__).with_name("fixtures") / "bluetooth_dh1_prbs9_16msps.npz"
    expected = generate_br_dh1(seed=11)

    with np.load(path, allow_pickle=False) as fixture:
        np.testing.assert_array_equal(fixture["iq"], expected.recording.iq)
        np.testing.assert_array_equal(fixture["packet_bits"], expected.packet_bits)
        np.testing.assert_array_equal(
            fixture["access_bits"], access_code_bits(0xC6967E)
        )
        np.testing.assert_array_equal(
            fixture["payload_body_bits"], expected.payload_body_bits
        )
        assert int(fixture["payload_length_bytes"]) == 27
