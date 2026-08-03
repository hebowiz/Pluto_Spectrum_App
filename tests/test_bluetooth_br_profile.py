from pathlib import Path

import numpy as np
import pytest

from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.profiles.bluetooth_br import (
    BLUETOOTH_GIAC_SYNC_WORD_HEX,
    BluetoothBRProfile,
    access_code_bits,
    build_packet_bits,
    fec13_decode,
    fec13_encode,
    giac_access_code_bits,
    header_error_check,
    modulate_packet_bits,
    whitening_sequence,
)


def _hex_bits(value: str) -> np.ndarray:
    return np.asarray(
        [int(bit) for digit in value for bit in f"{int(digit, 16):04b}"],
        dtype=np.uint8,
    )


def test_giac_matches_bluetooth_sig_sample_vector() -> None:
    access = giac_access_code_bits()
    text = "".join(str(int(bit)) for bit in access)

    assert access.size == 72
    assert text[:4] == "0101"
    assert text[4:68] == f"{int(BLUETOOTH_GIAC_SYNC_WORD_HEX, 16):064b}"
    assert text[68:] == "1010"
    assert giac_access_code_bits(include_trailer=False).size == 68


def test_shortened_inquiry_access_code_uses_known_symbols_for_tracking() -> None:
    access = giac_access_code_bits(include_trailer=False)
    iq = modulate_packet_bits(
        access,
        sample_rate_hz=4_000_000.0,
        frequency_deviation_hz=165_000.0,
        carrier_frequency_offset_hz=1_037_000.0,
        prefix_samples=113,
        suffix_samples=97,
        snr_db=18.0,
        seed=29,
    )
    recording = IQRecording(iq, sample_rate_hz=4_000_000.0)

    result = BluetoothBRProfile(access_bits=access).analyze(recording)

    assert result.demodulation.access_bit_errors == 0
    np.testing.assert_array_equal(result.demodulation.bits[: access.size], access)
    assert result.demodulation.carrier_frequency_offset_hz == pytest.approx(
        1_037_000.0, abs=10_000.0
    )
    assert result.demodulation.frequency_deviation_hz == pytest.approx(
        165_000.0, abs=15_000.0
    )


def test_pluto_smartphone_inquiry_capture_recovers_giac_without_errors() -> None:
    fixture = Path(__file__).with_name("fixtures") / "bluetooth_giac_inquiry_pluto_4msps.npz"
    with np.load(fixture, allow_pickle=False) as capture:
        recording = IQRecording(
            capture["iq"],
            sample_rate_hz=float(capture["sample_rate_hz"]),
            center_frequency_hz=float(capture["center_frequency_hz"]),
            source="Pluto smartphone Inquiry fixture",
        )
    access = giac_access_code_bits(include_trailer=False)

    result = BluetoothBRProfile(access_bits=access).analyze(recording)

    assert result.demodulation.access_correlation > 0.99
    assert result.demodulation.access_bit_errors == 0
    np.testing.assert_array_equal(result.demodulation.bits[: access.size], access)
    assert result.demodulation.carrier_frequency_offset_hz == pytest.approx(
        1_037_494.0, abs=5_000.0
    )
    assert result.demodulation.frequency_deviation_hz == pytest.approx(
        164_778.0, abs=5_000.0
    )


@pytest.mark.parametrize(
    ("lap", "preamble", "sync_word", "trailer"),
    [
        (0x000000, "5", "7e7041e34000000d", "5"),
        (0xFFFFFF, "a", "e758b5227ffffff2", "a"),
        (0x9E8B33, "5", "475c58cc73345e72", "a"),
        (0x9E8B34, "5", "28ed3c34cb345e72", "a"),
        (0x616CEC, "5", "586a491f0dcda18d", "5"),
    ],
)
def test_access_code_generator_matches_bluetooth_sig_vectors(
    lap, preamble, sync_word, trailer
) -> None:
    expected = _hex_bits(preamble + sync_word + trailer)
    np.testing.assert_array_equal(access_code_bits(lap), expected)


def test_whitening_lfsr_matches_bluetooth_sig_sample_sequence() -> None:
    expected = np.asarray(
        [
            1, 1, 1, 0, 0, 0, 1, 1,
            1, 0, 1, 1, 0, 0, 0, 1,
            0, 1, 0, 0, 1, 0, 1, 1,
            1, 1, 1, 0, 1, 0, 1, 0,
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(whitening_sequence(0x3F, expected.size), expected)


@pytest.mark.parametrize(
    ("uap", "data", "expected_hec"),
    [
        (0x00, 0x123, 0xE1),
        (0x47, 0x123, 0x06),
        (0x00, 0x124, 0x32),
        (0x47, 0x11F, 0x12),
    ],
)
def test_hec_matches_bluetooth_sig_sample_vectors(uap, data, expected_hec) -> None:
    data_bits = np.asarray([(data >> index) & 1 for index in range(10)], dtype=np.uint8)
    assert header_error_check(data_bits, uap) == expected_hec


def test_rate_third_fec_majority_corrects_one_bit_per_triplet() -> None:
    bits = np.asarray([0, 1, 1, 0, 1], dtype=np.uint8)
    encoded = fec13_encode(bits)
    encoded[::3] ^= 1

    decoded, corrected = fec13_decode(encoded)

    np.testing.assert_array_equal(decoded, bits)
    assert corrected == bits.size


def test_bluetooth_packet_recovers_header_and_payload_from_impaired_iq() -> None:
    rng = np.random.default_rng(33)
    payload = rng.integers(0, 2, size=96, dtype=np.uint8)
    packet_bits = build_packet_bits(
        clock_6_1=0x2B,
        uap=0x47,
        payload_bits=payload,
        lt_addr=3,
        packet_type=4,
        flow=1,
        arqn=0,
        seqn=1,
    )
    iq = modulate_packet_bits(
        packet_bits,
        sample_rate_hz=16_000_000.0,
        carrier_frequency_offset_hz=55_000.0,
        prefix_samples=37,
        suffix_samples=31,
        snr_db=14.0,
        seed=7,
    )
    recording = IQRecording(iq, sample_rate_hz=16_000_000.0, source="Bluetooth test")

    result = BluetoothBRProfile().analyze(
        recording,
        clock_6_1=0x2B,
        uap=0x47,
    )

    assert result.demodulation.access_correlation > 0.9
    assert result.demodulation.access_bit_errors == 0
    assert result.demodulation.access_start_sample == pytest.approx(37, abs=4)
    assert result.demodulation.carrier_frequency_offset_hz == pytest.approx(
        55_000.0, abs=8_000.0
    )
    assert result.header is not None
    assert result.header.lt_addr == 3
    assert result.header.packet_type == 4
    assert result.header.flow == 1
    assert result.header.arqn == 0
    assert result.header.seqn == 1
    assert result.header.hec_valid is True
    np.testing.assert_array_equal(result.payload_bits[: payload.size], payload)
    assert result.demodulation.burst_ranges


def test_bluetooth_packet_tracks_frequency_drift_and_iq_inversion() -> None:
    payload = np.tile(np.asarray([0, 1, 1, 0], dtype=np.uint8), 40)
    packet_bits = build_packet_bits(
        clock_6_1=0x15,
        uap=0x2A,
        payload_bits=payload,
        packet_type=3,
    )
    drift_hz_per_s = 150.0e6  # 150 Hz/us
    iq = np.conj(
        modulate_packet_bits(
            packet_bits,
            sample_rate_hz=8_000_000.0,
            carrier_frequency_offset_hz=-40_000.0,
            carrier_frequency_drift_hz_per_s=drift_hz_per_s,
            prefix_samples=19,
            suffix_samples=17,
            snr_db=13.0,
            seed=18,
        )
    )
    recording = IQRecording(iq, sample_rate_hz=8_000_000.0)

    result = BluetoothBRProfile().analyze(
        recording, clock_6_1=0x15, uap=0x2A
    )

    assert result.demodulation.iq_inverted is True
    # The decision-directed drift value is a coarse correction estimate rather
    # than a conformance measurement, but it must identify the inverted slope.
    assert -300.0e6 < result.demodulation.carrier_frequency_drift_hz_per_s < -50.0e6
    assert result.header is not None and result.header.hec_valid is True
    np.testing.assert_array_equal(result.payload_bits[: payload.size], payload)
