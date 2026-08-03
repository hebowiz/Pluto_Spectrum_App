from pathlib import Path

import numpy as np
import pytest

from pluto_sa.vsa.channel import extract_analysis_channel
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.profiles.bluetooth_br import (
    BLUETOOTH_GIAC_SYNC_WORD_HEX,
    BluetoothBRProfile,
    access_code_bits,
    build_packet_bits,
    decode_dh1_payload,
    fec13_decode,
    fec13_encode,
    find_dh1_candidates,
    giac_access_code_bits,
    header_error_check,
    match_prbs9,
    modulate_packet_bits,
    payload_crc_bytes,
    prbs9_period,
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


def test_pluto_fixed_br_capture_recovers_unwhitened_dh1_prbs9() -> None:
    fixture = Path(__file__).with_name("fixtures") / "bluetooth_br_prbs9_pluto_16msps.npz"
    with np.load(fixture, allow_pickle=False) as capture:
        wideband = IQRecording(
            capture["iq"],
            sample_rate_hz=float(capture["sample_rate_hz"]),
            center_frequency_hz=float(capture["center_frequency_hz"]),
            usable_bandwidth_hz=float(capture["usable_bandwidth_hz"]),
        )
    recording = extract_analysis_channel(
        wideband,
        center_frequency_hz=2_441_000_000.0,
        bandwidth_hz=1_500_000.0,
    )
    profile = BluetoothBRProfile(access_bits=access_code_bits(0xC6967E))

    raw = profile.analyze(recording)
    candidates = find_dh1_candidates(
        raw.header_air_bits,
        raw.payload_bits,
        require_crc=False,
    )
    unwhitened = [item for item in candidates if not item.header.whitening_enabled]

    assert raw.demodulation.access_correlation > 0.99
    assert raw.demodulation.access_bit_errors == 0
    assert len(unwhitened) == 1
    candidate = unwhitened[0]
    assert candidate.header.packet_type == 4
    assert candidate.payload.length_bytes == 27
    assert candidate.payload.crc_valid is False
    body_bits = np.asarray(
        [
            (byte >> index) & 1
            for byte in candidate.payload.body
            for index in range(8)
        ],
        dtype=np.uint8,
    )
    assert match_prbs9(body_bits).bit_errors == 0


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


def test_payload_crc_matches_bluetooth_sig_sample_vector() -> None:
    data = bytes([0x4E, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    bits = np.asarray(
        [(byte >> index) & 1 for byte in data for index in range(8)],
        dtype=np.uint8,
    )

    assert payload_crc_bytes(bits, 0x47) == bytes.fromhex("6dd2")


def test_decode_dh1_payload_matches_bluetooth_sig_complete_packet() -> None:
    air_text = (
        "01110100"
        "10000000"
        "01000000"
        "11000000"
        "00100000"
        "10100000"
        "11101100"
        "00110110"
    )
    payload = decode_dh1_payload(
        np.asarray([int(bit) for bit in air_text], dtype=np.uint8),
        uap=0x47,
    )

    assert payload.logical_channel == 2
    assert payload.flow == 1
    assert payload.length_bytes == 5
    assert payload.body == bytes([1, 2, 3, 4, 5])
    assert payload.received_crc == bytes.fromhex("376c")
    assert payload.crc_valid is True


def test_find_dh1_candidate_uses_header_hec_and_payload_crc() -> None:
    payload_header = np.asarray([0, 1, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
    body = np.asarray(
        [(byte >> index) & 1 for byte in range(1, 6) for index in range(8)],
        dtype=np.uint8,
    )
    crc_bytes = payload_crc_bytes(np.concatenate((payload_header, body)), 0x47)
    crc = np.asarray(
        [(byte >> index) & 1 for byte in crc_bytes for index in range(8)],
        dtype=np.uint8,
    )
    packet_bits = build_packet_bits(
        clock_6_1=0x2B,
        uap=0x47,
        payload_bits=np.concatenate((payload_header, body, crc)),
        packet_type=4,
    )
    recording = IQRecording(
        modulate_packet_bits(packet_bits, prefix_samples=23, suffix_samples=19),
        sample_rate_hz=8_000_000.0,
    )
    raw = BluetoothBRProfile().analyze(recording)

    candidates = find_dh1_candidates(
        raw.header_air_bits,
        raw.payload_bits,
        uaps=(0x47,),
    )

    assert len(candidates) == 1
    assert candidates[0].header.clock_6_1 == 0x2B
    assert candidates[0].payload.body == bytes([1, 2, 3, 4, 5])


def test_prbs9_match_finds_phase_polarity_and_bit_errors() -> None:
    period = prbs9_period()
    values = period[(np.arange(300) + 123) % period.size] ^ 1
    values[[5, 99, 201]] ^= 1

    match = match_prbs9(values)

    assert match.bit_errors == 3
    assert match.bit_count == 300
    assert match.phase == 123
    assert match.inverted is True
    assert match.time_reversed is False


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
