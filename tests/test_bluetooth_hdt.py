import numpy as np
import pytest

from pluto_protocol.bluetooth.hdt import (
    HDTRate,
    convolutional_encode,
    hdt_crc32,
    hdt_definition,
    hdt_rf_test_control_bits,
    hdt_rf_test_training_symbols,
    map_hdt_symbols,
    puncture,
)


def test_hdt_rate_definitions_cover_qpsk_psk8_and_qam16():
    assert hdt_definition(HDTRate.HDT2).modulation == "pi/4-QPSK"
    assert hdt_definition(HDTRate.HDT4).bits_per_symbol == 3
    assert hdt_definition(HDTRate.HDT6).modulation == "16QAM"
    assert hdt_definition(HDTRate.HDT7_5).payload_code_rate == "15/16"


def test_hdt_convolutional_encoder_adds_five_zero_tail_bits():
    encoded = convolutional_encode(np.asarray([1, 0, 1], dtype=np.uint8))
    assert encoded.size == 2 * (3 + 5)
    assert set(encoded.tolist()) <= {0, 1}


def test_hdt_puncturing_uses_requested_rate_mask():
    encoded = np.ones(60, dtype=np.uint8)
    assert puncture(encoded, "1/2").size == 60
    assert puncture(encoded, "3/4").size == 40
    assert puncture(encoded, "15/16").size == 32


@pytest.mark.parametrize("rate", list(HDTRate))
def test_hdt_symbol_mapping_uses_specified_constellation(rate):
    width = hdt_definition(rate).bits_per_symbol
    labels = np.arange(1 << width, dtype=np.uint8)
    shifts = np.arange(width - 1, -1, -1)
    bits = ((labels[:, None] >> shifts) & 1).reshape(-1)
    symbols = map_hdt_symbols(bits, rate)
    assert symbols.size == 1 << width
    assert np.mean(np.abs(symbols) ** 2) == pytest.approx(1.0)


def test_hdt_pi4_qpsk_alternates_even_and_odd_constellations():
    bits = np.asarray([0, 0, 0, 0, 1, 0, 1, 0], dtype=np.uint8)
    symbols = map_hdt_symbols(bits, HDTRate.HDT2)
    expected_phases = np.asarray([np.pi / 4, np.pi / 2, -np.pi / 4, 0.0])
    np.testing.assert_allclose(symbols, np.exp(1j * expected_phases), atol=1e-6)


def test_hdt_8psk_uses_bluetooth_label_order():
    labels = np.arange(8, dtype=np.uint8)
    bits = ((labels[:, None] >> np.arange(2, -1, -1)) & 1).reshape(-1)
    symbols = map_hdt_symbols(bits, HDTRate.HDT4)
    expected_phases = np.asarray(
        [0.0, np.pi / 4, 3 * np.pi / 4, np.pi / 2,
         -np.pi / 4, -np.pi / 2, -np.pi, -3 * np.pi / 4]
    )
    np.testing.assert_allclose(symbols, np.exp(1j * expected_phases), atol=1e-6)


def test_hdt_rf_test_training_uses_standard_sts_and_u7_lts():
    symbols = hdt_rf_test_training_symbols()

    assert symbols.size == 74
    np.testing.assert_array_equal(
        symbols[:36], np.tile(np.asarray([-1, -1j, 1j, 1]), 9)
    )
    np.testing.assert_allclose(symbols[36:40], symbols[53:57], atol=1e-6)
    np.testing.assert_allclose(symbols[40:57], symbols[57:74], atol=1e-6)


def test_hdt_rf_test_control_header_matches_known_hdt7_5_vector():
    bits = hdt_rf_test_control_bits(HDTRate.HDT7_5, 509)

    assert bits.size == 57
    assert bits[20:23].tolist() == [1, 0, 1]
    assert sum(int(bits[24 + index]) << index for index in range(9)) == 510
    assert sum(int(bit) << (23 - index) for index, bit in enumerate(bits[33:57])) == 0x13FB5A


def test_hdt_crc32_distinguishes_standard_and_legacy_rf_test_initialization():
    data = np.zeros(8, dtype=np.uint8)

    assert hdt_crc32(data, init=0xAA555555) != hdt_crc32(data, init=0x00555555)
