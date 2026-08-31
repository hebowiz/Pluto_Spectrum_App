import numpy as np
import pytest

from pluto_protocol.bluetooth.hdt import (
    HDTRate,
    convolutional_encode,
    hdt_definition,
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
    expected_power = 0.1 if rate in {HDTRate.HDT6, HDTRate.HDT7_5} else 1.0
    assert np.mean(np.abs(symbols) ** 2) == pytest.approx(expected_power)


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
