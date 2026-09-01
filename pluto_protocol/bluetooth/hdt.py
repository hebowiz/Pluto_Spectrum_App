"""Bluetooth HDT PHY definitions shared by VSG generation and VSA analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class HDTRate(StrEnum):
    HDT2 = "HDT2"
    HDT3 = "HDT3"
    HDT4 = "HDT4"
    HDT6 = "HDT6"
    HDT7_5 = "HDT7.5"


@dataclass(frozen=True)
class HDTDefinition:
    rate: HDTRate
    rate_indicator: int
    modulation: str
    bits_per_symbol: int
    payload_code_rate: str


HDT_DEFINITIONS = {
    HDTRate.HDT2: HDTDefinition(HDTRate.HDT2, 0b001, "pi/4-QPSK", 2, "1/2"),
    HDTRate.HDT3: HDTDefinition(HDTRate.HDT3, 0b010, "pi/4-QPSK", 2, "3/4"),
    HDTRate.HDT4: HDTDefinition(HDTRate.HDT4, 0b011, "8PSK", 3, "2/3"),
    HDTRate.HDT6: HDTDefinition(HDTRate.HDT6, 0b100, "16QAM", 4, "3/4"),
    HDTRate.HDT7_5: HDTDefinition(HDTRate.HDT7_5, 0b101, "16QAM", 4, "15/16"),
}


def hdt_definition(rate: HDTRate | str) -> HDTDefinition:
    return HDT_DEFINITIONS[HDTRate(rate)]


_PUNCTURE_MASKS = {
    "1/2": (1, 1),
    "2/3": (1, 1, 0, 1),
    "3/4": (1, 1, 0, 1, 0, 1),
    "15/16": (
        1, 1, 0, 1, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 1, 0, 0, 1,
    ),
}


def convolutional_encode(bits: np.ndarray, *, terminate: bool = True) -> np.ndarray:
    """Encode with HDT K=6, G0=1+x^2+x^4+x^5 and G1=1+x+x^2+x^3+x^5."""

    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("bits must be a one-dimensional binary array")
    if terminate:
        values = np.concatenate((values, np.zeros(5, dtype=np.uint8)))
    history = np.zeros(5, dtype=np.uint8)
    encoded = np.empty(values.size * 2, dtype=np.uint8)
    for index, bit in enumerate(values):
        taps = np.concatenate(([bit], history))
        encoded[2 * index] = taps[0] ^ taps[2] ^ taps[4] ^ taps[5]
        encoded[2 * index + 1] = taps[0] ^ taps[1] ^ taps[2] ^ taps[3] ^ taps[5]
        history[1:] = history[:-1]
        history[0] = bit
    return encoded


def puncture(bits: np.ndarray, code_rate: str) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    try:
        mask = np.asarray(_PUNCTURE_MASKS[str(code_rate)], dtype=bool)
    except KeyError as error:
        raise ValueError(f"Unsupported HDT code rate: {code_rate}") from error
    return values[np.resize(mask, values.size)]


def map_hdt_symbols(bits: np.ndarray, rate: HDTRate | str) -> np.ndarray:
    """Map coded MSB-first bits to the HDT air-interface constellation."""

    definition = hdt_definition(rate)
    values = np.asarray(bits, dtype=np.uint8)
    width = definition.bits_per_symbol
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("bits must be a one-dimensional binary array")
    if values.size % width:
        values = np.pad(values, (0, width - values.size % width))
    grouped = values.reshape(-1, width)
    labels = grouped.dot(1 << np.arange(width - 1, -1, -1))
    if definition.modulation == "pi/4-QPSK":
        even_phases = np.asarray(
            [np.pi / 4.0, 3.0 * np.pi / 4.0, -np.pi / 4.0, -3.0 * np.pi / 4.0]
        )
        odd_phases = np.asarray([np.pi / 2.0, np.pi, 0.0, -np.pi / 2.0])
        phases = np.where(
            np.arange(labels.size) % 2 == 0,
            even_phases[labels],
            odd_phases[labels],
        )
        return np.exp(1j * phases).astype(np.complex64)
    if definition.modulation == "8PSK":
        phases = np.asarray(
            [0.0, np.pi / 4.0, 3.0 * np.pi / 4.0, np.pi / 2.0,
             -np.pi / 4.0, -np.pi / 2.0, -np.pi, -3.0 * np.pi / 4.0]
        )
        return np.exp(1j * phases[labels]).astype(np.complex64)
    levels = np.asarray([-3.0, -1.0, 3.0, 1.0])
    points = levels[labels >> 2] + 1j * levels[labels & 0x3]
    # HDT_VSr03_PR tabulates S_k x sqrt(10), so recover S_k with the
    # conventional unit-mean-power 16QAM normalization.
    return (points / np.sqrt(10.0)).astype(np.complex64)


__all__ = [
    "HDTDefinition", "HDT_DEFINITIONS", "HDTRate", "convolutional_encode",
    "hdt_definition", "map_hdt_symbols", "puncture",
]
