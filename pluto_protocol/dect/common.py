"""Bit-order and error-control helpers from EN 300 175-3 clauses 6.2.5.2/4."""

from __future__ import annotations

import numpy as np


R_CRC_POLYNOMIAL = 0x10589
X_CRC4_POLYNOMIAL = 0x11


def _binary_bits(bits: np.ndarray) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("bits must be a one-dimensional binary array")
    return values


def _polynomial_remainder(bits: np.ndarray, polynomial: int, degree: int) -> np.ndarray:
    work = [int(bit) for bit in _binary_bits(bits)]
    taps = [((int(polynomial) >> shift) & 1) for shift in range(degree, -1, -1)]
    for index in range(max(0, len(work) - degree)):
        if work[index]:
            for offset, tap in enumerate(taps):
                work[index + offset] ^= tap
    return np.asarray(work[-degree:], dtype=np.uint8)


def r_crc_bits(data_bits: np.ndarray) -> np.ndarray:
    """Generate the transmitted 16 R-CRC bits, including final-bit inversion."""

    data = _binary_bits(data_bits)
    remainder = _polynomial_remainder(
        np.concatenate((data, np.zeros(16, dtype=np.uint8))),
        R_CRC_POLYNOMIAL,
        16,
    )
    transmitted = remainder.copy()
    transmitted[-1] ^= 1
    return transmitted


def r_crc_valid(codeword: np.ndarray) -> bool:
    values = _binary_bits(codeword)
    if values.size != 64:
        return False
    adjusted = values.copy()
    adjusted[-1] ^= 1
    return not bool(np.any(_polynomial_remainder(adjusted, R_CRC_POLYNOMIAL, 16)))


def x_crc_bits(test_data_bits: np.ndarray) -> np.ndarray:
    """Generate the four transmitted X-CRC bits for 2-level modulation."""

    data = _binary_bits(test_data_bits)
    return _polynomial_remainder(
        np.concatenate((data, np.zeros(4, dtype=np.uint8))),
        X_CRC4_POLYNOMIAL,
        4,
    )


def x_crc_valid(test_pattern: np.ndarray) -> bool:
    values = _binary_bits(test_pattern)
    if values.size < 4:
        return False
    return not bool(np.any(_polynomial_remainder(values, X_CRC4_POLYNOMIAL, 4)))


def dect_p_range(start_bit: int, stop_bit: int, p0_internal_bit: int) -> tuple[int, int]:
    """Convert half-open internal bit indices to half-open ETSI p-symbol indices."""

    offset = int(p0_internal_bit)
    return int(start_bit) - offset, int(stop_bit) - offset
