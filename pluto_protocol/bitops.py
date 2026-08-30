"""Bit-order helpers shared by protocol generators and analyzers."""

from __future__ import annotations

import numpy as np


def bits_lsb(value: int, width: int) -> np.ndarray:
    return np.asarray([(int(value) >> index) & 1 for index in range(int(width))], dtype=np.uint8)


def bits_to_int_lsb(bits: np.ndarray) -> int:
    return sum(int(bit) << index for index, bit in enumerate(np.asarray(bits)))


def bits_to_int_msb(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits):
        value = (value << 1) | int(bit)
    return value


def bytes_to_bits_lsb(data: bytes) -> np.ndarray:
    return np.asarray([bit for value in data for bit in ((value >> index) & 1 for index in range(8))], dtype=np.uint8)


def bits_to_bytes_lsb(bits: np.ndarray, *, require_complete: bool = True) -> bytes:
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or np.any(values > 1):
        raise ValueError("bits must be a one-dimensional binary array")
    if require_complete and values.size % 8:
        raise ValueError("bits must contain complete octets")
    stop = values.size - (values.size % 8)
    return bytes(bits_to_int_lsb(values[index:index + 8]) for index in range(0, stop, 8))


def bits_hex_lsb(bits: np.ndarray) -> str:
    return bits_to_bytes_lsb(bits, require_complete=False).hex().upper()
