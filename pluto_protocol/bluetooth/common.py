"""Bluetooth BR/EDR and LE protocol primitives without VSA/VSG dependencies."""

from __future__ import annotations

import numpy as np

from pluto_protocol.bitops import bits_to_int_lsb


def reverse_byte(value: int) -> int:
    return int(f"{int(value) & 0xFF:08b}"[::-1], 2)


def br_whitening_sequence(clock_6_1: int, count: int) -> np.ndarray:
    if not 0 <= int(clock_6_1) <= 0x3F:
        raise ValueError("clock_6_1 must be a six-bit value")
    state = np.asarray([1] + [(int(clock_6_1) >> shift) & 1 for shift in range(5, -1, -1)], dtype=np.uint8)
    output = np.empty(int(count), dtype=np.uint8)
    for index in range(output.size):
        output[index] = state[0]
        state = np.asarray([state[1], state[2], state[3] ^ state[0], state[4], state[5], state[6], state[0]], dtype=np.uint8)
    return output


def header_error_check(data_10_bits: np.ndarray, uap: int) -> int:
    bits = np.asarray(data_10_bits, dtype=np.uint8)
    if bits.shape != (10,) or np.any(bits > 1):
        raise ValueError("data_10_bits must contain exactly ten binary bits")
    register = int(uap)
    for bit in bits:
        feedback = ((register >> 7) & 1) ^ int(bit)
        register = (register << 1) & 0xFF
        if feedback:
            register ^= 0xA7
    return reverse_byte(register)


def fec13_decode(bits: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or values.size % 3 or np.any(values > 1):
        raise ValueError("rate 1/3 FEC input must contain complete binary triplets")
    triplets = values.reshape(-1, 3)
    decoded = (np.sum(triplets, axis=1) >= 2).astype(np.uint8)
    corrected = int(np.count_nonzero(np.any(triplets != decoded[:, None], axis=1)))
    return decoded, corrected


def payload_crc_bytes(bits: np.ndarray, uap: int) -> bytes:
    values = np.asarray(bits, dtype=np.uint8)
    register = int(uap)
    for bit in values:
        feedback = ((register >> 15) & 1) ^ int(bit)
        register = (register << 1) & 0xFFFF
        if feedback:
            register ^= 0x1021
    return bytes((reverse_byte(register >> 8), reverse_byte(register)))


def le_whitening_sequence(channel_index: int, count: int) -> np.ndarray:
    channel = int(channel_index)
    if not 0 <= channel <= 39:
        raise ValueError("channel_index must be in the range 0 through 39")
    register = np.asarray([1] + [(channel >> index) & 1 for index in range(5, -1, -1)], dtype=np.uint8)
    output = np.empty(int(count), dtype=np.uint8)
    for index in range(output.size):
        feedback = int(register[6])
        output[index] = feedback
        previous = register.copy()
        register[0], register[1], register[2], register[3] = feedback, previous[0], previous[1], previous[2]
        register[4], register[5], register[6] = previous[3] ^ feedback, previous[4], previous[5]
    return output


def le_crc24_bits(bits: np.ndarray, init: int = 0x555555) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    register = int(init)
    for bit in values:
        feedback = int(bit) ^ ((register >> 23) & 1)
        register = (register << 1) & 0xFFFFFF
        if feedback:
            register ^= 0x00065B
    return np.asarray([(register >> position) & 1 for position in range(23, -1, -1)], dtype=np.uint8)


def decode_acl_header(bits: np.ndarray) -> tuple[int, int, int]:
    values = np.asarray(bits, dtype=np.uint8)
    if values.size not in (8, 16):
        raise ValueError("ACL payload header must be 8 or 16 bits")
    return bits_to_int_lsb(values[:2]), int(values[2]), bits_to_int_lsb(values[3:])
