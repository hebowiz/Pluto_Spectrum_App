"""PSK symbol-number to physical phase mapping definitions."""

from __future__ import annotations

import numpy as np

from pluto_sa.vsa.model import ModulationKind


NATURAL_MAPPING = "Natural"
GRAY_MAPPING = "Gray"
BLUETOOTH_EDR_MAPPING = "Bluetooth EDR"
BLUETOOTH_HDT_MAPPING = "Bluetooth HDT"
PSK_SYMBOL_MAPPINGS = (
    NATURAL_MAPPING,
    GRAY_MAPPING,
    BLUETOOTH_EDR_MAPPING,
    BLUETOOTH_HDT_MAPPING,
)


def reverse_symbol_bits(symbols: np.ndarray, order: int) -> np.ndarray:
    """Reverse the bits within each symbol value.

    DSP internals use one canonical (MSB-numbered) symbol index.  R&S-style
    LSB symbol displays and pattern files use the bit-reversed index instead.
    The operation is its own inverse, so the same helper converts in either
    direction.
    """
    bit_count = int(round(np.log2(order)))
    if order != 1 << bit_count:
        raise ValueError("modulation order must be a power of two")
    values = np.asarray(symbols, dtype=np.int16)
    if np.any(values < 0) or np.any(values >= order):
        raise ValueError("symbol value is outside the modulation order")
    reversed_values = np.zeros(values.shape, dtype=np.int16)
    for bit in range(bit_count):
        reversed_values |= ((values >> bit) & 1) << (bit_count - 1 - bit)
    return reversed_values


def normalize_symbol_mapping(value: str) -> str:
    normalized = str(value).strip() or NATURAL_MAPPING
    aliases = {name.casefold(): name for name in PSK_SYMBOL_MAPPINGS}
    try:
        return aliases[normalized.casefold()]
    except KeyError as exc:
        raise ValueError(f"unsupported modulation mapping: {value}") from exc


def logical_to_phase_indices(kind: ModulationKind, mapping: str) -> np.ndarray:
    """Return the physical phase index for each logical symbol number."""
    normalized = normalize_symbol_mapping(mapping)
    if normalized == NATURAL_MAPPING:
        return np.arange(kind.order, dtype=np.int16)

    if normalized == GRAY_MAPPING:
        if kind is ModulationKind.BPSK:
            values = (0, 1)
        elif kind in {
            ModulationKind.QPSK,
            ModulationKind.OQPSK,
            ModulationKind.PI4_DQPSK,
            ModulationKind.PI4_QPSK,
        }:
            values = (0, 1, 3, 2)
        elif kind in {ModulationKind.DPSK8, ModulationKind.PSK8}:
            # R&S generic D8PSK Gray mapping.
            values = (0, 1, 3, 2, 6, 7, 5, 4)
        else:
            raise ValueError(f"{kind.value} does not support PSK Gray mapping")
        return np.asarray(values, dtype=np.int16)

    if normalized == BLUETOOTH_HDT_MAPPING:
        if kind in {ModulationKind.PI4_QPSK, ModulationKind.PSK8, ModulationKind.QAM16}:
            return np.arange(kind.order, dtype=np.int16)
        raise ValueError("Bluetooth HDT mapping requires pi/4-QPSK, 8PSK or 16QAM")

    if kind is ModulationKind.PI4_DQPSK:
        # Bluetooth: 00, 01, 11, 10 -> +pi/4, +3pi/4, -3pi/4, -pi/4.
        values = (0, 1, 3, 2)
    elif kind is ModulationKind.DPSK8:
        # Bluetooth EDR logical value -> physical phase index.
        values = (0, 1, 3, 2, 7, 6, 4, 5)
    else:
        raise ValueError(
            "Bluetooth EDR mapping is only valid for pi/4-DQPSK and 8DPSK"
        )
    return np.asarray(values, dtype=np.int16)


def psk_constellation(kind: ModulationKind, mapping: str = NATURAL_MAPPING) -> np.ndarray:
    """Return constellation points ordered by logical symbol number."""
    if kind is ModulationKind.BPSK:
        phases = np.asarray([0.0, np.pi])
    elif kind in {
        ModulationKind.QPSK,
        ModulationKind.OQPSK,
        ModulationKind.PI4_DQPSK,
        ModulationKind.PI4_QPSK,
    }:
        phases = np.pi / 4.0 + np.arange(4) * np.pi / 2.0
    elif kind in {ModulationKind.DPSK8, ModulationKind.PSK8}:
        phases = np.arange(8) * np.pi / 4.0
    elif kind is ModulationKind.QAM16:
        normalized = normalize_symbol_mapping(mapping)
        if normalized == NATURAL_MAPPING:
            levels = np.asarray([-3.0, -1.0, 1.0, 3.0])
        elif normalized in {GRAY_MAPPING, BLUETOOTH_HDT_MAPPING}:
            levels = np.asarray([-3.0, -1.0, 3.0, 1.0])
        else:
            raise ValueError(f"{mapping} is not valid for 16QAM")
        points = np.asarray(
            [levels[value >> 2] + 1j * levels[value & 0x3] for value in range(16)],
            dtype=np.complex128,
        )
        if normalized == BLUETOOTH_HDT_MAPPING:
            return points / 10.0
        return points / np.sqrt(10.0)
    else:
        raise ValueError(f"{kind.value} does not have a PSK constellation")
    return np.exp(1j * phases[logical_to_phase_indices(kind, mapping)])


def phase_indices_to_logical_symbols(
    kind: ModulationKind, mapping: str, phase_indices: np.ndarray
) -> np.ndarray:
    """Convert physical phase indices to the logical symbols shown in the table."""
    logical_to_phase = logical_to_phase_indices(kind, mapping)
    phase_to_logical = np.empty(kind.order, dtype=np.int16)
    phase_to_logical[logical_to_phase] = np.arange(kind.order, dtype=np.int16)
    indices = np.asarray(phase_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= kind.order):
        raise ValueError("phase index is outside the modulation order")
    return phase_to_logical[indices]
