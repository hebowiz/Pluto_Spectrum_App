"""ETSI EN 300 176-1 clause 11 RF-modulation test patterns.

The arrays in this module are the single source of truth for both VSG air-bit
generation and VSA air-bit identification.  Figure boundaries follow V2.4.1
figures 27 through 31; they intentionally exclude X/Z fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class DectRFPattern(StrEnum):
    NORMAL = "Normal data"
    CASE_A = "Case A"
    CASE_B_ETSI = "Case B (ETSI)"


class DectScramblingMode(StrEnum):
    NONE = "None"
    STANDARD = "Standard"


class DectCaseBFormat(StrEnum):
    A_FIELD_ONLY = "A-field only"
    HALF_SLOT = "Half-slot"
    FULL_SLOT = "Full-slot"
    VARIABLE_640 = "Variable length j=640"
    DOUBLE_SLOT = "Double-slot"


@dataclass(frozen=True)
class DectCaseBDefinition:
    format: DectCaseBFormat
    figure: int
    bit_count: int
    repetitions: int

    @property
    def label(self) -> str:
        return f"Case B / Figure {self.figure}"


@dataclass(frozen=True)
class DectPatternIdentification:
    label: str
    case: str | None
    figure: int | None
    exact_etsi_pattern: bool
    pattern_errors: int
    bit_count: int
    dsv_max: int

    @property
    def pattern_valid(self) -> bool:
        return self.pattern_errors == 0 and self.case is not None


CASE_B_DEFINITIONS = {
    DectCaseBFormat.A_FIELD_ONLY: DectCaseBDefinition(
        DectCaseBFormat.A_FIELD_ONLY, 27, 32, 100
    ),
    DectCaseBFormat.HALF_SLOT: DectCaseBDefinition(
        DectCaseBFormat.HALF_SLOT, 28, 80, 40
    ),
    DectCaseBFormat.FULL_SLOT: DectCaseBDefinition(
        DectCaseBFormat.FULL_SLOT, 29, 320, 10
    ),
    DectCaseBFormat.VARIABLE_640: DectCaseBDefinition(
        DectCaseBFormat.VARIABLE_640, 30, 640, 5
    ),
    DectCaseBFormat.DOUBLE_SLOT: DectCaseBDefinition(
        DectCaseBFormat.DOUBLE_SLOT, 31, 800, 5
    ),
}

DECT_RF_MODULATION_LIMITS_HZ = {
    "A": (259_000.0, 403_000.0),
    "B": (202_000.0, 403_000.0),
}


def deviation_in_limits(deviation_hz: float, case: str) -> bool:
    lower, upper = DECT_RF_MODULATION_LIMITS_HZ[str(case).upper()]
    return bool(lower < abs(float(deviation_hz)) < upper)


def _alternating(count: int) -> np.ndarray:
    # ETSI figures define even-order bits as 1 and odd-order bits as 0.
    return (1 - (np.arange(count, dtype=np.uint8) & 1)).astype(np.uint8)


def case_a_bits(bit_count: int) -> np.ndarray:
    if bit_count <= 0 or bit_count % 8:
        raise ValueError("DECT Case A test field length must be a positive multiple of 8")
    return np.tile(np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8), bit_count // 8)


def case_b_bits(format: DectCaseBFormat | str) -> np.ndarray:
    selected = DectCaseBFormat(format)
    if selected is DectCaseBFormat.A_FIELD_ONLY:
        return np.concatenate((np.ones(16, np.uint8), np.zeros(16, np.uint8)))
    if selected is DectCaseBFormat.HALF_SLOT:
        return np.concatenate((_alternating(8), np.ones(32, np.uint8), np.zeros(32, np.uint8), _alternating(8)))
    if selected is DectCaseBFormat.FULL_SLOT:
        return np.concatenate((_alternating(128), np.ones(64, np.uint8), np.zeros(64, np.uint8), _alternating(64)))
    if selected is DectCaseBFormat.VARIABLE_640:
        middle = np.tile(np.concatenate((np.ones(64, np.uint8), np.zeros(64, np.uint8))), 3)
        return np.concatenate((_alternating(128), middle, _alternating(128)))
    middle = np.tile(np.concatenate((np.ones(64, np.uint8), np.zeros(64, np.uint8))), 4)
    return np.concatenate((_alternating(144), middle, _alternating(144)))


def case_b_format_for_packet(packet_type: str, bit_count: int | None = None) -> DectCaseBFormat | None:
    name = str(packet_type).upper()
    count = bit_count
    if name == "P00" and count in (None, 32):
        return DectCaseBFormat.A_FIELD_ONLY
    if count == 80:
        return DectCaseBFormat.HALF_SLOT
    if name in {"P32", "P32Z"} or count == 320:
        return DectCaseBFormat.FULL_SLOT
    if count == 640:
        return DectCaseBFormat.VARIABLE_640
    if name in {"P80", "P80Z"} or count == 800:
        return DectCaseBFormat.DOUBLE_SLOT
    return None


def maximum_dsv(bits: np.ndarray) -> int:
    values = np.where(np.asarray(bits, dtype=np.uint8) != 0, 1, -1)
    return int(np.max(np.abs(np.cumsum(values, dtype=np.int64)))) if values.size else 0


def identify_rf_pattern(bits: np.ndarray, packet_type: str = "") -> DectPatternIdentification:
    observed = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if not observed.size:
        return DectPatternIdentification("Unknown", None, None, False, 0, 0, 0)
    candidate_errors: list[int] = []
    if observed.size % 8 == 0:
        reference = case_a_bits(observed.size)
        errors = int(np.count_nonzero(observed != reference))
        candidate_errors.append(errors)
        if errors == 0:
            return DectPatternIdentification("Case A", "A", None, True, 0, observed.size, maximum_dsv(observed))
    format = case_b_format_for_packet(packet_type, observed.size)
    if format is not None:
        reference = case_b_bits(format)
        errors = int(np.count_nonzero(observed != reference))
        candidate_errors.append(errors)
        if errors == 0:
            definition = CASE_B_DEFINITIONS[format]
            return DectPatternIdentification(definition.label, "B", definition.figure, True, 0, observed.size, maximum_dsv(observed))
    errors = min(candidate_errors, default=0)
    dsv = maximum_dsv(observed)
    if dsv <= 64:
        return DectPatternIdentification("Case B / Generic", "B", None, False, errors, observed.size, dsv)
    return DectPatternIdentification("Unknown", None, None, False, errors, observed.size, dsv)


def scrambling_sequence(bit_count: int, frame_phase: int) -> np.ndarray:
    """Generate EN 300 175-3 clause 6.2.4 sequence sf.

    The five-stage x^5+x^2+1 register is initialized Q0..Q2 from f and
    Q3=Q4=1.  Q4 is emitted before each shift.  The prescribed inversion
    toggles after the all-ones state and starts inverted.
    """

    if not 0 <= int(frame_phase) <= 7:
        raise ValueError("DECT scrambling phase must be in the range 0..7")
    q = [int(frame_phase) & 1, (int(frame_phase) >> 1) & 1, (int(frame_phase) >> 2) & 1, 1, 1]
    invert = True
    output = np.empty(int(bit_count), dtype=np.uint8)
    for index in range(output.size):
        output[index] = q[4] ^ int(invert)
        was_all_ones = all(q)
        feedback = q[4] ^ q[1]
        q = [feedback, q[0], q[1], q[2], q[3]]
        if was_all_ones:
            invert = not invert
    return output


def apply_scrambling(bits: np.ndarray, mode: DectScramblingMode | str, frame_phase: int | None) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    if DectScramblingMode(mode) is DectScramblingMode.NONE:
        return values.copy()
    if frame_phase is None:
        raise ValueError("DECT standard scrambling requires a known frame phase")
    return values ^ scrambling_sequence(values.size, int(frame_phase))
