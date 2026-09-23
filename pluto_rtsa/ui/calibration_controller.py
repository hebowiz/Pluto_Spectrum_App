"""Calibration controller for frequency dependent display correction."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MIN_CALIBRATION_FREQ_HZ = 100_000_000
MAX_CALIBRATION_FREQ_HZ = 5_900_000_000
DEFAULT_CALIBRATION_STEP_HZ = 100_000_000


@dataclass
class FrequencyDependentOffsetTable:
    """Frequency dependent correction table with interpolation support."""

    frequency_hz: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    calibration_offset_db: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=float))
    enabled: bool = False
    loaded: bool = False

    def clear(self) -> None:
        self.frequency_hz = np.empty(0, dtype=np.int64)
        self.calibration_offset_db = np.empty(0, dtype=float)
        self.loaded = False

    def set_table(self, frequency_hz: np.ndarray, calibration_offset_db: np.ndarray) -> None:
        if len(frequency_hz) == 0 or len(calibration_offset_db) == 0:
            self.clear()
            return
        if len(frequency_hz) != len(calibration_offset_db):
            raise ValueError("frequency_hz and calibration_offset_db length mismatch")
        self.frequency_hz = np.asarray(frequency_hz, dtype=np.int64)
        self.calibration_offset_db = np.asarray(calibration_offset_db, dtype=float)
        self.loaded = True

    def offset_db_for_frequency(self, frequency_hz: float) -> float:
        if not self.enabled or not self.loaded or len(self.frequency_hz) == 0:
            return 0.0
        if len(self.frequency_hz) == 1:
            return float(self.calibration_offset_db[0])
        return float(
            np.interp(
                float(frequency_hz),
                self.frequency_hz.astype(float),
                self.calibration_offset_db.astype(float),
                left=float(self.calibration_offset_db[0]),
                right=float(self.calibration_offset_db[-1]),
            )
        )

    def offsets_db_for_frequencies(self, frequency_hz: np.ndarray) -> np.ndarray:
        if len(frequency_hz) == 0:
            return np.empty(0, dtype=float)
        if not self.enabled or not self.loaded or len(self.frequency_hz) == 0:
            return np.zeros(len(frequency_hz), dtype=float)
        if len(self.frequency_hz) == 1:
            return np.full(len(frequency_hz), float(self.calibration_offset_db[0]), dtype=float)
        return np.interp(
            np.asarray(frequency_hz, dtype=float),
            self.frequency_hz.astype(float),
            self.calibration_offset_db.astype(float),
            left=float(self.calibration_offset_db[0]),
            right=float(self.calibration_offset_db[-1]),
        ).astype(float)


@dataclass
class CalibrationCsvSummary:
    """Simple load summary for logging/UI notifications."""

    count: int
    min_frequency_hz: int
    max_frequency_hz: int
    source_path: str


@dataclass
class CalibrationPointResult:
    """One finalized calibration point result."""

    frequency_hz: int
    measured_power_dbm: float
    reference_power_dbm: float
    calibration_offset_db: float


@dataclass
class CalibrationSequencePoint:
    """One target point in calibration sequence."""

    frequency_hz: int
    reference_power_dbm: float


def _read_non_comment_csv_lines(path: str | Path) -> list[str]:
    rows: list[str] = []
    with open(path, "r", encoding="utf-8-sig") as csv_file:
        for line in csv_file:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            rows.append(line)
    return rows


def _normalize_frequency_value_pairs(
    frequency_hz_values: list[float],
    data_values: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    if len(frequency_hz_values) != len(data_values):
        raise ValueError("frequency/data value count mismatch")
    latest_by_frequency: dict[int, float] = {}
    for frequency_hz, data_value in zip(frequency_hz_values, data_values):
        latest_by_frequency[int(round(float(frequency_hz)))] = float(data_value)
    sorted_items = sorted(latest_by_frequency.items(), key=lambda item: item[0])
    if not sorted_items:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    frequencies = np.asarray([item[0] for item in sorted_items], dtype=np.int64)
    values = np.asarray([item[1] for item in sorted_items], dtype=float)
    return frequencies, values


class CalibrationController:
    """Thin state/controller for calibration data and CSV loading."""

    def __init__(self) -> None:
        self.offset_table = FrequencyDependentOffsetTable()
        self.reference_frequency_hz = np.empty(0, dtype=np.int64)
        self.reference_power_dbm = np.empty(0, dtype=float)
        self.reference_loaded = False
        self.sequence_points: list[CalibrationSequencePoint] = []
        self.sequence_index = 0
        self.sequence_retry_waiting = False
        self.sequence_is_measuring = False
        self.measurement_results: list[CalibrationPointResult] = []
        self.last_loaded_correction_csv: str | None = None
        self.last_loaded_reference_csv: str | None = None

    @property
    def correction_enabled(self) -> bool:
        return bool(self.offset_table.enabled)

    @property
    def correction_loaded(self) -> bool:
        return bool(self.offset_table.loaded)

    def set_correction_enabled(self, enabled: bool) -> None:
        self.offset_table.enabled = bool(enabled)

    def toggle_correction_enabled(self) -> bool:
        self.offset_table.enabled = not self.offset_table.enabled
        return bool(self.offset_table.enabled)

    def get_frequency_offset_db(self, frequency_hz: float) -> float:
        return self.offset_table.offset_db_for_frequency(frequency_hz)

    def get_frequency_offsets_db(self, frequency_hz: np.ndarray) -> np.ndarray:
        return self.offset_table.offsets_db_for_frequencies(frequency_hz)

    def load_correction_csv(self, path: str | Path) -> CalibrationCsvSummary:
        rows = _read_non_comment_csv_lines(path)
        if not rows:
            raise ValueError("Calibration CSV has no data rows.")
        reader = csv.DictReader(rows)
        if reader.fieldnames is None:
            raise ValueError("Calibration CSV header is missing.")
        required_columns = {"frequency_hz", "calibration_offset_db"}
        if not required_columns.issubset(set(reader.fieldnames)):
            raise ValueError(
                "Calibration CSV must contain columns: frequency_hz, calibration_offset_db"
            )

        frequency_values: list[float] = []
        offset_values: list[float] = []
        for row in reader:
            frequency_values.append(float(row["frequency_hz"]))
            offset_values.append(float(row["calibration_offset_db"]))

        frequencies_hz, offsets_db = _normalize_frequency_value_pairs(
            frequency_values, offset_values
        )
        if len(frequencies_hz) == 0:
            raise ValueError("Calibration CSV contains no valid calibration rows.")

        self.offset_table.set_table(frequencies_hz, offsets_db)
        self.last_loaded_correction_csv = str(path)
        return CalibrationCsvSummary(
            count=int(len(frequencies_hz)),
            min_frequency_hz=int(frequencies_hz[0]),
            max_frequency_hz=int(frequencies_hz[-1]),
            source_path=str(path),
        )

    def load_reference_csv(self, path: str | Path) -> CalibrationCsvSummary:
        rows = _read_non_comment_csv_lines(path)
        if not rows:
            raise ValueError("Reference CSV has no data rows.")
        reader = csv.DictReader(rows)
        if reader.fieldnames is None:
            raise ValueError("Reference CSV header is missing.")
        required_columns = {"frequency_hz", "reference_power_dbm"}
        if not required_columns.issubset(set(reader.fieldnames)):
            raise ValueError(
                "Reference CSV must contain columns: frequency_hz, reference_power_dbm"
            )

        frequency_values: list[float] = []
        reference_values: list[float] = []
        for row in reader:
            frequency_values.append(float(row["frequency_hz"]))
            reference_values.append(float(row["reference_power_dbm"]))

        frequencies_hz, powers_dbm = _normalize_frequency_value_pairs(
            frequency_values, reference_values
        )
        if len(frequencies_hz) == 0:
            raise ValueError("Reference CSV contains no valid rows.")

        self.reference_frequency_hz = frequencies_hz
        self.reference_power_dbm = powers_dbm
        self.reference_loaded = True
        self.last_loaded_reference_csv = str(path)
        self.reset_sequence()
        return CalibrationCsvSummary(
            count=int(len(frequencies_hz)),
            min_frequency_hz=int(frequencies_hz[0]),
            max_frequency_hz=int(frequencies_hz[-1]),
            source_path=str(path),
        )

    def reset_sequence(self) -> None:
        self.sequence_points = []
        self.sequence_index = 0
        self.sequence_retry_waiting = False
        self.sequence_is_measuring = False
        self.measurement_results = []

    def ensure_sequence_initialized(self) -> None:
        if self.sequence_points:
            return
        if self.reference_loaded and len(self.reference_frequency_hz) > 0:
            self.sequence_points = [
                CalibrationSequencePoint(
                    frequency_hz=int(freq_hz),
                    reference_power_dbm=float(ref_dbm),
                )
                for freq_hz, ref_dbm in zip(
                    self.reference_frequency_hz.tolist(),
                    self.reference_power_dbm.tolist(),
                )
            ]
            return
        default_freq_hz = self.build_default_reference_frequency_hz()
        self.sequence_points = [
            CalibrationSequencePoint(
                frequency_hz=int(freq_hz),
                reference_power_dbm=0.0,
            )
            for freq_hz in default_freq_hz.tolist()
        ]

    def current_sequence_point(self) -> CalibrationSequencePoint | None:
        self.ensure_sequence_initialized()
        if self.sequence_index < 0 or self.sequence_index >= len(self.sequence_points):
            return None
        return self.sequence_points[self.sequence_index]

    def sequence_completed(self) -> bool:
        self.ensure_sequence_initialized()
        return self.sequence_index >= len(self.sequence_points)

    def mark_measurement_started(self) -> None:
        self.sequence_is_measuring = True

    def mark_measurement_finished(self) -> None:
        self.sequence_is_measuring = False

    def mark_retry_waiting(self) -> None:
        self.sequence_retry_waiting = True

    def clear_retry_waiting(self) -> None:
        self.sequence_retry_waiting = False

    def append_result_and_advance(
        self,
        *,
        measured_power_dbm: float,
        calibration_offset_db: float,
    ) -> CalibrationPointResult:
        point = self.current_sequence_point()
        if point is None:
            raise ValueError("No calibration sequence point is active.")
        result = CalibrationPointResult(
            frequency_hz=int(point.frequency_hz),
            measured_power_dbm=float(measured_power_dbm),
            reference_power_dbm=float(point.reference_power_dbm),
            calibration_offset_db=float(calibration_offset_db),
        )
        self.measurement_results.append(result)
        self.sequence_index += 1
        self.sequence_retry_waiting = False
        return result

    @staticmethod
    def build_default_reference_frequency_hz() -> np.ndarray:
        return np.arange(
            MIN_CALIBRATION_FREQ_HZ,
            MAX_CALIBRATION_FREQ_HZ + DEFAULT_CALIBRATION_STEP_HZ,
            DEFAULT_CALIBRATION_STEP_HZ,
            dtype=np.int64,
        )
