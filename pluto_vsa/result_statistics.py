"""Incremental all-packet statistics for VSA Result Summary values."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import re

import numpy as np


_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class _NumericSpec:
    unit: str
    decimals: int
    mode: str = "mean"
    weighted: bool = False


_NUMERIC_SPECS = {
    "power": _NumericSpec("dBm", 2, "power", True),
    "carrier_frequency_error": _NumericSpec("kHz", 3),
    "estimated_carrier": _NumericSpec("MHz", 6),
    "evm_rms": _NumericSpec("%", 2, "rms", True),
    "differential_symbol_evm_rms": _NumericSpec("%", 2, "rms", True),
    "bluetooth_devm_rms": _NumericSpec("%", 2, "rms", True),
    "symbol_rate_error": _NumericSpec("ppm", 2),
    "frequency_error_rms": _NumericSpec("%", 2, "rms", True),
    "fsk_deviation_error": _NumericSpec("Hz", 0),
    "fsk_measured_deviation": _NumericSpec("kHz", 3),
    "fsk_reference_deviation": _NumericSpec("kHz", 3),
    "carrier_frequency_drift": _NumericSpec("Hz/Sym", 3),
    "iq_correlation": _NumericSpec("%", 2),
    "result_symbols": _NumericSpec("symbols", 1),
    "psk_carrier_drift": _NumericSpec("kHz/ms", 3),
    "sync_evm_rms": _NumericSpec("%", 2, "rms", True),
    "frequency_fit_rms": _NumericSpec("kHz", 3, "rms"),
    "timing_confidence": _NumericSpec("", 3),
    "deviation_error_percent": _NumericSpec("%", 2),
    "applied_drift": _NumericSpec("kHz/ms", 3),
}

_CATEGORICAL_IDS = {
    "modulation",
    "pattern_symbols_correct",
    "pattern_match_variant",
    "display",
    "drift_model",
    "pattern_error",
}


@dataclass
class _NumericAccumulator:
    spec: _NumericSpec
    count: int = 0
    total_weight: float = 0.0
    weighted_sum: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float, weight: float) -> None:
        if not np.isfinite(value) or not np.isfinite(weight) or weight <= 0.0:
            return
        transformed = (
            10.0 ** (value / 10.0)
            if self.spec.mode == "power"
            else value * value
            if self.spec.mode == "rms"
            else value
        )
        self.count += 1
        self.total_weight += weight
        self.weighted_sum += transformed * weight
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def display(self) -> str:
        if self.count == 0 or self.total_weight <= 0.0:
            return "—"
        mean = self.weighted_sum / self.total_weight
        if self.spec.mode == "power":
            mean = 10.0 * math.log10(mean)
        elif self.spec.mode == "rms":
            mean = math.sqrt(max(0.0, mean))
        decimals = self.spec.decimals
        sign = "+" if self.spec.unit not in {"symbols", "%"} else ""
        average = f"{mean:{sign}.{decimals}f}"
        minimum = f"{self.minimum:{sign}.{decimals}f}"
        maximum = f"{self.maximum:{sign}.{decimals}f}"
        unit = f" {self.spec.unit}" if self.spec.unit else ""
        return f"{average} [{minimum} … {maximum}]{unit} (N={self.count})"


@dataclass
class ResultSummaryAccumulator:
    """Keep bounded statistics without retaining captures or packet sessions."""

    packet_count: int = 0
    _numeric: dict[str, _NumericAccumulator] = field(default_factory=dict)
    _categorical: dict[str, Counter[str]] = field(default_factory=dict)

    def clear(self) -> None:
        self.packet_count = 0
        self._numeric.clear()
        self._categorical.clear()

    def add(self, values: dict[str, str]) -> None:
        self.packet_count += 1
        symbol_weight = self._first_number(values.get("result_symbols", "")) or 1.0
        for item_id, display in values.items():
            spec = _NUMERIC_SPECS.get(item_id)
            if spec is not None:
                value = self._first_number(display)
                if value is None:
                    continue
                accumulator = self._numeric.setdefault(
                    item_id, _NumericAccumulator(spec)
                )
                accumulator.add(value, symbol_weight if spec.weighted else 1.0)
            elif item_id in _CATEGORICAL_IDS and display and display != "—":
                self._categorical.setdefault(item_id, Counter())[display] += 1

    def values(self) -> dict[str, str]:
        result = {
            item_id: accumulator.display()
            for item_id, accumulator in self._numeric.items()
        }
        for item_id, counts in self._categorical.items():
            if item_id == "pattern_symbols_correct":
                yes = counts.get("Yes", 0)
                result[item_id] = f"Yes {yes}/{sum(counts.values())}"
                continue
            ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            result[item_id] = ", ".join(
                f"{value}: {count}" for value, count in ordered
            )
            result[item_id] += f" (N={sum(counts.values())})"
        result["match_selection"] = f"{self.packet_count} packet(s)"
        return result

    @staticmethod
    def _first_number(display: str) -> float | None:
        match = _NUMBER.search(str(display))
        if match is None:
            return None
        value = float(match.group(0))
        return value if np.isfinite(value) else None


__all__ = ["ResultSummaryAccumulator"]
