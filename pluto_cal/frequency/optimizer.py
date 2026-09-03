"""Bounded XO search and safe calibration state machine."""

from __future__ import annotations

from collections.abc import Callable
import math
import threading

from pluto_cal.model import (
    CalibrationSample,
    CalibrationState,
    FrequencyCalibrationConfig,
    FrequencyCalibrationResult,
    FrequencyMeasurement,
)

from .backend import FrequencyBackend
from .measurement import measure_frequency
from .persistence import XOCorrectionPersistence


class CalibrationCancelled(RuntimeError):
    """Raised after a user cancellation has safely restored runtime state."""


class CalibrationRunError(RuntimeError):
    """Raised after an unsuccessful run has safely restored runtime state."""


def clamp_xo_correction(value: int, bounds: tuple[int, int]) -> int:
    lower, upper = (int(bounds[0]), int(bounds[1]))
    if lower > upper:
        raise ValueError("XO correction bounds are reversed")
    return min(upper, max(lower, int(value)))


def calculate_xo_candidate(
    current_xo: int,
    reference_frequency_hz: float,
    measured_frequency_hz: float,
    bounds: tuple[int, int],
) -> int:
    """Apply X_new = X * f_ref / f_meas, rounded and range limited."""

    reference = float(reference_frequency_hz)
    measured = float(measured_frequency_hz)
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference_frequency_hz must be positive and finite")
    if not math.isfinite(measured) or measured <= 0.0:
        raise ValueError("measured_frequency_hz must be positive and finite")
    candidate = int(round(int(current_xo) * reference / measured))
    return clamp_xo_correction(candidate, bounds)


def xo_quantization_error_hz(
    reference_frequency_hz: float, xo_correction: int
) -> float:
    """Approximate half-LSB RF error for integer XO correction values."""

    return abs(float(reference_frequency_hz) / max(abs(int(xo_correction)), 1)) / 2.0


def has_converged(
    frequency_error_hz: float,
    *,
    reference_frequency_hz: float,
    xo_correction: int,
    requested_error_hz: float,
) -> bool:
    limit = max(
        float(requested_error_hz),
        xo_quantization_error_hz(reference_frequency_hz, xo_correction),
    )
    return abs(float(frequency_error_hz)) <= limit


class XOOptimizer:
    """Remember every result and fall back to a shrinking local search."""

    def __init__(self, bounds: tuple[int, int], *, local_initial_step: int = 16):
        self.bounds = (int(bounds[0]), int(bounds[1]))
        if self.bounds[0] > self.bounds[1]:
            raise ValueError("XO correction bounds are reversed")
        self.local_step = max(1, int(local_initial_step))
        self.observations: dict[int, FrequencyMeasurement] = {}
        self.best: FrequencyMeasurement | None = None

    def observe(self, measurement: FrequencyMeasurement) -> bool:
        xo = int(measurement.xo_correction)
        if not self.bounds[0] <= xo <= self.bounds[1]:
            raise ValueError("Observed XO correction is outside device bounds")
        existing = self.observations.get(xo)
        if existing is None or abs(measurement.frequency_error_hz) < abs(
            existing.frequency_error_hz
        ):
            self.observations[xo] = measurement
        candidate = self.observations[xo]
        improved = self.best is None or abs(candidate.frequency_error_hz) < abs(
            self.best.frequency_error_hz
        )
        if improved:
            self.best = candidate
        return improved

    def next_candidate(
        self,
        current: FrequencyMeasurement,
        *,
        reference_frequency_hz: float,
    ) -> int | None:
        direct = calculate_xo_candidate(
            current.xo_correction,
            reference_frequency_hz,
            current.measured_frequency_hz,
            self.bounds,
        )
        if direct != current.xo_correction and direct not in self.observations:
            return direct
        if self.best is None:
            return None
        while self.local_step >= 1:
            best_xo = int(self.best.xo_correction)
            for candidate in (best_xo - self.local_step, best_xo + self.local_step):
                candidate = clamp_xo_correction(candidate, self.bounds)
                if candidate not in self.observations:
                    return candidate
            if self.local_step == 1:
                return None
            self.local_step = max(1, self.local_step // 2)
        return None


StateCallback = Callable[[CalibrationState, str], None]
MeasurementCallback = Callable[[FrequencyMeasurement, int], None]
XOCallback = Callable[[int], None]


class FrequencyCalibrator:
    """Run calibration without UI dependencies and with deterministic rollback."""

    def __init__(
        self,
        backend: FrequencyBackend,
        persistence: XOCorrectionPersistence,
        config: FrequencyCalibrationConfig,
        *,
        cancel_event: threading.Event | None = None,
        state_callback: StateCallback | None = None,
        measurement_callback: MeasurementCallback | None = None,
        xo_callback: XOCallback | None = None,
    ) -> None:
        self.backend = backend
        self.persistence = persistence
        self.config = config
        self.cancel_event = cancel_event or threading.Event()
        self.state_callback = state_callback
        self.measurement_callback = measurement_callback
        self.xo_callback = xo_callback
        self.state = CalibrationState.IDLE
        self.samples: list[CalibrationSample] = []

    def _set_state(self, state: CalibrationState, message: str) -> None:
        self.state = state
        if self.state_callback is not None:
            self.state_callback(state, message)

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise CalibrationCancelled("Calibration cancelled")

    def _measure(self, xo_correction: int, iteration: int, count: int) -> FrequencyMeasurement:
        self._check_cancelled()
        measurement = measure_frequency(
            self.backend.capture_iq,
            xo_correction=xo_correction,
            config=self.config,
            capture_count=count,
        )
        self.samples.append(CalibrationSample(iteration, measurement))
        if self.measurement_callback is not None:
            self.measurement_callback(measurement, iteration)
        self._check_cancelled()
        return measurement

    def _rollback(self, original_xo: int) -> None:
        self._set_state(CalibrationState.ROLLBACK, "Restoring original runtime XO correction")
        self._set_runtime_xo(original_xo)

    def _set_runtime_xo(self, value: int) -> None:
        self.backend.set_xo_correction(value)
        if self.xo_callback is not None:
            self.xo_callback(int(value))

    def run(self) -> FrequencyCalibrationResult:
        original_xo: int | None = None
        optimizer: XOOptimizer | None = None
        try:
            original_xo = self.backend.get_xo_correction()
            if self.xo_callback is not None:
                self.xo_callback(original_xo)
            lower, upper = self.backend.xo_correction_range
            if not lower <= original_xo <= upper:
                raise CalibrationRunError(
                    f"Current XO correction {original_xo} is outside [{lower}, {upper}]"
                )
            optimizer = XOOptimizer(
                (lower, upper), local_initial_step=self.config.local_initial_step
            )
            self._set_state(
                CalibrationState.SIGNAL_CHECK,
                "Checking the CW at the configured IF",
            )
            current = self._measure(
                original_xo, 0, self.config.captures_per_measurement
            )
            optimizer.observe(current)

            for iteration in range(1, self.config.maximum_iterations + 1):
                assert optimizer.best is not None
                if has_converged(
                    optimizer.best.frequency_error_hz,
                    reference_frequency_hz=self.config.reference_frequency_hz,
                    xo_correction=optimizer.best.xo_correction,
                    requested_error_hz=self.config.convergence_error_hz,
                ):
                    break
                self._check_cancelled()
                candidate = optimizer.next_candidate(
                    current,
                    reference_frequency_hz=self.config.reference_frequency_hz,
                )
                if candidate is None:
                    break
                previous_error = abs(current.frequency_error_hz)
                self._set_state(
                    CalibrationState.ADJUST,
                    f"Setting runtime XO correction to {candidate}",
                )
                self._set_runtime_xo(candidate)
                self._set_state(
                    CalibrationState.MEASURE,
                    f"Measuring candidate {candidate}",
                )
                measured = self._measure(
                    candidate, iteration, self.config.captures_per_measurement
                )
                badly_worse = abs(measured.frequency_error_hz) > max(
                    previous_error * self.config.deterioration_factor,
                    previous_error + self.config.deterioration_floor_hz,
                )
                if badly_worse:
                    self._set_state(
                        CalibrationState.MEASURE,
                        f"Rechecking degraded candidate {candidate}",
                    )
                    measured = self._measure(
                        candidate, iteration, self.config.captures_per_measurement
                    )
                optimizer.observe(measured)
                current = measured

            assert optimizer.best is not None
            best = optimizer.best
            self._set_state(
                CalibrationState.ADJUST,
                f"Restoring best runtime XO correction {best.xo_correction}",
            )
            self._set_runtime_xo(best.xo_correction)
            self._set_state(
                CalibrationState.VERIFY,
                "Verifying the best XO correction with repeated captures",
            )
            verified = self._measure(
                best.xo_correction,
                self.config.maximum_iterations + 1,
                self.config.verification_captures,
            )
            verification_limit_hz = max(
                self.config.convergence_error_hz,
                xo_quantization_error_hz(
                    self.config.reference_frequency_hz, best.xo_correction
                ),
            ) + self.config.maximum_frequency_spread_hz
            if abs(verified.frequency_error_hz) > (
                abs(best.frequency_error_hz) + verification_limit_hz
            ):
                raise CalibrationRunError(
                    "Final XO verification is inconsistent with the best measurement"
                )
            self._check_cancelled()
            self._set_state(
                CalibrationState.PERSIST,
                "Writing and reading back persistent XO correction",
            )
            self.persistence.persist(
                best.xo_correction, before_write=self._check_cancelled
            )
            self._set_state(CalibrationState.COMPLETE, "Calibration complete")
            return FrequencyCalibrationResult(
                state=CalibrationState.COMPLETE,
                original_xo_correction=original_xo,
                best_xo_correction=best.xo_correction,
                best_frequency_error_hz=verified.frequency_error_hz,
                best_frequency_error_ppm=verified.frequency_error_ppm,
                persisted=True,
                verified=True,
                samples=tuple(self.samples),
                message="Calibration complete",
            )
        except BaseException as error:
            rollback_error: BaseException | None = None
            if original_xo is not None:
                try:
                    self._rollback(original_xo)
                except BaseException as caught:
                    rollback_error = caught
            self._set_state(CalibrationState.FAILED, str(error))
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            detail = str(error)
            if rollback_error is not None:
                detail += f"; runtime rollback also failed: {rollback_error}"
            exception_type = (
                CalibrationCancelled
                if isinstance(error, CalibrationCancelled)
                else CalibrationRunError
            )
            raise exception_type(detail) from error
        finally:
            self.backend.close()
