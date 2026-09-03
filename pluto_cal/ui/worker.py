"""Background worker for the blocking Pluto calibration sequence."""

from __future__ import annotations

import threading

from PySide6 import QtCore

from pluto_cal.frequency.backend import PlutoFrequencyBackend
from pluto_cal.frequency.measurement import measure_frequency
from pluto_cal.frequency.optimizer import CalibrationCancelled, FrequencyCalibrator
from pluto_cal.frequency.persistence import (
    SSHXOCorrectionPersistence,
)
from pluto_cal.model import FrequencyCalibrationConfig


class FrequencyCheckWorker(QtCore.QThread):
    """Measure the current frequency error without changing or saving XO."""

    state_changed = QtCore.Signal(str, str)
    measurement_ready = QtCore.Signal(object, int)
    current_xo_changed = QtCore.Signal(int)
    check_complete = QtCore.Signal(object)
    check_failed = QtCore.Signal(str)

    def __init__(
        self,
        device_target: str | None,
        config: FrequencyCalibrationConfig,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.device_target = device_target
        self.config = config
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _capture(self, backend: PlutoFrequencyBackend):
        if self._cancel_event.is_set():
            raise CalibrationCancelled("Frequency check cancelled")
        return backend.capture_iq()

    def run(self) -> None:
        backend = None
        try:
            self.state_changed.emit(
                "SIGNAL_CHECK", "Checking the CW without changing XO correction"
            )
            backend = PlutoFrequencyBackend.open(self.device_target, self.config)
            current_xo = backend.get_xo_correction()
            self.current_xo_changed.emit(current_xo)
            measurement = measure_frequency(
                lambda: self._capture(backend),
                xo_correction=current_xo,
                config=self.config,
            )
            if self._cancel_event.is_set():
                raise CalibrationCancelled("Frequency check cancelled")
            self.measurement_ready.emit(measurement, 0)
            self.state_changed.emit(
                "COMPLETE", "Frequency check complete; XO correction was not changed"
            )
            self.check_complete.emit(measurement)
        except Exception as error:
            self.state_changed.emit("FAILED", str(error))
            self.check_failed.emit(str(error))
        finally:
            if backend is not None:
                backend.close()


class FrequencyCalibrationWorker(QtCore.QThread):
    state_changed = QtCore.Signal(str, str)
    measurement_ready = QtCore.Signal(object, int)
    current_xo_changed = QtCore.Signal(int)
    calibration_complete = QtCore.Signal(object)
    calibration_failed = QtCore.Signal(str)

    def __init__(
        self,
        device_target: str | None,
        config: FrequencyCalibrationConfig,
        ssh_password: str | None = "analog",
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.device_target = device_target
        self.config = config
        self.ssh_password = ssh_password
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        backend = None
        try:
            backend = PlutoFrequencyBackend.open(self.device_target, self.config)
            if backend.persistence_host is None:
                raise RuntimeError(
                    "No network endpoint matches the selected Pluto serial; "
                    "connect/select its IP context before calibration so the "
                    "verified result can be persisted safely"
                )
            persistence = SSHXOCorrectionPersistence(
                backend.persistence_host,
                expected_serial=backend.device_serial,
                password=self.ssh_password,
            )
            calibrator = FrequencyCalibrator(
                backend,
                persistence,
                self.config,
                cancel_event=self._cancel_event,
                state_callback=lambda state, message: self.state_changed.emit(
                    state.value, message
                ),
                measurement_callback=lambda measurement, iteration: (
                    self.measurement_ready.emit(measurement, iteration)
                ),
                xo_callback=self.current_xo_changed.emit,
            )
            result = calibrator.run()
        except Exception as error:
            if backend is not None:
                backend.close()
            self.calibration_failed.emit(str(error))
            return
        self.calibration_complete.emit(result)
