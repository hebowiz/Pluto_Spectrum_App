"""Main window for the standalone Pluto CAL application."""

from __future__ import annotations

import math

import iio
from PySide6 import QtCore, QtGui, QtWidgets

from pluto_cal.model import FrequencyCalibrationConfig, FrequencyMeasurement
from pluto_common import discover_pluto_devices

from .worker import FrequencyCalibrationWorker, FrequencyCheckWorker


DEFAULT_PLUTO_SSH_PASSWORD = "analog"


class PlutoCalMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Pluto CAL")
        self._worker: FrequencyCalibrationWorker | FrequencyCheckWorker | None = None
        self._run_mode: str | None = None
        self._best_error_hz = math.inf
        self._close_when_finished = False
        self._persistence_started = False

        self.device_combo = QtWidgets.QComboBox()
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        device_row = QtWidgets.QWidget()
        device_layout = QtWidgets.QHBoxLayout(device_row)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.addWidget(self.device_combo, 1)
        device_layout.addWidget(self.refresh_button)

        self.frequency_spin = QtWidgets.QDoubleSpinBox()
        self.frequency_spin.setRange(70.0, 6000.0)
        self.frequency_spin.setDecimals(6)
        self.frequency_spin.setSingleStep(1.0)
        self.frequency_spin.setValue(2440.0)
        self.frequency_spin.setSuffix(" MHz")
        self.ssh_password_edit = QtWidgets.QLineEdit(DEFAULT_PLUTO_SSH_PASSWORD)
        self.ssh_password_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.ssh_password_default_button = QtWidgets.QPushButton("Default")
        self.ssh_password_default_button.setToolTip(
            "Enter the standard Pluto SSH password"
        )
        password_row = QtWidgets.QWidget()
        password_layout = QtWidgets.QHBoxLayout(password_row)
        password_layout.setContentsMargins(0, 0, 0, 0)
        password_layout.addWidget(self.ssh_password_edit, 1)
        password_layout.addWidget(self.ssh_password_default_button)

        self.if_label = QtWidgets.QLabel("+500.000 kHz")
        self.rx_lo_label = QtWidgets.QLabel("2439.500000 MHz")
        self.current_xo_label = QtWidgets.QLabel("—")
        self.error_hz_label = QtWidgets.QLabel("—")
        self.error_ppm_label = QtWidgets.QLabel("—")
        self.best_xo_label = QtWidgets.QLabel("—")
        self.best_error_label = QtWidgets.QLabel("—")
        self.status_state_label = QtWidgets.QLabel("IDLE")
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setWordWrap(True)

        form_widget = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(form_widget)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("Pluto Device", device_row)
        form.addRow("Calibration Frequency", self.frequency_spin)
        form.addRow("Measurement IF", self.if_label)
        form.addRow("RX LO", self.rx_lo_label)
        form.addRow("Pluto SSH Password", password_row)
        form.addRow("Current XO Correction", self.current_xo_label)
        form.addRow("Measured Frequency Error [Hz]", self.error_hz_label)
        form.addRow("Frequency Error [ppm]", self.error_ppm_label)
        form.addRow("Best XO Correction", self.best_xo_label)
        form.addRow("Best Frequency Error", self.best_error_label)
        form.addRow("State", self.status_state_label)
        form.addRow("Status", self.status_label)

        self.start_button = QtWidgets.QPushButton("Start Calibration")
        self.start_button.setDefault(True)
        self.measure_button = QtWidgets.QPushButton("Measure Frequency Error")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.measure_button)
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.cancel_button)

        frequency_page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(frequency_page)
        instructions = QtWidgets.QLabel(
            "Connect a stable signal generator CW to the selected Pluto RX. "
            "The signal is measured at +500 kHz IF to avoid the zero-IF DC region."
        )
        instructions.setWordWrap(True)
        page_layout.addWidget(instructions)
        page_layout.addWidget(form_widget)
        page_layout.addStretch(1)
        page_layout.addLayout(button_row)

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(frequency_page, "Frequency Calibration")
        self.setCentralWidget(tabs)

        self.refresh_button.clicked.connect(self.refresh_devices)
        self.frequency_spin.valueChanged.connect(self._update_derived_frequency)
        self.ssh_password_default_button.clicked.connect(
            lambda: self.ssh_password_edit.setText(DEFAULT_PLUTO_SSH_PASSWORD)
        )
        self.measure_button.clicked.connect(self.start_frequency_check)
        self.start_button.clicked.connect(self.start_calibration)
        self.cancel_button.clicked.connect(self.cancel_calibration)
        self.resize(660, 500)
        self._update_derived_frequency()
        self.refresh_devices()

    def refresh_devices(self) -> None:
        had_selection = self.device_combo.count() > 0
        previous = self.device_combo.currentData()
        self.device_combo.clear()
        try:
            devices = discover_pluto_devices(iio.scan_contexts())
        except Exception as error:
            devices = ()
            self.status_label.setText(f"Device scan failed: {error}")
        for device in devices:
            self.device_combo.addItem(device.label, device.selector)
        self.device_combo.addItem("Auto-detect ADALM-Pluto", None)
        if had_selection:
            index = self.device_combo.findData(previous)
            if index >= 0:
                self.device_combo.setCurrentIndex(index)
        elif devices:
            self.device_combo.setCurrentIndex(0)
        if devices:
            self.status_label.setText(f"Found {len(devices)} Pluto device(s)")

    def _update_derived_frequency(self, _value: float | None = None) -> None:
        frequency_mhz = self.frequency_spin.value()
        self.rx_lo_label.setText(f"{frequency_mhz - 0.5:.6f} MHz")

    def _set_running(self, running: bool) -> None:
        self.device_combo.setEnabled(not running)
        self.refresh_button.setEnabled(not running)
        self.frequency_spin.setEnabled(not running)
        self.ssh_password_edit.setEnabled(not running)
        self.ssh_password_default_button.setEnabled(not running)
        self.measure_button.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def _clear_measurement_results(self) -> None:
        self.current_xo_label.setText("—")
        self.error_hz_label.setText("—")
        self.error_ppm_label.setText("—")
        self.best_xo_label.setText("—")
        self.best_error_label.setText("—")

    def start_frequency_check(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        frequency_hz = self.frequency_spin.value() * 1e6
        answer = QtWidgets.QMessageBox.question(
            self,
            "Measure Frequency Error",
            "Set the signal generator to an unmodulated CW at\n\n"
            f"    {frequency_hz / 1e6:.6f} MHz\n\n"
            "Connect it to the selected Pluto RX at a safe input level.\n"
            "The current XO correction will not be changed or saved. Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        config = FrequencyCalibrationConfig(reference_frequency_hz=frequency_hz)
        self._best_error_hz = math.inf
        self._persistence_started = False
        self._run_mode = "measure"
        self._clear_measurement_results()
        self._worker = FrequencyCheckWorker(
            self.device_combo.currentData(), config, self
        )
        self._worker.state_changed.connect(self._on_state_changed)
        self._worker.measurement_ready.connect(self._on_measurement)
        self._worker.current_xo_changed.connect(
            lambda value: self.current_xo_label.setText(str(value))
        )
        self._worker.check_complete.connect(self._on_check_complete)
        self._worker.check_failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._set_running(True)
        self.status_state_label.setText("SIGNAL_CHECK")
        self.status_label.setText("Opening the selected Pluto and checking the CW…")
        self._worker.start()

    def start_calibration(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        frequency_hz = self.frequency_spin.value() * 1e6
        answer = QtWidgets.QMessageBox.question(
            self,
            "Start Frequency Calibration",
            "Set the signal generator to an unmodulated CW at\n\n"
            f"    {frequency_hz / 1e6:.6f} MHz\n\n"
            "Connect it to the selected Pluto RX at a safe input level.\n"
            "The Pluto RX LO will be 500 kHz below the SG frequency.\n\n"
            "After stable final verification, Pluto CAL will write "
            "xo_correction to non-volatile storage over SSH. Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        config = FrequencyCalibrationConfig(reference_frequency_hz=frequency_hz)
        self._best_error_hz = math.inf
        self._persistence_started = False
        self._run_mode = "calibrate"
        self._clear_measurement_results()
        self._worker = FrequencyCalibrationWorker(
            self.device_combo.currentData(),
            config,
            self.ssh_password_edit.text(),
            self,
        )
        self._worker.state_changed.connect(self._on_state_changed)
        self._worker.measurement_ready.connect(self._on_measurement)
        self._worker.current_xo_changed.connect(
            lambda value: self.current_xo_label.setText(str(value))
        )
        self._worker.calibration_complete.connect(self._on_complete)
        self._worker.calibration_failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._set_running(True)
        self.status_state_label.setText("SIGNAL_CHECK")
        self.status_label.setText("Opening the selected Pluto and checking the CW…")
        self._worker.start()

    def cancel_calibration(self) -> None:
        if self._worker is None or not self._worker.isRunning():
            return
        if self._persistence_started:
            self.cancel_button.setEnabled(False)
            self.status_label.setText(
                "Persistent write/read-back is in progress; waiting for it to finish"
            )
            return
        self.cancel_button.setEnabled(False)
        if self._run_mode == "measure":
            self.status_label.setText("Frequency check cancellation requested…")
        else:
            self.status_label.setText("Cancellation requested; restoring runtime XO…")
        self._worker.cancel()

    @QtCore.Slot(str, str)
    def _on_state_changed(self, state: str, message: str) -> None:
        self.status_state_label.setText(state)
        self.status_label.setText(message)
        if state == "PERSIST":
            self._persistence_started = True
            self.cancel_button.setEnabled(False)

    @QtCore.Slot(object, int)
    def _on_measurement(
        self, measurement: FrequencyMeasurement, iteration: int
    ) -> None:
        self.current_xo_label.setText(str(measurement.xo_correction))
        self.error_hz_label.setText(f"{measurement.frequency_error_hz:+.3f} Hz")
        self.error_ppm_label.setText(f"{measurement.frequency_error_ppm:+.6f} ppm")
        if (
            self._run_mode == "calibrate"
            and abs(measurement.frequency_error_hz) < self._best_error_hz
        ):
            self._best_error_hz = abs(measurement.frequency_error_hz)
            self.best_xo_label.setText(str(measurement.xo_correction))
            self.best_error_label.setText(
                f"{measurement.frequency_error_hz:+.3f} Hz"
            )
        self.status_label.setText(
            f"Iteration {iteration}: SNR {measurement.snr_db:.1f} dB, "
            f"spread {measurement.spread_hz:.2f} Hz"
        )

    @QtCore.Slot(object)
    def _on_complete(self, result: object) -> None:
        self.status_state_label.setText("COMPLETE")
        self.current_xo_label.setText(str(result.best_xo_correction))
        self.best_xo_label.setText(str(result.best_xo_correction))
        self.best_error_label.setText(f"{result.best_frequency_error_hz:+.3f} Hz")
        self.status_label.setText("Calibration verified, persisted, and read back")

    @QtCore.Slot(object)
    def _on_check_complete(self, measurement: FrequencyMeasurement) -> None:
        self.status_state_label.setText("COMPLETE")
        self.status_label.setText(
            "Frequency check complete: "
            f"SNR {measurement.snr_db:.1f} dB, "
            f"spread {measurement.spread_hz:.2f} Hz; XO correction unchanged"
        )

    @QtCore.Slot(str)
    def _on_failed(self, message: str) -> None:
        self.status_state_label.setText("FAILED")
        self.status_label.setText(message)

    @QtCore.Slot()
    def _on_worker_finished(self) -> None:
        self._set_running(False)
        worker, self._worker = self._worker, None
        self._run_mode = None
        if worker is not None:
            worker.deleteLater()
        if self._close_when_finished:
            self._close_when_finished = False
            QtCore.QTimer.singleShot(0, self.close)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._close_when_finished = True
            if not self._persistence_started:
                self.cancel_calibration()
            event.ignore()
            return
        super().closeEvent(event)
