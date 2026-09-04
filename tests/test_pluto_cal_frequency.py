import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import subprocess
import threading

import numpy as np
import pytest
from PySide6 import QtWidgets

from pluto_cal.frequency.backend import (
    PlutoFrequencyBackend,
    XOCorrectionRangeError,
    parse_xo_correction_available,
)
from pluto_cal.frequency.measurement import (
    CWDetectionError,
    MeasurementQualityError,
    estimate_cw_frequency,
    measure_frequency,
)
from pluto_cal.frequency.optimizer import (
    CalibrationCancelled,
    CalibrationRunError,
    FrequencyCalibrator,
    XOOptimizer,
    calculate_xo_candidate,
    has_converged,
)
from pluto_cal.frequency.persistence import (
    FallbackSSHXOCorrectionPersistence,
    normalize_ssh_host,
    PersistenceError,
    SSHXOCorrectionPersistence,
)
from pluto_cal.model import (
    CalibrationState,
    FrequencyCalibrationConfig,
    FrequencyMeasurement,
)
from pluto_cal.ui.main_window import PlutoCalMainWindow
from pluto_cal.ui.worker import FrequencyCalibrationWorker, FrequencyCheckWorker


def _tone(
    frequency_hz: float,
    *,
    sample_rate_hz: float = 4_000_000.0,
    count: int = 32_768,
    noise_rms: float = 0.002,
    seed: int = 1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time_s = np.arange(count, dtype=np.float64) / sample_rate_hz
    noise = noise_rms * (
        rng.standard_normal(count) + 1j * rng.standard_normal(count)
    )
    return 0.35 * np.exp(2j * np.pi * frequency_hz * time_s) + noise


def _measurement(xo: int, error_hz: float) -> FrequencyMeasurement:
    reference = 2_440_000_000.0
    return FrequencyMeasurement(
        xo_correction=xo,
        measured_if_hz=500_000.0 + error_hz,
        measured_frequency_hz=reference + error_hz,
        frequency_error_hz=error_hz,
        frequency_error_ppm=error_hz / reference * 1e6,
        snr_db=40.0,
        spread_hz=0.1,
    )


def test_cw_estimator_resolves_below_fft_bin_width() -> None:
    expected_hz = 500_123.456
    estimate = estimate_cw_frequency(
        _tone(expected_hz),
        4_000_000.0,
        expected_frequency_hz=500_000.0,
    )
    assert estimate.frequency_hz == pytest.approx(expected_hz, abs=0.5)
    assert estimate.snr_db > 30.0


def test_cw_estimator_rejects_missing_signal() -> None:
    with pytest.raises(CWDetectionError):
        estimate_cw_frequency(
            np.random.default_rng(2).standard_normal(4096).astype(np.complex128),
            4_000_000.0,
            expected_frequency_hz=500_000.0,
            minimum_snr_db=30.0,
        )


def test_repeated_measurement_rejects_excessive_frequency_spread() -> None:
    frequencies = iter((499_900.0, 500_000.0, 500_100.0))
    config = FrequencyCalibrationConfig(
        captures_per_measurement=3,
        maximum_frequency_spread_hz=10.0,
    )
    with pytest.raises(MeasurementQualityError):
        measure_frequency(
            lambda: _tone(next(frequencies), count=8192, noise_rms=0.0001),
            xo_correction=40_000_000,
            config=config,
        )


def test_xo_candidate_uses_reference_ratio_and_limits_range() -> None:
    candidate = calculate_xo_candidate(
        40_000_000,
        2_440_000_000.0,
        2_440_006_100.0,
        (39_999_000, 40_001_000),
    )
    assert candidate == 39_999_900
    assert calculate_xo_candidate(
        40_000_000, 2_440_000_000.0, 2_000_000_000.0, (39_999_000, 40_001_000)
    ) == 40_001_000


def test_convergence_accounts_for_integer_xo_quantization() -> None:
    assert has_converged(
        25.0,
        reference_frequency_hz=2_440_000_000.0,
        xo_correction=40_000_000,
        requested_error_hz=1.0,
    )
    assert not has_converged(
        40.0,
        reference_frequency_hz=2_440_000_000.0,
        xo_correction=40_000_000,
        requested_error_hz=1.0,
    )


def test_optimizer_keeps_best_during_divergence_and_uses_local_search() -> None:
    optimizer = XOOptimizer((39_999_000, 40_001_000), local_initial_step=8)
    assert optimizer.observe(_measurement(40_000_000, 500.0))
    assert optimizer.observe(_measurement(39_999_992, 100.0))
    assert not optimizer.observe(_measurement(39_999_984, 900.0))
    assert optimizer.best is not None
    assert optimizer.best.xo_correction == 39_999_992
    candidate = optimizer.next_candidate(
        optimizer.observations[39_999_984],
        reference_frequency_hz=2_440_000_000.0,
    )
    assert candidate is not None
    assert 39_999_000 <= candidate <= 40_001_000


def test_backend_rejects_runtime_xo_outside_available_range(monkeypatch) -> None:
    class Attribute:
        value = "40000000"

    backend = object.__new__(PlutoFrequencyBackend)
    backend._xo_correction_range = (39_999_900, 40_000_100)
    backend._xo_attribute = Attribute()
    backend.config = FrequencyCalibrationConfig(settle_time_s=0.001)
    backend._discard_capture = lambda: None
    monkeypatch.setattr("pluto_cal.frequency.backend.time.sleep", lambda _value: None)
    with pytest.raises(XOCorrectionRangeError):
        backend.set_xo_correction(40_000_101)
    assert parse_xo_correction_available("[39999900 1 40000100]") == (
        39_999_900,
        40_000_100,
    )


def test_backend_prefers_same_serial_ip_context_for_persistence() -> None:
    backend = object.__new__(PlutoFrequencyBackend)
    backend.device_serial = "selected-serial"
    backend._network_hosts = {"selected-serial": "192.168.10.42"}
    assert backend.persistence_hosts == (
        "192.168.10.42",
        "pluto.local",
        "192.168.2.1",
    )


def test_backend_uses_standard_ssh_hosts_without_ip_context() -> None:
    backend = object.__new__(PlutoFrequencyBackend)
    backend.device_serial = "selected-serial"
    backend._network_hosts = {}
    assert backend.persistence_hosts == ("pluto.local", "192.168.2.1")


def test_ssh_persistence_writes_only_then_reads_back() -> None:
    commands: list[tuple[str, ...]] = []

    def runner(command, **_kwargs):
        commands.append(tuple(command))
        output = "40000012\n" if "fw_printenv" in command[-1] else ""
        return subprocess.CompletedProcess(command, 0, output, "")

    persistence = SSHXOCorrectionPersistence("pluto.local", runner=runner)
    assert persistence.persist(40_000_012) == 40_000_012
    assert "fw_setenv xo_correction 40000012" in commands[0][-1]
    assert "fw_printenv -n xo_correction" in commands[1][-1]


def test_ssh_persistence_reports_write_failure_without_readback() -> None:
    calls = 0

    def runner(command, **_kwargs):
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 1, "", "permission denied")

    with pytest.raises(PersistenceError, match="permission denied"):
        SSHXOCorrectionPersistence("pluto.local", runner=runner).persist(40_000_000)
    assert calls == 1


def test_ssh_persistence_checks_selected_pluto_serial_before_write() -> None:
    commands: list[str] = []

    def runner(command, **_kwargs):
        commands.append(command[-1])
        return subprocess.CompletedProcess(command, 0, "different-serial\n", "")

    persistence = SSHXOCorrectionPersistence(
        "pluto.local", expected_serial="selected-serial", runner=runner
    )
    with pytest.raises(PersistenceError, match="does not match"):
        persistence.persist(40_000_000)
    assert commands == ["cat /etc/serial"]


def test_fallback_ssh_uses_usb_address_after_pluto_local_connection_failure() -> None:
    commands: list[tuple[str, str]] = []

    def runner(command, **_kwargs):
        host, remote_command = command[-2], command[-1]
        commands.append((host, remote_command))
        if host == "root@pluto.local":
            return subprocess.CompletedProcess(command, 255, "", "connection failed")
        if remote_command == "cat /etc/serial":
            output = "selected-serial\n"
        elif remote_command == "fw_printenv -n xo_correction":
            output = "40000012\n"
        else:
            output = ""
        return subprocess.CompletedProcess(command, 0, output, "")

    persistence = FallbackSSHXOCorrectionPersistence(
        ("pluto.local", "192.168.2.1"),
        expected_serial="selected-serial",
        runner=runner,
    )
    assert persistence.persist(40_000_012) == 40_000_012
    assert commands[0] == ("root@pluto.local", "cat /etc/serial")
    assert commands[1] == ("root@192.168.2.1", "cat /etc/serial")
    assert any(command.startswith("fw_setenv") for _host, command in commands)


def test_fallback_ssh_matching_serial_allows_persistence() -> None:
    commands: list[str] = []

    def runner(command, **_kwargs):
        commands.append(command[-1])
        output = {
            "cat /etc/serial": "selected-serial\n",
            "fw_printenv -n xo_correction": "40000012\n",
        }.get(command[-1], "")
        return subprocess.CompletedProcess(command, 0, output, "")

    persistence = FallbackSSHXOCorrectionPersistence(
        ("pluto.local",), expected_serial="selected-serial", runner=runner
    )
    assert persistence.persist(40_000_012) == 40_000_012
    assert commands == [
        "cat /etc/serial",
        "fw_setenv xo_correction 40000012",
        "fw_printenv -n xo_correction",
    ]


def test_fallback_ssh_serial_mismatch_never_writes_or_tries_next_host() -> None:
    commands: list[tuple[str, str]] = []

    def runner(command, **_kwargs):
        commands.append((command[-2], command[-1]))
        return subprocess.CompletedProcess(command, 0, "different-serial\n", "")

    persistence = FallbackSSHXOCorrectionPersistence(
        ("pluto.local", "192.168.2.1"),
        expected_serial="selected-serial",
        runner=runner,
    )
    with pytest.raises(PersistenceError, match="does not match"):
        persistence.persist(40_000_012)
    assert commands == [("root@pluto.local", "cat /etc/serial")]


def test_fallback_ssh_unreadable_serial_never_writes_or_tries_next_host() -> None:
    commands: list[tuple[str, str]] = []

    def runner(command, **_kwargs):
        commands.append((command[-2], command[-1]))
        return subprocess.CompletedProcess(command, 1, "", "serial unavailable")

    persistence = FallbackSSHXOCorrectionPersistence(
        ("pluto.local", "192.168.2.1"),
        expected_serial="selected-serial",
        runner=runner,
    )
    with pytest.raises(PersistenceError, match="serial unavailable"):
        persistence.persist(40_000_012)
    assert commands == [("root@pluto.local", "cat /etc/serial")]


def test_fallback_ssh_connection_failure_never_writes() -> None:
    commands: list[str] = []

    def runner(command, **_kwargs):
        commands.append(command[-1])
        return subprocess.CompletedProcess(command, 255, "", "connection refused")

    persistence = FallbackSSHXOCorrectionPersistence(
        ("pluto.local", "192.168.2.1"),
        expected_serial="selected-serial",
        runner=runner,
    )
    with pytest.raises(PersistenceError, match="Unable to connect"):
        persistence.persist(40_000_012)
    assert commands == ["cat /etc/serial", "cat /etc/serial"]


def test_scoped_ipv6_ssh_host_uses_socket_interface_index(monkeypatch) -> None:
    monkeypatch.setattr("socket.if_nametoindex", lambda name: 9)
    assert normalize_ssh_host("fe80::1%ethernet_32773") == "fe80::1%9"


class _MockPluto:
    def __init__(self, config: FrequencyCalibrationConfig) -> None:
        self.config = config
        self.current_xo = 40_000_000
        self.optimal_xo = 40_000_120
        self.xo_correction_range = (39_999_000, 40_001_000)
        self.set_history: list[int] = []
        self.capture_index = 0
        self.closed = False

    def get_xo_correction(self) -> int:
        return self.current_xo

    def set_xo_correction(self, value: int) -> None:
        lower, upper = self.xo_correction_range
        if not lower <= value <= upper:
            raise XOCorrectionRangeError
        self.current_xo = int(value)
        self.set_history.append(int(value))

    def capture_iq(self) -> np.ndarray:
        measured_rf_hz = (
            self.config.reference_frequency_hz
            * self.current_xo
            / self.optimal_xo
        )
        measured_if_hz = (
            self.config.if_offset_hz
            + measured_rf_hz
            - self.config.reference_frequency_hz
        )
        self.capture_index += 1
        return _tone(measured_if_hz, count=16_384, seed=self.capture_index)

    def close(self) -> None:
        self.closed = True


class _MemoryPersistence:
    def __init__(self, *, fail: bool = False) -> None:
        self.values: list[int] = []
        self.fail = fail

    def persist(self, value: int, *, before_write=None) -> int:
        if before_write is not None:
            before_write()
        self.values.append(int(value))
        if self.fail:
            raise PersistenceError("injected persistence failure")
        return int(value)


def _fast_config() -> FrequencyCalibrationConfig:
    return FrequencyCalibrationConfig(
        captures_per_measurement=3,
        verification_captures=5,
        maximum_frequency_spread_hz=15.0,
        maximum_iterations=8,
    )


def test_mock_pluto_runs_complete_calibration_and_persists_best() -> None:
    config = _fast_config()
    backend = _MockPluto(config)
    persistence = _MemoryPersistence()
    states: list[CalibrationState] = []
    result = FrequencyCalibrator(
        backend,
        persistence,
        config,
        state_callback=lambda state, _message: states.append(state),
    ).run()
    assert result.state is CalibrationState.COMPLETE
    assert result.best_xo_correction == backend.optimal_xo
    assert abs(result.best_frequency_error_hz) < 1.0
    assert backend.current_xo == backend.optimal_xo
    assert persistence.values == [backend.optimal_xo]
    assert backend.closed
    assert states[-3:] == [
        CalibrationState.VERIFY,
        CalibrationState.PERSIST,
        CalibrationState.COMPLETE,
    ]


def test_cancel_rolls_back_runtime_xo_and_never_persists() -> None:
    config = _fast_config()
    backend = _MockPluto(config)
    persistence = _MemoryPersistence()
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(CalibrationCancelled):
        FrequencyCalibrator(
            backend, persistence, config, cancel_event=cancel
        ).run()
    assert backend.current_xo == 40_000_000
    assert backend.set_history[-1] == 40_000_000
    assert persistence.values == []
    assert backend.closed


def test_persistence_failure_rolls_back_runtime_xo() -> None:
    config = _fast_config()
    backend = _MockPluto(config)
    persistence = _MemoryPersistence(fail=True)
    with pytest.raises(CalibrationRunError, match="injected persistence failure"):
        FrequencyCalibrator(backend, persistence, config).run()
    assert backend.current_xo == 40_000_000
    assert backend.set_history[-1] == 40_000_000
    assert backend.closed


def test_cancel_before_nonvolatile_write_rolls_back_without_writing() -> None:
    config = _fast_config()
    backend = _MockPluto(config)
    cancel = threading.Event()

    class CancellingPersistence:
        wrote = False

        def persist(self, value: int, *, before_write=None) -> int:
            cancel.set()
            if before_write is not None:
                before_write()
            self.wrote = True
            return value

    persistence = CancellingPersistence()
    with pytest.raises(CalibrationCancelled):
        FrequencyCalibrator(
            backend, persistence, config, cancel_event=cancel
        ).run()
    assert persistence.wrote is False
    assert backend.current_xo == 40_000_000
    assert backend.closed


def test_pluto_cal_window_exposes_frequency_calibration_controls(monkeypatch) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr("pluto_cal.ui.main_window.iio.scan_contexts", lambda: {})
    window = PlutoCalMainWindow()
    try:
        assert window.windowTitle() == "Pluto CAL"
        assert window.frequency_spin.value() == pytest.approx(2440.0)
        assert window.measure_button.text() == "Measure Frequency Error"
        assert window.start_button.text() == "Start Calibration"
        assert window.cancel_button.text() == "Cancel"
        assert window.status_state_label.text() == "IDLE"
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


@pytest.mark.parametrize(
    ("start_method", "worker_class"),
    (
        ("start_frequency_check", FrequencyCheckWorker),
        ("start_calibration", FrequencyCalibrationWorker),
    ),
)
def test_new_run_clears_previous_measurement_values(
    monkeypatch, start_method, worker_class
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr("pluto_cal.ui.main_window.iio.scan_contexts", lambda: {})
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *_args, **_kwargs: QtWidgets.QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(worker_class, "start", lambda _worker: None)
    window = PlutoCalMainWindow()
    try:
        labels = (
            window.current_xo_label,
            window.error_hz_label,
            window.error_ppm_label,
            window.best_xo_label,
            window.best_error_label,
        )
        for label in labels:
            label.setText("old result")
        getattr(window, start_method)()
        assert [label.text() for label in labels] == ["—"] * 5
        assert window.status_state_label.text() == "SIGNAL_CHECK"
        assert "checking the CW" in window.status_label.text()
    finally:
        if window._worker is not None:
            window._worker.deleteLater()
            window._worker = None
        window.close()
        window.deleteLater()
        app.processEvents()


def test_frequency_check_worker_does_not_change_or_persist_xo(monkeypatch) -> None:
    config = _fast_config()
    backend = _MockPluto(config)
    measurements: list[FrequencyMeasurement] = []
    completed: list[FrequencyMeasurement] = []
    monkeypatch.setattr(
        "pluto_cal.ui.worker.PlutoFrequencyBackend.open",
        lambda _target, _config: backend,
    )
    worker = FrequencyCheckWorker(None, config)
    worker.measurement_ready.connect(
        lambda measurement, _iteration: measurements.append(measurement)
    )
    worker.check_complete.connect(completed.append)

    worker.run()

    assert len(measurements) == 1
    assert completed == measurements
    assert backend.current_xo == 40_000_000
    assert backend.set_history == []
    assert backend.closed
