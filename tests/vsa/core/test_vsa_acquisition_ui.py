import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
from pluto_vsa.ui.main_window import VSAWindow
from pluto_vsa.pluto_source import CaptureCancelledError
from pluto_vsa.session import VSASession
from pluto_vsa.sources import GeneratedIQSource
from _vsa_ui_test_helpers import _isolated_preferences, _wait_for_background_analysis


def test_pluto_run_single_uses_async_capture_and_updates_session(tmp_path) -> None:
    pg.mkQApp("VSA Pluto UI test")

    class FakePlutoSource:
        def __init__(self) -> None:
            self.settings = None
            self.closed = False

        def capture_single(self, settings, *, cancelled=None, armed=None):
            self.settings = settings
            if armed is not None:
                armed()
            recording, _signal = GeneratedIQSource.fsk(
                symbol_count=64,
                symbol_rate_hz=settings.symbol_rate_hz,
                samples_per_symbol=settings.samples_per_symbol,
            )
            return recording

        def close(self) -> None:
            self.closed = True

    source = FakePlutoSource()
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "pluto-run-single"),
        pluto_source=source,
    )
    try:
        window._run_pluto_single()
        for _index in range(200):
            QtWidgets.QApplication.processEvents()
            thread = window._pluto_capture_thread
            if thread is None:
                break
            thread.wait(10)

        _wait_for_background_analysis(window)

        assert window._pluto_capture_thread is None
        assert source.settings is not None
        assert source.settings.samples_per_symbol == 8
        assert source.settings.requested_sample_rate_hz == 8_000_000
        assert source.settings.capture_samples == 24_000
        assert window.input_source_combo.currentText() == "Pluto"
        assert window.run_single_action.isEnabled()
        assert window.session.recording.sample_rate_hz == 8_000_000.0
        status = window.statusBar().currentMessage()
        assert "Capture" in status
        assert "DSP" in status
        assert "Display" in status
        assert "Total" in status
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()
    assert source.closed


def test_pluto_continuous_applies_backpressure_and_updates_all_packets(
    tmp_path,
) -> None:
    pg.mkQApp("VSA Pluto Continuous UI test")

    class FakePlutoSource:
        def __init__(self) -> None:
            self.capture_count = 0

        def capture_single(
            self,
            settings,
            *,
            cancelled=None,
            armed=None,
            prefer_buffered=False,
        ):
            assert prefer_buffered
            self.capture_count += 1
            if armed is not None:
                armed()
            recording, _signal = GeneratedIQSource.fsk(
                symbol_count=64,
                symbol_rate_hz=settings.symbol_rate_hz,
                samples_per_symbol=settings.samples_per_symbol,
                seed=self.capture_count,
            )
            return recording

        def close(self) -> None:
            pass

    source = FakePlutoSource()
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "pluto-continuous"),
        pluto_source=source,
    )
    window.analysis_published.connect(window._toggle_pluto_continuous)
    try:
        window._toggle_pluto_continuous()
        for _index in range(500):
            QtWidgets.QApplication.processEvents()
            capture = window._pluto_capture_thread
            analysis = window._analysis_thread
            if capture is not None:
                capture.wait(10)
            if analysis is not None:
                analysis.wait(10)
            if (
                not window._continuous_run_requested
                and window._pluto_capture_thread is None
                and window._analysis_thread is None
                and window.run_continuous_action.isEnabled()
            ):
                break
        else:
            raise AssertionError("Continuous did not stop after the active analysis")

        assert source.capture_count == 1
        assert window._continuous_sweep_count == 1
        assert window._all_packet_statistics.packet_count == 1
        assert window.result_summary.columnCount() == 3
        assert window.result_summary.horizontalHeaderItem(2).text() == "All Packets"
        assert window._all_packet_summary_values["match_selection"] == "1 packet(s)"
        assert window.run_single_action.isEnabled()
        assert window.run_single_action.text() == "Run Single"
        assert window.run_single_button.text() == "Run Single (Pluto)"
        assert window.open_config_action.isEnabled()
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_pluto_continuous_keeps_running_after_transient_failures(
    tmp_path, monkeypatch
) -> None:
    pg.mkQApp("VSA Pluto Continuous transient failure test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "pluto-continuous-retry")
    )
    dialogs: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "critical",
        lambda _parent, _title, message: dialogs.append(str(message)),
    )
    try:
        window._continuous_run_requested = True
        window._active_analysis_context = {"continuous": True}
        window._pluto_capture_failed("temporary USB timeout")
        assert window._continuous_run_requested
        assert dialogs == []

        generation = window._analysis_generation
        window._analysis_failed(generation, None, "packet not found")
        assert window._continuous_run_requested
        assert "retrying Continuous" in window.statusBar().currentMessage()
    finally:
        window._continuous_run_requested = False
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()


def test_run_single_action_stops_pending_power_trigger_wait(tmp_path) -> None:
    pg.mkQApp("VSA Pluto cancellation test")
    started = threading.Event()

    class WaitingPlutoSource:
        def capture_single(self, settings, *, cancelled=None, armed=None):
            if armed is not None:
                armed()
            started.set()
            while cancelled is None or not cancelled():
                threading.Event().wait(0.002)
            raise CaptureCancelledError("Pluto capture cancelled")

        def close(self) -> None:
            pass

    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "pluto-cancel-single"),
        pluto_source=WaitingPlutoSource(),
    )
    try:
        window.acquisition_trigger_source_combo.setCurrentIndex(
            window.acquisition_trigger_source_combo.findData("power_level")
        )
        window._run_pluto_single()
        assert started.wait(timeout=2.0)
        assert window.run_single_action.text() == "Stop Single"
        assert window.run_single_action.isEnabled()

        window._run_pluto_single()
        for _index in range(200):
            QtWidgets.QApplication.processEvents()
            thread = window._pluto_capture_thread
            if thread is None:
                break
            thread.wait(10)

        assert window._pluto_capture_thread is None
        assert window.run_single_action.text() == "Run Single"
        assert window.run_single_action.isEnabled()
        assert "cancelled" in window.statusBar().currentMessage().lower()
    finally:
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()


def test_analysis_runs_outside_gui_thread_and_latest_request_wins(
    tmp_path, monkeypatch
) -> None:
    pg.mkQApp("VSA background analysis test")
    window = VSAWindow(
        preferences=_isolated_preferences(tmp_path, "background-analysis")
    )
    entered = threading.Event()
    release = threading.Event()
    observed_threads: list[QtCore.QThread] = []
    original_analyze = VSASession.analyze

    def delayed_analyze(session):
        observed_threads.append(QtCore.QThread.currentThread())
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=5.0)
        return original_analyze(session)

    monkeypatch.setattr(VSASession, "analyze", delayed_analyze)
    try:
        recording, signal = GeneratedIQSource.fsk(symbol_count=64, seed=919)
        window.load_recording(recording, signal)
        assert entered.wait(timeout=2.0)

        window.symbol_rate_spin.setValue(900_000.0)
        assert window._request_analysis()
        window.symbol_rate_spin.setValue(800_000.0)
        assert window._request_analysis()
        assert window._pending_analysis is not None

        release.set()
        _wait_for_background_analysis(window)

        assert observed_threads
        assert all(
            thread is not QtWidgets.QApplication.instance().thread()
            for thread in observed_threads
        )
        assert len(observed_threads) == 2
        assert window.session.signal.symbol_rate_hz == pytest.approx(800_000.0)
        assert window.session.result is not None
        assert "Analysis complete" in window.statusBar().currentMessage()
    finally:
        release.set()
        _wait_for_background_analysis(window)
        window._meas_config_dialog.close()
        window.close()
        window.deleteLater()
        QtWidgets.QApplication.processEvents()
