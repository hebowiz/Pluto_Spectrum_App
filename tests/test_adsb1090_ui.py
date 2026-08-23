import os
import json
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
import numpy as np
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_sa.standards.adsb1090.model import ADSB1090Settings
from pluto_sa.standards.adsb1090.ui import (
    ADSB1090Window,
    _ADSBCaptureBatch,
    _ADSBPlutoCaptureThread,
    _ADSBStreamAnalysisThread,
    _ADSBStreamProcessor,
)
from pluto_sa.vsa.pluto_source import CaptureCancelledError, PlutoCaptureSettings
from pluto_sa.vsa.sources import FileIQSource
from pluto_sa.vsa.ui.measurement_chrome import FixedInteractionViewBox


@pytest.fixture(autouse=True)
def _isolate_default_qsettings(tmp_path, monkeypatch):
    original = QtCore.QSettings

    class _IsolatedQSettings(original):
        def __init__(self, *args, **kwargs):
            if args == ("PlutoSA", "PlutoVSA-ADSB1090") and not kwargs:
                super().__init__(
                    str(tmp_path / "adsb1090-default.ini"),
                    original.Format.IniFormat,
                )
            else:
                super().__init__(*args, **kwargs)

    monkeypatch.setattr(QtCore, "QSettings", _IsolatedQSettings)
    yield


def test_adsb_workspace_starts_without_iq() -> None:
    pg.mkQApp("ADS-B workspace test")
    window = ADSB1090Window()
    try:
        assert window.recording is None
        assert window.packet_table.rowCount() == 0
        assert "ADS-B 1090ES" in window.windowTitle()
        assert window.preamble_snr_spin.value() == pytest.approx(5.0)
    finally:
        window.close()


def test_adsb_user_settings_are_restored_from_dedicated_preferences(tmp_path) -> None:
    pg.mkQApp("ADS-B settings persistence test")
    path = tmp_path / "adsb1090.ini"
    first_preferences = QtCore.QSettings(
        str(path), QtCore.QSettings.Format.IniFormat
    )
    first = ADSB1090Window(preferences=first_preferences)
    try:
        first.sample_rate_combo.setCurrentIndex(
            first.sample_rate_combo.findData(16)
        )
        first.capture_length_spin.setValue(375.0)
        first.internal_gain_spin.setValue(42.5)
        first.preamble_snr_spin.setValue(4.5)
        first.prepare_for_shutdown()
    finally:
        first.close()

    restored_preferences = QtCore.QSettings(
        str(path), QtCore.QSettings.Format.IniFormat
    )
    second = ADSB1090Window(preferences=restored_preferences)
    try:
        assert second.sample_rate_combo.currentData() == 16
        assert second.capture_length_spin.value() == pytest.approx(375.0)
        assert second.internal_gain_spin.value() == pytest.approx(42.5)
        assert second.preamble_snr_spin.value() == pytest.approx(4.5)
    finally:
        second.close()


def test_adsb_user_preamble_snr_threshold_is_forwarded_to_analysis() -> None:
    pg.mkQApp("ADS-B SNR setting test")
    path = Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    recording = FileIQSource.load(path)
    window = ADSB1090Window()
    observed: list[float] = []
    original_analyzer = window._analyzer

    class _AnalyzerSpy:
        def analyze(self, target, settings):
            observed.append(settings.minimum_preamble_snr_db)
            return original_analyzer.analyze(target, settings)

    try:
        window._analyzer = _AnalyzerSpy()
        window.preamble_snr_spin.setValue(6.5)

        window.analyze_recording(recording)
        window._continuous_scan = True
        window._reset_stream_state(recording.sample_rate_hz)
        window._process_stream_block(recording)

        assert observed == [6.5, 6.5]
    finally:
        window.close()


def test_adsb_workspace_displays_saved_multi_packet_fixture() -> None:
    pg.mkQApp("ADS-B fixture workspace test")
    path = Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    window = ADSB1090Window(FileIQSource.load(path))
    try:
        assert window.packet_table.rowCount() == 4
        assert window.packet_table.item(0, 5).text() == "4840D6"
        assert window.packet_table.item(0, 7).text() == "OK"
        assert float(window.packet_table.item(0, 8).text()) > -20.0
        assert window.packet_table.item(0, 10).text() == "KLM1023"
        assert window.packet_table.item(0, 0).text() == "1"
        assert window.packet_table.item(0, 1).text() == "0.000180"
        assert window.packet_table.item(0, 2).text()
        assert window.packet_table.item(0, 3).text() == "8D4840D6202CC371C32CE0576098"
        assert window.aircraft_table.rowCount() == 3
        assert window.power_dock.windowTitle() == "IQ Power"
        assert window.ppm_dock.windowTitle() == "PPM Demodulation"
        assert window.packet_dock.windowTitle() == "Packet List"
        assert window.summary_dock.windowTitle() == "Message Summary"
        assert window.aircraft_dock.windowTitle() == "Detected Aircraft"
        assert window.aircraft_summary_dock.windowTitle() == "Aircraft Summary"
    finally:
        window.close()


def test_adsb_aircraft_summary_aggregates_messages_by_confirmed_icao() -> None:
    pg.mkQApp("ADS-B aircraft aggregation test")
    path = Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    window = ADSB1090Window(FileIQSource.load(path))
    try:
        row = window._aircraft_row_by_icao["40621D"]
        window.aircraft_table.selectRow(row)
        values = {
            window.aircraft_summary_table.item(index, 0).text():
            window.aircraft_summary_table.item(index, 1).text()
            for index in range(window.aircraft_summary_table.rowCount())
        }

        assert window.aircraft_table.item(row, 6).text() == "2"
        assert window.aircraft_table.item(row, 7).text() == "38000 / 11582"
        assert values["ICAO Address"] == "40621D"
        assert values["Messages"] == "2"
        assert values["Parity Verified"] == "2"
        assert values["Latest Altitude"] == "38000 ft / 11582 m"
        assert values["Air/Ground"] == "airborne"
        assert values["Latitude"] == "52.265780 degree"
        assert values["Longitude"] == "3.938913 degree"
        assert values["Type Codes"] == "11"
    finally:
        window.close()


def test_adsb_iq_power_display_has_a_finite_dbm_floor() -> None:
    pg.mkQApp("ADS-B IQ power floor test")
    path = Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    recording = FileIQSource.load(path)
    iq = recording.iq.copy()
    iq[-100:] = 0.0
    window = ADSB1090Window(replace(recording, iq=iq))
    try:
        _, displayed_power = window.power_plot.listDataItems()[0].getData()
        assert np.min(displayed_power) == pytest.approx(-140.0)
    finally:
        window.close()


def test_adsb_packet_list_exports_versioned_json_lines(tmp_path, monkeypatch) -> None:
    pg.mkQApp("ADS-B packet export test")
    path = Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    output = tmp_path / "packets.jsonl"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(output), "JSON Lines (*.jsonl)"),
    )
    window = ADSB1090Window(FileIQSource.load(path))
    try:
        window._export_packet_list()

        records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        assert len(records) == 4
        assert records[0]["schema"] == "pluto-vsa.adsb1090.packet"
        assert records[0]["version"] == 1
        assert records[0]["raw_message"] == "8D4840D6202CC371C32CE0576098"
        assert records[0]["icao_address"] == "4840D6"
        assert records[0]["parity"]["verified"] is True
        assert records[0]["decoded_fields"]["callsign"] == "KLM1023"
    finally:
        window.close()


def test_adsb_workspace_appends_history_with_elapsed_and_os_time() -> None:
    pg.mkQApp("ADS-B continuous history test")
    path = Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    recording = FileIQSource.load(path)
    window = ADSB1090Window()
    try:
        window.analyze_recording(
            recording,
            append=True,
            capture_started_at=datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc),
            elapsed_base_s=1.5,
        )
        window.analyze_recording(
            recording,
            append=True,
            capture_started_at=datetime(2026, 8, 22, 12, 0, 1, tzinfo=timezone.utc),
            elapsed_base_s=2.5,
        )

        assert window.packet_table.rowCount() == 8
        assert window.packet_table.item(0, 1).text() == "1.500180"
        assert window.packet_table.item(4, 1).text() == "2.500180"
        assert window.packet_table.item(0, 2).text() == "2026-08-22 12:00:00.000180"
        assert window.packet_table.currentRow() == 7
        assert window.power_plot.viewRange()[0][0] >= 2500.0
    finally:
        window.close()


def test_adsb_continuous_history_preserves_user_selected_packet() -> None:
    pg.mkQApp("ADS-B continuous selection test")
    recording = FileIQSource.load(
        Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    )
    window = ADSB1090Window()
    try:
        window.analyze_recording(recording, append=True)
        window.packet_table.selectRow(0)

        window.analyze_recording(recording, append=True, elapsed_base_s=1.0)

        assert window.packet_table.rowCount() == 8
        assert window.packet_table.currentRow() == 0
        assert window.summary_table.item(0, 1).text() == "0.000180 s"
    finally:
        window.close()


def test_adsb_plots_share_vsa_interaction_and_ppm_soft_decisions() -> None:
    pg.mkQApp("ADS-B plot interaction test")
    path = Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    window = ADSB1090Window(FileIQSource.load(path))
    try:
        assert isinstance(window.power_plot.getViewBox(), FixedInteractionViewBox)
        assert window.power_dock.font().bold() is True
        menu_text = [
            action.text() for action in window.ppm_plot.getViewBox().getMenu(None).actions()
        ]
        assert "Reset" in menu_text
        assert "View All" in menu_text
        assert "Mouse Mode" not in menu_text

        _, soft_decision_db = window.ppm_plot.listDataItems()[0].getData()
        bits = window._packet_history[0].message.bits
        assert soft_decision_db.size == bits.size
        assert np.all(soft_decision_db[bits == 1] > 0.0)
        assert np.all(soft_decision_db[bits == 0] < 0.0)
    finally:
        window.close()


def test_adsb_continuous_capture_preserves_buffer_after_first_block() -> None:
    path = Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    recording = FileIQSource.load(path)

    class _Source:
        def __init__(self) -> None:
            self.fresh_values: list[bool] = []

        def capture_single(self, _settings, *, cancelled, fresh=True):
            self.fresh_values.append(bool(fresh))
            if len(self.fresh_values) > 2:
                raise CaptureCancelledError("test complete")
            return recording

    source = _Source()
    thread = _ADSBPlutoCaptureThread(
        source,
        PlutoCaptureSettings(),
    )
    captures: list[object] = []
    thread.capture_ready.connect(captures.append)

    thread.run()

    assert len(captures) == 2
    assert source.fresh_values == [True, False, False]


def test_adsb_stream_dsp_runs_outside_gui_thread() -> None:
    pg.mkQApp("ADS-B stream DSP thread test")
    recording = FileIQSource.load(
        Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    )
    processor = _ADSBStreamProcessor(
        PlutoCaptureSettings(
            center_frequency_hz=recording.center_frequency_hz,
            symbol_rate_hz=1_000_000.0,
            samples_per_symbol=8,
            capture_length_s=0.001,
            rf_bandwidth_hz=4_000_000.0,
        ),
        ADSB1090Settings(minimum_preamble_snr_db=5.0),
        continuous=True,
        scan_started_wall_time=datetime.now().astimezone(),
    )
    original_analyzer = processor.analyzer
    worker_thread_ids: list[int] = []

    class _AnalyzerSpy:
        def analyze(self, target, settings):
            worker_thread_ids.append(threading.get_ident())
            return original_analyzer.analyze(target, settings)

    processor.analyzer = _AnalyzerSpy()
    thread = _ADSBStreamAnalysisThread(processor)
    views: list[object] = []
    loop = QtCore.QEventLoop()
    thread.view_ready.connect(views.append)
    thread.view_ready.connect(loop.quit)
    thread.analysis_failed.connect(loop.quit)
    thread.start()
    try:
        thread.enqueue(_ADSBCaptureBatch(recording))
        QtCore.QTimer.singleShot(3000, loop.quit)
        loop.exec()
    finally:
        thread.stop()
        thread.wait(3000)

    assert views
    assert worker_thread_ids
    assert worker_thread_ids[0] != threading.get_ident()


def test_adsb_window_continuous_scan_uses_background_dsp_pipeline() -> None:
    pg.mkQApp("ADS-B background pipeline integration test")
    recording = FileIQSource.load(
        Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    )

    class _Source:
        def __init__(self) -> None:
            self.calls = 0

        def capture_single(self, _settings, *, cancelled, fresh=True):
            self.calls += 1
            if self.calls == 1:
                return recording
            while not cancelled():
                threading.Event().wait(0.005)
            raise CaptureCancelledError("test stopped")

    source = _Source()
    window = ADSB1090Window(pluto_source=source, owns_pluto_source=False)
    wait_for_packets = QtCore.QEventLoop()
    packet_poll = QtCore.QTimer()
    packet_poll.setInterval(10)
    packet_poll.timeout.connect(
        lambda: wait_for_packets.quit()
        if window.packet_table.rowCount() >= 4
        else None
    )
    packet_poll.start()
    QtCore.QTimer.singleShot(3000, wait_for_packets.quit)
    try:
        window._run_pluto_continuous()
        wait_for_packets.exec()
        assert window.packet_table.rowCount() == 4
        assert window._analysis_stream_thread is not None
        window._run_pluto_continuous()

        wait_for_stop = QtCore.QEventLoop()
        stop_poll = QtCore.QTimer()
        stop_poll.setInterval(10)
        stop_poll.timeout.connect(
            lambda: wait_for_stop.quit()
            if window._capture_thread is None
            and window._analysis_stream_thread is None
            else None
        )
        stop_poll.start()
        QtCore.QTimer.singleShot(3000, wait_for_stop.quit)
        wait_for_stop.exec()
        assert window._capture_thread is None
        assert window._analysis_stream_thread is None
    finally:
        packet_poll.stop()
        window.close()


def test_adsb_stream_detects_packets_across_internal_block_boundaries() -> None:
    pg.mkQApp("ADS-B stream overlap test")
    path = Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    recording = FileIQSource.load(path)
    window = ADSB1090Window()
    try:
        window.capture_length_spin.setValue(1.0)
        window._continuous_scan = True
        window._scan_started_wall_time = datetime(
            2026, 8, 22, 12, 0, tzinfo=timezone.utc
        )
        window._reset_stream_state(recording.sample_rate_hz)
        # Deliberately use a block length that cuts through Mode S messages.
        for start in range(0, recording.sample_count, 733):
            window._process_stream_block(
                replace(recording, iq=recording.iq[start : start + 733])
            )

        assert window.packet_table.rowCount() == 4
        assert [window.packet_table.item(row, 5).text() for row in range(4)] == [
            "4840D6",
            "40621D",
            "40621D",
            "485020",
        ]
        x_range = window.power_plot.viewRange()[0]
        assert x_range[1] - x_range[0] == pytest.approx(1.0, abs=0.01)
    finally:
        window.close()


def test_adsb_stream_does_not_repaint_without_a_detected_packet() -> None:
    pg.mkQApp("ADS-B quiet stream test")
    recording = FileIQSource.load(
        Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    )
    noise = replace(
        recording,
        iq=np.zeros(4000, dtype=np.complex64),
    )
    window = ADSB1090Window()
    try:
        window._continuous_scan = True
        window._reset_stream_state(noise.sample_rate_hz)
        before = len(window.power_plot.listDataItems())

        window._process_stream_block(noise)

        assert window.packet_table.rowCount() == 0
        assert len(window.power_plot.listDataItems()) == before
    finally:
        window.close()


def test_adsb_single_waits_for_packet_then_keeps_configured_post_time() -> None:
    pg.mkQApp("ADS-B single post-trigger test")
    recording = FileIQSource.load(
        Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    )
    window = ADSB1090Window()
    try:
        window.capture_length_spin.setValue(1.0)
        window._continuous_scan = False
        window._scan_started_wall_time = datetime(
            2026, 8, 22, 12, 0, tzinfo=timezone.utc
        )
        window._reset_stream_state(recording.sample_rate_hz)

        window._process_stream_block(recording)

        assert window._single_complete is True
        assert window._single_trigger_sample is not None
        assert window.packet_table.rowCount() == 3
        expected_stop = (
            window._single_trigger_sample
            + int(round(1e-3 * recording.sample_rate_hz))
        )
        assert window.recording.end_sample_index == expected_stop
    finally:
        window.close()


def test_adsb_shutdown_disconnects_packet_selection_callback() -> None:
    pg.mkQApp("ADS-B shutdown lifecycle test")
    path = Path(__file__).parent / "fixtures" / "adsb1090_multi_8msps.npz"
    window = ADSB1090Window(FileIQSource.load(path))
    calls: list[object] = []
    window._show_message_plot = calls.append
    try:
        window.prepare_for_shutdown()
        window.packet_table.selectRow(1)
        assert calls == []
    finally:
        window.close()
