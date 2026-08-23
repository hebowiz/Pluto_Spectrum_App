import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
import numpy as np
import pytest

from pluto_sa.standards.adsb1090.ui import ADSB1090Window, _ADSBPlutoCaptureThread
from pluto_sa.vsa.pluto_source import CaptureCancelledError, PlutoCaptureSettings
from pluto_sa.vsa.sources import FileIQSource
from pluto_sa.vsa.ui.measurement_chrome import FixedInteractionViewBox


def test_adsb_workspace_starts_without_iq() -> None:
    pg.mkQApp("ADS-B workspace test")
    window = ADSB1090Window()
    try:
        assert window.recording is None
        assert window.packet_table.rowCount() == 0
        assert "ADS-B 1090ES" in window.windowTitle()
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
