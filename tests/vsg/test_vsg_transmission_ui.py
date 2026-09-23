import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtWidgets
from pluto_vsg.backends import (
    PlutoOutputBackend,
    PlutoPlaybackMode,
    PlutoTransmitSettings,
)
from pluto_vsg.ui.main_window import PlutoVSGWindow, _PlutoOutputDialog


def test_vsg_close_stops_active_pluto_transmission_before_closing() -> None:
    pg.mkQApp("Pluto VSG graceful-close test")
    window = PlutoVSGWindow()

    class _Worker:
        cancel_count = 0

        def cancel(self) -> None:
            self.cancel_count += 1

    worker = _Worker()
    window._tx_worker = worker
    window._tx_thread = object()
    window.show()

    assert window.close() is False
    assert worker.cancel_count == 1
    assert window._close_after_tx is True

    # Repeated user close requests must not send duplicate stop commands.
    assert window.close() is False
    assert worker.cancel_count == 1

    window._tx_worker = None
    window._tx_thread = None
    window.close()


def test_vsg_rf_button_shows_transfer_then_blue_on_state() -> None:
    pg.mkQApp("Pluto VSG RF button states")
    window = PlutoVSGWindow()
    try:
        window._pluto_prepared_signature = window._pluto_configuration_signature()
        window._rf_transfer_pending = True
        window._update_vsg_control_labels()
        assert window.rf_button.text() == "RF\nTransferring..."
        assert window.rf_button.isChecked() is False

        window._tx_thread = object()
        window._pluto_first_tx_completed()
        assert window.rf_button.text() == "RF\nON"
        assert window.rf_button.isChecked() is True

        window._pluto_transmission_finished(True, "complete")
        assert window.rf_button.text() == "RF\nOFF"
        assert window.rf_button.isChecked() is False
    finally:
        window._tx_thread = None
        window.close()


def test_vsg_calibration_button_starts_without_confirmation(monkeypatch) -> None:
    pg.mkQApp("Pluto VSG direct RF calibration")
    window = PlutoVSGWindow()
    started_states: list[tuple[str, bool]] = []

    def start_calibration() -> None:
        started_states.append(
            (window.rf_button.text(), window.rf_button.isChecked())
        )

    monkeypatch.setattr(window, "_start_pluto_preparation", start_calibration)
    try:
        window._pluto_prepared_signature = None
        window.rf_button.click()

        assert started_states == [("Calibration", False)]
        assert window.rf_button.text() == "Calibration"
        assert window.rf_button.isChecked() is False
    finally:
        window.close()


def test_vsg_rf_button_tracks_calibration_lifecycle() -> None:
    pg.mkQApp("Pluto VSG RF calibration lifecycle")
    window = PlutoVSGWindow()
    try:
        assert window.rf_button.text() == "Calibration"
        assert window.rf_button.isChecked() is False

        window._calibration_in_progress = True
        window._update_vsg_control_labels()
        assert window.rf_button.text() == "Calibrating..."
        assert window.rf_button.isChecked() is False

        window._calibration_in_progress = False
        window._pluto_prepared_signature = window._pluto_configuration_signature()
        window._update_vsg_control_labels()
        assert window.rf_button.text() == "RF\nOFF"
        assert window.rf_button.isChecked() is False
    finally:
        window.close()


def test_pluto_output_dialog_uses_dbm_and_preserves_target_across_backoff(
    monkeypatch,
) -> None:
    pg.mkQApp("Pluto VSG dBm output dialog test")
    monkeypatch.setattr(PlutoOutputBackend, "discover_devices", lambda: ())
    parent = QtWidgets.QWidget()
    dialog = _PlutoOutputDialog(
        PlutoTransmitSettings(
            center_frequency_hz=2_440_000_000.0,
            sample_rate_hz=8_000_000.0,
            rf_bandwidth_hz=8_000_000.0,
            hardware_gain_db=-10.0,
            digital_backoff_db=0.0,
            output_power_dbm=-9.4,
        ),
        packet_count=1,
        parent=parent,
    )
    try:
        assert dialog.output_power_spin.suffix() == " dBm"
        assert dialog.output_power_spin.value() == pytest.approx(-9.4)
        assert dialog.applied_gain_label.text().startswith("-10.00 dB")

        dialog.digital_backoff_combo.setCurrentIndex(
            dialog.digital_backoff_combo.findData(-3.0)
        )

        assert dialog.output_power_spin.value() == pytest.approx(-9.4)
        assert dialog.applied_gain_label.text().startswith("-6.74 dB")
        dialog._accept_settings()
        assert dialog.settings.output_power_dbm == pytest.approx(-9.4)
        assert dialog.settings.resolved_hardware_gain_db == pytest.approx(-6.73913)
    finally:
        dialog.close()
        parent.close()


def test_pluto_output_dialog_selects_continuous_playback_without_changing_project_count(
    monkeypatch,
) -> None:
    pg.mkQApp("Pluto VSG continuous output dialog test")
    monkeypatch.setattr(PlutoOutputBackend, "discover_devices", lambda: ())
    parent = QtWidgets.QWidget()
    dialog = _PlutoOutputDialog(
        PlutoTransmitSettings(
            center_frequency_hz=2_440_000_000.0,
            sample_rate_hz=8_000_000.0,
            rf_bandwidth_hz=8_000_000.0,
            output_power_dbm=-9.4,
        ),
        packet_count=10,
        parent=parent,
    )
    try:
        dialog.playback_mode_combo.setCurrentIndex(
            dialog.playback_mode_combo.findData(
                PlutoPlaybackMode.CONTINUOUS.value
            )
        )

        assert dialog.packet_count_label.text().startswith("Ignored")
        assert not dialog.dma_preroll_spin.isEnabled()
        assert not dialog.stop_guard_spin.isEnabled()
        assert dialog.dma_preroll_label.text() == "Finite TX Lead-in (Zero IQ)"
        assert dialog.stop_guard_label.text() == "Finite TX Minimum Hold"
        assert "DMA startup" in dialog.dma_preroll_spin.toolTip()
        assert "after DMA submission" in dialog.stop_guard_spin.toolTip()
        dialog._accept_settings()
        assert dialog.settings.playback_mode is PlutoPlaybackMode.CONTINUOUS
        assert dialog.settings.burst_count == 10
    finally:
        dialog.close()
        parent.close()
