import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import pyqtgraph as pg
import numpy as np
from pluto_vsa.ui.measurement_chrome import FixedInteractionViewBox
from pluto_vsg.engine import BluetoothBRWaveformEngine
from pluto_vsg.model import BluetoothPacketKind
from pluto_vsg.profiles import (
    bluetooth_br_edr_project,
    bluetooth_br_fields,
    bluetooth_hdt_project,
)
from pluto_vsg.ui.main_window import (
    PlutoVSGWindow,
    _instantaneous_frequency_khz,
    _preview_active_x_range_us,
)


def test_vsg_preview_keeps_one_period_for_ten_thousand_packet_project() -> None:
    pg.mkQApp("Pluto VSG compact repeat preview")
    project = replace(bluetooth_hdt_project(), repeat_count=10_000)
    window = PlutoVSGWindow(project)
    try:
        assert window.result is not None
        assert window.result.iq.size == window.result.metadata["period_sample_count"]
        settings = window._current_pluto_settings()
        assert settings.burst_count == 10_000
        assert settings.single_period_template is True
    finally:
        window.close()


def test_vsg_preview_draws_only_first_packet_when_schedule_repeats() -> None:
    pg.mkQApp("Pluto VSG single packet preview test")
    project = replace(bluetooth_br_edr_project(), repeat_count=4)
    result = BluetoothBRWaveformEngine().generate(project)
    window = PlutoVSGWindow(project)
    try:
        window._update_previews(result)

        expected_samples = result.iq.size // project.repeat_count
        iq_x, _ = window.iq_waveform_plot.listDataItems()[0].getData()
        power_x, _ = window.power_plot.listDataItems()[0].getData()
        assert iq_x.size == expected_samples
        assert power_x.size == expected_samples
        assert result.iq.size == expected_samples * 4

        lines = [
            item
            for item in window.power_plot.getPlotItem().items
            if isinstance(item, pg.InfiniteLine) and item.label is not None
        ]
        labels = [line.label.format for line in lines]
        assert labels.count("Packet End") == 1
        assert not any(label.endswith(" [1]") for label in labels)
    finally:
        window.close()


def test_vsg_preview_initial_time_range_tracks_active_window_not_post_idle() -> None:
    pg.mkQApp("Pluto VSG active-window preview range test")
    base = bluetooth_br_edr_project()
    project = replace(base, period_symbols=2000.0)
    result = BluetoothBRWaveformEngine().generate(project)
    window = PlutoVSGWindow(project)
    try:
        window._update_previews(result)
        expected = _preview_active_x_range_us(
            result, result.metadata["period_sample_count"]
        )
        for plot in (
            window.iq_waveform_plot,
            window.power_plot,
            window.frequency_plot,
        ):
            np.testing.assert_allclose(plot.viewRange()[0], expected)

        plotted_time, _ = window.power_plot.listDataItems()[0].getData()
        assert plotted_time[-1] > expected[1]
        active_start, active_stop = result.metadata["active_ranges_samples"][0]
        active_width_us = (active_stop - active_start) / result.sample_rate_hz * 1e6
        assert expected[1] - expected[0] <= active_width_us * 1.10 + 1e-9
    finally:
        window.close()


def test_edr_constellation_preview_uses_mapped_psk_symbols_only() -> None:
    pg.mkQApp("Pluto VSG EDR constellation preview test")
    base = bluetooth_br_edr_project()
    settings = replace(base.bluetooth_br, packet_kind=BluetoothPacketKind.DH1_2)
    project = replace(
        base,
        bluetooth_br=settings,
        fields=bluetooth_br_fields(settings),
    )
    result = BluetoothBRWaveformEngine().generate(project)
    assert len(result.constellation_traces) == 1
    trace = result.constellation_traces[0]
    assert trace.modulation == "pi/4-DQPSK"
    np.testing.assert_allclose(np.abs(trace.symbols), 1.0, atol=1e-12)
    phase_bins = np.mod(
        np.rint(np.angle(trace.symbols) / (np.pi / 4.0)).astype(int), 8
    )
    assert np.unique(phase_bins).size == 8

    window = PlutoVSGWindow(project)
    try:
        window._update_previews(result)
        plotted = window.constellation_plot.listDataItems()
        assert len(plotted) == 1
        x, y = plotted[0].getData()
        np.testing.assert_allclose(x + 1j * y, trace.symbols)
        title = window.constellation_plot.getPlotItem().titleLabel.text
        assert "before pulse shaping" in title
    finally:
        window.close()


def test_fsk_only_preview_does_not_plot_raw_iq_as_a_constellation() -> None:
    pg.mkQApp("Pluto VSG FSK constellation exclusion test")
    project = bluetooth_br_edr_project()
    result = BluetoothBRWaveformEngine().generate(project)
    assert result.constellation_traces == ()
    window = PlutoVSGWindow(project)
    try:
        window._update_previews(result)
        assert window.constellation_plot.listDataItems() == []
        assert "No I/Q symbol constellation" in (
            window.constellation_plot.getPlotItem().titleLabel.text
        )
    finally:
        window.close()


def test_vsg_field_labels_are_present_and_keep_a_fixed_side() -> None:
    pg.mkQApp("Pluto VSG field label anchor test")
    window = PlutoVSGWindow()
    try:
        for plot in (
            window.iq_waveform_plot,
            window.power_plot,
            window.frequency_plot,
        ):
            lines = [
                item
                for item in plot.getPlotItem().items
                if isinstance(item, pg.InfiniteLine)
            ]
            assert lines
            assert all(line.label is not None for line in lines)
            assert all(
                line.label.anchors == [(0.0, 0.5), (0.0, 0.5)]
                for line in lines
            )
            assert any(line.label.format == "Packet End" for line in lines)
    finally:
        window.close()


def test_vsg_plots_use_fixed_vsa_interaction_and_reset() -> None:
    pg.mkQApp("Pluto VSG VSA plot interaction test")
    window = PlutoVSGWindow()
    try:
        for name, plot in window._plot_widgets():
            view_box = plot.getViewBox()
            assert isinstance(view_box, FixedInteractionViewBox)
            assert view_box.state["mouseMode"] == pg.ViewBox.RectMode
            menu = view_box.getMenu(None)
            assert menu is not None
            assert "Mouse Mode" not in [action.text() for action in menu.actions()]
            assert name in window._plot_context_actions

        power_initial = window._plot_initial_ranges["power"]
        window.power_plot.setRange(
            xRange=[0.0, 1.0], yRange=[-10.0, 10.0], padding=0.0
        )
        window._plot_context_actions["power"]["reset"].trigger()
        x_range, y_range = window.power_plot.viewRange()
        np.testing.assert_allclose(x_range, power_initial[0])
        np.testing.assert_allclose(y_range, power_initial[1])
    finally:
        window.close()


def test_vsg_frequency_preview_does_not_connect_burst_to_zero_hz() -> None:
    phase_step = 2.0 * np.pi * 100_000.0 / 8_000_000.0
    active = np.exp(1j * phase_step * np.arange(8))
    iq = np.concatenate((np.zeros(2), active, np.zeros(2)))

    frequency = _instantaneous_frequency_khz(iq, 8_000_000.0)

    assert np.isnan(frequency[0:2]).all()
    np.testing.assert_allclose(frequency[2:9], 100.0)
    assert np.isnan(frequency[9:]).all()
