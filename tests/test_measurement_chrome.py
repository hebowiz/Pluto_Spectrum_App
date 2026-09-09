import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pyqtgraph as pg

from pluto_sa.vsa.ui.measurement_chrome import PersistentPlotRanges


def test_persistent_plot_range_tracks_packet_relative_x_and_reset() -> None:
    pg.mkQApp("persistent plot ranges")
    plot = pg.PlotWidget()
    ranges = PersistentPlotRanges((("modulation", plot),))
    try:
        plot.setRange(xRange=[100.0, 110.0], yRange=[-5.0, 5.0], padding=0.0)
        ranges.finish_update(
            contexts={"modulation": "fsk_time"},
            relative_x_origins={"modulation": 100.0},
        )

        plot.setRange(xRange=[102.0, 104.0], yRange=[-1.0, 1.0], padding=0.0)
        ranges.prepare_for_update()
        assert ranges.has_manual_range("modulation")

        plot.setRange(xRange=[200.0, 210.0], yRange=[-6.0, 6.0], padding=0.0)
        ranges.finish_update(
            contexts={"modulation": "fsk_time"},
            relative_x_origins={"modulation": 200.0},
        )
        np.testing.assert_allclose(plot.viewRange()[0], [202.0, 204.0])
        np.testing.assert_allclose(plot.viewRange()[1], [-1.0, 1.0])

        assert ranges.reset("modulation")
        np.testing.assert_allclose(plot.viewRange()[0], [200.0, 210.0])
        np.testing.assert_allclose(plot.viewRange()[1], [-6.0, 6.0])
        assert not ranges.has_manual_range("modulation")
    finally:
        plot.close()
        plot.deleteLater()


def test_persistent_plot_range_separates_time_and_iq_contexts() -> None:
    pg.mkQApp("persistent plot range contexts")
    plot = pg.PlotWidget()
    ranges = PersistentPlotRanges((("modulation", plot),))
    try:
        plot.setRange(xRange=[10.0, 20.0], yRange=[-5.0, 5.0], padding=0.0)
        ranges.finish_update(
            contexts={"modulation": "fsk_time"},
            relative_x_origins={"modulation": 10.0},
        )
        plot.setRange(xRange=[12.0, 14.0], yRange=[-2.0, 2.0], padding=0.0)
        ranges.prepare_for_update()

        plot.setRange(xRange=[-1.25, 1.25], yRange=[-1.25, 1.25], padding=0.0)
        ranges.finish_update(contexts={"modulation": "iq_plane"})
        np.testing.assert_allclose(plot.viewRange()[0], [-1.25, 1.25])

        ranges.prepare_for_update()
        plot.setRange(xRange=[30.0, 40.0], yRange=[-6.0, 6.0], padding=0.0)
        ranges.finish_update(
            contexts={"modulation": "fsk_time"},
            relative_x_origins={"modulation": 30.0},
        )
        np.testing.assert_allclose(plot.viewRange()[0], [32.0, 34.0])
        np.testing.assert_allclose(plot.viewRange()[1], [-2.0, 2.0])
    finally:
        plot.close()
        plot.deleteLater()
