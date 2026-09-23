import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from dataclasses import replace
from types import SimpleNamespace
import time
import numpy as np
import pyqtgraph as pg
import pytest
from pyqtgraph.Qt import QtCore, QtWidgets
from pluto_vsa.model import IQRecording
from pluto_vsa.protocol_modes.wifi.ui import WiFiAnalyzerWindow
from pluto_vsa.ui.application_window import PlutoAnalysisWindow
from pluto_vsa.ui.config_transaction import create_draft_editor
from pluto_vsa.ui.measurement_chrome import DedicatedSummaryTable, SYMBOL_PLOT_FLAT_SIZE, SymbolDensitySpread, view_all_traces
from pluto_vsa.persistence import save_mode_meas_config
from pluto_vsg.engine.wifi_legacy_ofdm import WiFiLegacyOFDMWaveformEngine
from pluto_vsg.profiles.wifi import wifi_project
from pluto_vsg.model import WiFiSettings


def recording():
    wave = WiFiLegacyOFDMWaveformEngine().generate(wifi_project(WiFiSettings(packet_period_us=400,legacy_rate_mbps=24)))
    return IQRecording(np.r_[wave.iq,wave.iq],wave.sample_rate_hz,2437e6)


class Source:
    def __init__(self):
        self.closed = 0
        self.captures = 0
        self.stopped = 0
    def capture_single(self,settings,**kwargs):
        self.captures += 1
        kwargs['armed']()
        return recording()
    def stop_stream(self):
        self.stopped += 1
    def close(self):
        self.closed += 1


def prefs(tmp_path):
    return QtCore.QSettings(str(tmp_path/'wifi.ini'),QtCore.QSettings.Format.IniFormat)


def wait(app,predicate):
    deadline = time.monotonic()+10
    while not predicate() and time.monotonic()<deadline:
        app.processEvents()
        time.sleep(.005)
    assert predicate()


def test_workspace_has_six_ordered_panes_and_selected_packet_views(tmp_path):
    app = pg.mkQApp()
    w = WiFiAnalyzerWindow(pluto_source=Source(),preferences=prefs(tmp_path))
    try:
        w.show()
        app.processEvents()
        docks = w.findChildren(QtWidgets.QDockWidget)
        assert len(docks)==6
        assert [d.windowTitle() for d in docks] == ['IQ Power','Spectrum','Result Summary','Modulation','Symbol Plot','Packet Analysis']
        assert w.power_dock.x() < w.spectrum_dock.x() < w.summary_dock.x()
        assert w.modulation_dock.y() > w.power_dock.y()
        assert w.modulation_tabs.count()==7
        assert w.symbol_tabs.count()==2
        for i in range(2):
            assert w.modulation_tabs.widget(i) is w.resource_plots[i]
            assert w.symbol_tabs.widget(i) is w.constellation_plots[i]
        assert w.modulation_tabs.widget(2) is w.evm_carrier_plot
        assert w.modulation_tabs.widget(5) is w.channel_phase_plot
        assert w.modulation_tabs.widget(6) is w.flatness_plot
        assert w.spectrum_tabs.widget(1) is w.mask_plot
        assert isinstance(w.summary_table,DedicatedSummaryTable)
        assert w._meas_config_dialog.button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok) is not None
        assert not w.summary_table.font().bold()
        w.analyze_recording(recording())
        assert w.packet_table.rowCount()==2
        assert w.modulation_tabs.tabText(1)=='DATA - 16QAM'
        assert w.symbol_tabs.tabText(1)=='DATA - 16QAM'
        for resource,region in zip(w.resource_plots,(w._results[0].signal,w._results[0].data)):
            assert resource.getAxis('bottom').labelText=='Subcarrier Index'
            assert resource.getAxis('left').labelText=='OFDM Symbol Index'
            grid = next(item for item in resource.items() if isinstance(item,pg.ImageItem))
            assert grid.image.shape==(len(region.measured),53)
            carriers = np.r_[np.arange(-26,-21),np.arange(-20,-7),np.arange(-6,0),
                             np.arange(1,7),np.arange(8,21),np.arange(22,27)]
            np.testing.assert_allclose(grid.image[:,carriers+26],100*abs(region.error))
        assert w.evm_symbol_plot.getAxis('bottom').labelText=='OFDM Symbol Index'
        w.packet_table.selectRow(1)
        assert w._selected_result_index==1
        assert w._results[1].start_sample > w._results[0].start_sample
        assert w.decode_tree.topLevelItem(1).text(2)=='PSDU logical'
        assert w.payload_text.toPlainText()
        plot = w.constellation_plots[1]
        traces = plot.listDataItems()
        assert len(traces)==2  # Shared measured-point trace and unit circle.
        points = next(item for item in traces if item.opts['symbol']=='o')
        assert points.opts['symbolSize']==SYMBOL_PLOT_FLAT_SIZE
        assert points.opts['symbolBrush'].color()==pg.mkColor('y')
        measured = w._results[1].data.measured.ravel()
        np.testing.assert_allclose(points.xData+1j*points.yData,measured)
        assert not points.opts['autoDownsample']
        assert not points.opts['clipToView']
        plot.setRange(xRange=(-.3,.4),yRange=(-.4,.3),padding=0)
        displayed_x,displayed_y = points.getData()
        np.testing.assert_allclose(displayed_x+1j*displayed_y,measured)
        assert all(spot.size()==SYMBOL_PLOT_FLAT_SIZE for spot in points.scatter.points())
        before = plot.viewRange()
        w._render_selected()
        np.testing.assert_allclose(plot.viewRange(),before)
        w.symbol_trace_combo.setCurrentText('Density')
        assert w.density_check.isChecked()
        w._render_selected()
        assert any(isinstance(item,pg.ImageItem) for item in plot.items())
        np.testing.assert_allclose(plot.viewRange(),before)
        w.reset()
        assert w._recording is None
        assert w.packet_table.rowCount()==0
    finally:
        w.close()


def test_single_continuous_stop_and_shutdown_use_shared_source(tmp_path):
    app = pg.mkQApp()
    source = Source()
    w = WiFiAnalyzerWindow(pluto_source=source,preferences=prefs(tmp_path))
    try:
        w._toggle_capture()
        assert w.shutdown_busy_reason()
        wait(app,lambda: w.shutdown_busy_reason() is None)
        assert w._result.counts['fcs_valid']==2
        assert source.captures==1
        w._toggle_continuous_capture()
        wait(app,lambda: source.captures>=3)
        w.request_shutdown()
        wait(app,lambda: w.shutdown_busy_reason() is None)
        assert source.closed==0
    finally:
        w.close()


def test_draft_settings_isolated_and_startup_restore(tmp_path):
    pg.mkQApp()
    settings = prefs(tmp_path)
    w = WiFiAnalyzerWindow(pluto_source=Source(),preferences=settings)
    try:
        before = w._meas_config_values()
        draft,dialog = create_draft_editor(w)
        draft.channel_combo.setCurrentIndex(10)
        draft.frequency_reference_check.setChecked(True)
        draft.receiver_response_check.setChecked(True)
        draft.random_payload_check.setChecked(True)
        draft.non_vht_dut_check.setChecked(True)
        dialog.reject()
        assert w._meas_config_values()==before
        dialog.deleteLater()
        draft.deleteLater()
        w.channel_combo.setCurrentIndex(7)
        w.duration_spin.setValue(50)
        w.analyze_recording(recording())
        def status(key):
            for row in range(w.summary_table.rowCount()):
                if w.summary_table.item(row,0).data(QtCore.Qt.ItemDataRole.UserRole)==key:
                    return w.summary_table.item(row,3).text()
        assert status('carrier_frequency_error')=='Not Measured'
        w._apply_meas_config_values(dict(frequency_reference_check=True,receiver_response_check=True,
                                        random_payload_check=True,non_vht_dut_check=True))
        assert status('carrier_frequency_error')=='PASS'
        assert status('spectral_flatness')=='PASS'
        assert status('relative_constellation_error')=='Insufficient Data'
        assert status('transmit_spectrum_mask')=='Insufficient Data'
        w._apply_meas_config_values({'density_check':True})  # Legacy file compatibility.
        assert w.symbol_trace_combo.currentText()=='Density'
        w.density_spread_combo.setCurrentIndex(w.density_spread_combo.findData(SymbolDensitySpread.MEDIUM))
        w._save_config()
        other = WiFiAnalyzerWindow(pluto_source=Source(),preferences=settings)
        assert other._meas_config_values()==w._meas_config_values()
        assert other.symbol_trace_combo.currentText()=='Density'
        assert other.density_spread_combo.currentData()==SymbolDensitySpread.MEDIUM
        assert other.symbol_tabs.currentIndex()==0
        other.close()
        with pytest.raises(ValueError):
            w._apply_meas_config_values({'sample_rate':10e6})
    finally:
        w.close()


def test_shell_mode_config_busy_guard_and_shared_connection(tmp_path,monkeypatch):
    app = pg.mkQApp()
    source = Source()
    monkeypatch.setattr(QtWidgets.QMessageBox,'information',lambda *a: None)
    shell = PlutoAnalysisWindow(pluto_source=source,preferences=prefs(tmp_path))
    try:
        shell.show()
        shell.set_analysis_mode('wifi')
        app.processEvents()
        assert shell._active_mode()=='wifi'
        assert shell.wifi_workspace._pluto_source is source
        assert shell.control_panel.buttons['Analyzer Mode'].text().endswith('Wi-Fi')
        shell.wifi_workspace._continuous_run_requested = True
        shell.set_analysis_mode('generic')
        assert shell._active_mode()=='wifi'
        shell.wifi_workspace._continuous_run_requested = False
        path = tmp_path/'wifi.json'
        values = shell._collect_meas_config('wifi')
        save_mode_meas_config(path,analysis_mode='wifi',settings=values)
        shell.set_analysis_mode('generic')
        assert shell.recall_meas_config_path(path)
        assert shell._active_mode()=='wifi'
        assert shell._collect_meas_config('wifi')==values
    finally:
        shell.close()
        app.processEvents()
    assert source.closed==1


def test_capture_frontend_bandwidth_constraints(tmp_path):
    pg.mkQApp()
    w = WiFiAnalyzerWindow(pluto_source=Source(),preferences=prefs(tmp_path))
    try:
        assert w._capture_settings().requested_sample_rate_hz==40e6
        w.sample_rate_combo.setCurrentIndex(0)
        with pytest.raises(ValueError,match='usable bandwidth'):
            w._capture_settings()
        w.sample_rate_combo.setCurrentIndex(1)
        w.analysis_check.setChecked(True)
        w.lo_offset_spin.setValue(3)
        with pytest.raises(ValueError,match='LO Offset'):
            w._capture_settings()
    finally:
        w.close()


def test_power_plot_preserves_low_duty_packet_and_resamples_on_zoom(tmp_path):
    pg.mkQApp()
    w = WiFiAnalyzerWindow(pluto_source=Source(),preferences=prefs(tmp_path))
    try:
        iq = np.full(2_000_002,1e-5,dtype=complex)
        start,stop = 1_432_100,1_433_100
        iq[start:stop] = .1*(1+.2*np.sin(np.arange(stop-start)))
        w._capture_recording = IQRecording(iq,40e6,2437e6)
        # Analysis coordinates may have a different sample rate from raw power.
        w._recording = SimpleNamespace(sample_rate_hz=20e6)
        w._results = (SimpleNamespace(start_sample=start//2,stop_sample=stop//2),)
        w._render_power()
        trace = w._power_display[-1]
        assert not trace.opts['autoDownsample']
        assert not trace.opts['clipToView']
        plotted = np.rint(trace.xData*40_000).astype(int)
        assert np.isin(np.arange(start,stop),plotted).all()
        np.testing.assert_allclose(trace.yData[np.isin(plotted,np.arange(start,stop))],
            20*np.log10(abs(iq[start:stop]))+w._capture_recording.dbfs_to_dbm_offset_db)
        # Zoom into an unprotected background region to prove resampling occurs.
        w.power_plot.setXRange(10.,10.02,padding=0)
        plotted = np.rint(trace.xData*40_000).astype(int)
        assert np.isin(np.arange(400_000,400_800),plotted).all()
        view_all_traces(w.power_plot)
        left,right = w.power_plot.viewRange()[0]
        assert left<=0 and right>=w._capture_recording.duration_s*1000
        w.reset()
        assert w._power_display is None
    finally:
        w.close()
