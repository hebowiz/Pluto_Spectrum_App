"""Six-pane Non-HT workspace using shared VSA acquisition and display chrome."""
from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from pluto_common.file_dialogs import file_dialog_path, remember_file_directory
from pluto_vsa.analysis import capture_power_traces, recording_spectrum_trace
from pluto_vsa.channel import extract_analysis_channel, validate_analysis_channel_capture
from pluto_vsa.pluto_source import PlutoCaptureSettings, PlutoLiveSource
from pluto_vsa.sources import FileIQSource
from pluto_vsa.ui.capture_thread import PlutoSingleCaptureThread
from pluto_vsa.ui.iq_export import export_iq_recording
from pluto_vsa.ui.packet_decode import PacketDecodeTabs
from pluto_vsa.ui.measurement_chrome import (
    DedicatedSummaryTable, PersistentPlotRanges, IQ_POWER_DISPLAY_FLOOR_DBM, apply_dedicated_table_style,
    configure_iq_power_plot, dedicated_status_color, install_measurement_plot_menu,
    limit_iq_power_display_dbm, make_measurement_dock, make_measurement_plot,
    plot_complex_symbol_distribution, plot_unit_circle, set_iq_plane_range,
    set_iq_power_default_y_range,
)
from . import configuration
from .analysis import analyze_wifi_recording
from .acquisition import WiFiAnalysisThread
from .measurement import packet_power
from .display import power_display_indices
from .summary import visible_results

_CONFIG_KEY = "wifi_dedicated/startup_meas_config"


class WiFiAnalyzerWindow(QtWidgets.QMainWindow):
    analysis_mode_requested = QtCore.Signal(str)
    application_close_requested = QtCore.Signal()
    shutdown_ready = QtCore.Signal()

    def __init__(self, pluto_source=None, preferences=None, owns_pluto_source=False):
        super().__init__()
        self._pluto_source = pluto_source or PlutoLiveSource()
        self._owns_source = owns_pluto_source or pluto_source is None
        self._preferences = preferences or QtCore.QSettings("PlutoSA", "PlutoVSA")
        self._target = ""
        self._recording = self._capture_recording = None
        self._result = None
        self._results = ()
        self._selected_result_index = 0
        self._capture_thread = self._analysis_thread = None
        self._continuous_run_requested = False
        self._run_cancelled = False
        self._shutdown_requested = False
        self._shutdown_finalized = False
        self._layout_initialized = False
        self._power_display = None
        self._updating_power_display = False
        self._build_actions()
        configuration.build_config(self)
        self._build_docks()
        self._default_meas_config = deepcopy(self._meas_config_values())
        saved = self._preferences.value(_CONFIG_KEY)
        if saved:
            try:
                self._apply_meas_config_values(json.loads(str(saved)))
            except (ValueError,TypeError,KeyError):
                self._apply_meas_config_values(self._default_meas_config)
        self.setWindowTitle("Wi-Fi Dedicated Analyzer")
        self.resize(1500,900)
        self._update_actions()
        self.shutdown_ready.connect(self._close_when_ready)

    def _build_actions(self):
        menu = self.menuBar().addMenu("File")
        for attr, label, callback in (
            ("open_iq_action","Import IQ",self._open_iq),
            ("export_iq_action","Export IQ",self._export_iq_recording),
            ("open_config_action","Meas Config",lambda: self.open_config_page("Signal Description")),
            ("run_action","Single",self._toggle_capture),
            ("run_continuous_action","Continuous",self._toggle_continuous_capture),
            ("refresh_analysis_action","Refresh Analysis",self.refresh),
            ("clear_measurement_history_action","Reset",self.reset)):
            action = menu.addAction(label)
            action.triggered.connect(callback)
            setattr(self,attr,action)
        menu.addAction("Close").triggered.connect(self.application_close_requested.emit)
        mode_menu = self.menuBar().addMenu("Analysis Mode")
        for label,mode in (("General VSA...","generic"),("Bluetooth Dedicated Analyzer...","bluetooth"),
                           ("DECT Dedicated Analyzer...","dect"),("Wi-Fi Dedicated Analyzer...","wifi"),
                           ("ADS-B 1090ES...","adsb1090")):
            action = mode_menu.addAction(label)
            action.triggered.connect(lambda checked=False, selected=mode: self.analysis_mode_requested.emit(selected))

    def _dock(self, name, widget):
        return make_measurement_dock(name,widget,self,object_prefix="wifi",closable=False)

    def _build_docks(self):
        self.setDockNestingEnabled(True)
        self.power_plot = make_measurement_plot("IQ Power (dBm)","Time (ms)")
        configure_iq_power_plot(self.power_plot)
        # Our display sampler preserves nonuniform packet detail and timestamps.
        self.power_plot.setDownsampling(auto=False)
        self.power_plot.setClipToView(False)
        self.power_plot.getViewBox().sigXRangeChanged.connect(self._update_power_trace)
        self.power_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.spectrum_plot = make_measurement_plot("Magnitude (dBm)","Frequency (MHz)")
        self.spectrum_tabs = QtWidgets.QTabWidget()
        self.spectrum_tabs.addTab(self.spectrum_plot,"Spectrum")
        self.mask_plot = make_measurement_plot("PSD (dBm/MHz)","Frequency (MHz)")
        self.mask_plot.addLegend()
        self.spectrum_tabs.addTab(self.mask_plot,"Mask (IEEE 2024)")
        self.summary_table = DedicatedSummaryTable()
        self.modulation_tabs = QtWidgets.QTabWidget()
        self.symbol_tabs = QtWidgets.QTabWidget()
        self.constellation_plots = [make_measurement_plot("Q","I") for _ in range(2)]
        for plot in self.constellation_plots:
            # I coordinates are not sorted: time-series clipping/downsampling
            # would discard symbols or invent extrema between unrelated points.
            plot.setDownsampling(auto=False)
            plot.setClipToView(False)
        self.resource_plots = [make_measurement_plot("OFDM Symbol Index","Subcarrier Index") for _ in range(2)]
        for i,label in enumerate(("L-SIG - BPSK","DATA - Auto")):
            self.modulation_tabs.addTab(self.resource_plots[i],label)
            self.symbol_tabs.addTab(self.constellation_plots[i],label)
        self.evm_carrier_plot = make_measurement_plot("EVM RMS (%)","Subcarrier Index")
        self.evm_symbol_plot = make_measurement_plot("EVM RMS (%)","OFDM Symbol Index")
        self.channel_amplitude_plot = make_measurement_plot("Relative magnitude (dB)","Subcarrier Index")
        self.channel_phase_plot = make_measurement_plot("Phase (rad)","Subcarrier Index")
        for label,plot in (("DATA EVM / Carrier",self.evm_carrier_plot),("DATA EVM / Symbol",self.evm_symbol_plot),
                           ("Channel Magnitude",self.channel_amplitude_plot),("Channel Phase",self.channel_phase_plot)):
            self.modulation_tabs.addTab(plot,label)
        self.flatness_plot = make_measurement_plot("Deviation (dB)","Subcarrier Index")
        self.flatness_plot.addLegend()
        self.modulation_tabs.addTab(self.flatness_plot,"Spectral Flatness")
        self.packet_tabs = PacketDecodeTabs()
        self.decode_tree = self.packet_tabs.decode_tree
        self.payload_text = self.packet_tabs.payload_text
        self.issues_table = self.packet_tabs.issues_table
        self.packet_table = QtWidgets.QTableWidget(0,6)
        self.packet_table.setHorizontalHeaderLabels(("#","Rate","Type","Length","FCS","Power"))
        apply_dedicated_table_style(self.packet_table)
        self.packet_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.packet_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.packet_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.packet_table.itemSelectionChanged.connect(self._packet_selected)
        self.packet_tabs.addTab(self.packet_table,"Packet List")
        for attr,name,widget in (("power_dock","IQ Power",self.power_plot),
                                 ("spectrum_dock","Spectrum",self.spectrum_tabs),
                                 ("summary_dock","Result Summary",self.summary_table),
                                 ("modulation_dock","Modulation",self.modulation_tabs),
                                 ("symbol_dock","Symbol Plot",self.symbol_tabs),
                                 ("packet_dock","Packet Analysis",self.packet_tabs)):
            setattr(self,attr,self._dock(name,widget))
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,self.power_dock)
        self.splitDockWidget(self.power_dock,self.spectrum_dock,QtCore.Qt.Orientation.Horizontal)
        self.splitDockWidget(self.spectrum_dock,self.summary_dock,QtCore.Qt.Orientation.Horizontal)
        for top,bottom in ((self.power_dock,self.modulation_dock),(self.spectrum_dock,self.symbol_dock),
                           (self.summary_dock,self.packet_dock)):
            self.splitDockWidget(top,bottom,QtCore.Qt.Orientation.Vertical)
        plots = [self.power_plot,self.spectrum_plot,*self.constellation_plots,*self.resource_plots,
                 self.evm_carrier_plot,self.evm_symbol_plot,self.channel_amplitude_plot,self.channel_phase_plot,
                 self.flatness_plot,self.mask_plot]
        self._plots = [(str(i),p) for i,p in enumerate(plots)]
        self._persistent_plot_ranges = PersistentPlotRanges(self._plots)
        for name,plot in self._plots:
            install_measurement_plot_menu(plot,reset=lambda n=name: self._persistent_plot_ranges.reset(n))

    def showEvent(self,event):
        super().showEvent(event)
        if not self._layout_initialized:
            self._layout_initialized = True
            QtCore.QTimer.singleShot(0,self._equalize_docks)

    def _equalize_docks(self):
        for row in ((self.power_dock,self.spectrum_dock,self.summary_dock),
                    (self.modulation_dock,self.symbol_dock,self.packet_dock)):
            self.resizeDocks(list(row),[500]*3,QtCore.Qt.Orientation.Horizontal)
        for top,bottom in ((self.power_dock,self.modulation_dock),(self.spectrum_dock,self.symbol_dock),
                           (self.summary_dock,self.packet_dock)):
            self.resizeDocks([top,bottom],[450,450],QtCore.Qt.Orientation.Vertical)

    def _meas_config_values(self):
        return configuration.collect(self)

    def _apply_meas_config_values(self,values):
        configuration.apply(self,values)
        if self._results:
            self._render_selected()

    def _save_config(self):
        self._preferences.setValue(_CONFIG_KEY,json.dumps(self._meas_config_values()))

    def open_config_page(self,name):
        if self.shutdown_busy_reason() is None:
            self._meas_config_dialog.open_page(name)

    def set_pluto_target(self,target):
        self._target = str(target)

    def _capture_settings(self):
        fs = float(self.sample_rate_combo.currentData())
        bandwidth = self.analysis_bandwidth_spin.value()*1e6 if self.analysis_check.isChecked() else None
        settings = PlutoCaptureSettings(center_frequency_hz=self.center_spin.value()*1e6,
            symbol_rate_hz=10e6, samples_per_symbol=int(fs/10e6), # Capture-rate factor, not QAM symbol rate.
            capture_length_s=self.duration_spin.value()/1000, rf_bandwidth_hz=self.bandwidth_spin.value()*1e6,
            lo_offset_hz=self.lo_offset_spin.value()*1e6, analysis_bandwidth_hz=bandwidth,
            sdr_uri=self._target or None, swap_iq=self._common_setup.swap_iq.isChecked(),
            power_correction=self._common_setup.power_correction(), **self._trigger_controls.acquisition_settings())
        if settings.nominal_usable_bandwidth_hz < 16.25e6:
            raise ValueError("Pluto usable bandwidth is too narrow for Non-HT: use 40 MS/s and RF bandwidth >= 20 MHz. 20 MS/s canonical IQ is supported offline.")
        validate_analysis_channel_capture(sample_rate_hz=fs,usable_bandwidth_hz=settings.nominal_usable_bandwidth_hz,
            lo_offset_hz=settings.lo_offset_hz,analysis_bandwidth_hz=bandwidth)
        if bandwidth is not None and abs(self.analysis_center_spin.value()*1e6-settings.center_frequency_hz)+bandwidth/2 > settings.nominal_usable_bandwidth_hz/2:
            raise ValueError("Analysis channel exceeds usable capture bandwidth")
        return settings

    def _prepare_recording(self,recording):
        if self.analysis_check.isChecked():
            return extract_analysis_channel(recording,center_frequency_hz=self.analysis_center_spin.value()*1e6,
                                            bandwidth_hz=self.analysis_bandwidth_spin.value()*1e6)
        return recording

    def load_recording(self,recording,*,capture_recording=None):
        if self._analysis_thread is not None or self._shutdown_requested:
            return
        self._capture_recording = capture_recording or recording
        self._run_cancelled = False
        self._recording = self._prepare_recording(recording)
        thread = WiFiAnalysisThread(self._recording,self)
        self._analysis_thread = thread
        thread.analysis_ready.connect(self._analysis_ready)
        thread.analysis_failed.connect(self._failed)
        thread.finished.connect(self._analysis_finished)
        thread.finished.connect(thread.deleteLater)
        self._update_actions()
        thread.start()

    def analyze_recording(self,recording):
        """Synchronous offline entry point for scripts/tests; UI uses a worker."""
        self._run_cancelled = False
        self._capture_recording = recording
        self._recording = self._prepare_recording(recording)
        self._analysis_ready(analyze_wifi_recording(self._recording))

    def _analysis_ready(self,result):
        if self._shutdown_requested or self._run_cancelled:
            return
        self._result = result
        self._results = result.packets
        self._selected_result_index = 0
        self.packet_table.blockSignals(True)
        self.packet_table.setRowCount(len(self._results))
        for index,p in enumerate(self._results):
            c = p.packet.decode_context
            power_plane = self._recording if self.power_filter_check.isChecked() else self._capture_recording
            power,_ = packet_power(power_plane,p.start_sample/self._recording.sample_rate_hz,p.stop_sample/self._recording.sample_rate_hz)
            for column,value in enumerate((index+1,c.get("rate_mbps","—"),p.packet.packet_type or "Incomplete",
                                           c.get("length","—"),p.packet.integrity.crc_valid,f"{power:.2f}")):
                self.packet_table.setItem(index,column,QtWidgets.QTableWidgetItem(str(value)))
        self.packet_table.resizeRowsToContents()
        self.packet_table.blockSignals(False)
        if self._results:
            self.packet_table.selectRow(0)
            self._render_selected()
        else:
            self._clear_results_display()
            self._render_power()
        counts = result.counts
        self.statusBar().showMessage(
            f"Wi-Fi: {counts['detected']} detected | {counts['complete']} complete | "
            f"{counts['measurement_eligible']} measurement eligible | {counts['decode_success']} PHY decoded | "
            f"{counts['fcs_valid']} FCS valid" + (" — " + "; ".join(result.issues) if result.issues else ""))
        self._common_setup.refresh()
        self._update_actions()

    def _packet_selected(self):
        index = self.packet_table.currentRow()
        if 0 <= index < len(self._results):
            self._selected_result_index = index
            self._render_selected()

    def _render_power(self):
        recording = self._recording if self.power_filter_check.isChecked() else self._capture_recording
        if recording is None:
            return
        _,_,power = capture_power_traces(recording)
        scale = recording.sample_rate_hz/self._recording.sample_rate_hz
        ranges = [(int(np.floor(p.start_sample*scale)),int(np.ceil(p.stop_sample*scale))) for p in self._results]
        trace = self.power_plot.plot(pen="y")
        self._power_display = (power,recording.sample_rate_hz,ranges,trace)
        # Keep View All anchored to the capture even after zoom resampling.
        finite = power[np.isfinite(power)]
        bounds = (limit_iq_power_display_dbm(np.array([finite.min(),finite.max()]))
                  if finite.size else np.full(2,IQ_POWER_DISPLAY_FLOOR_DBM))
        self.power_plot.plot([0,recording.duration_s*1000],bounds,pen=None,symbol=None)
        self.power_plot.setXRange(0,recording.duration_s*1000,padding=0)
        self._update_power_trace()
        set_iq_power_default_y_range(self.power_plot,power)

    def _update_power_trace(self,*_):
        if self._power_display is None or self._updating_power_display:
            return
        self._updating_power_display = True
        try:
            power,fs,ranges,trace = self._power_display
            left,right = self.power_plot.viewRange()[0]
            selected = power_display_indices(power,np.floor(left*fs/1000)-1,
                np.ceil(right*fs/1000)+2,ranges,self._selected_result_index)
            trace.setData(selected*1000/fs,limit_iq_power_display_dbm(power[selected]))
        finally:
            self._updating_power_display = False

    def _render_selected(self):
        if not self._results:
            return
        p = self._results[self._selected_result_index]
        recording = self._recording
        self._persistent_plot_ranges.prepare_for_update()
        self._power_display = None
        for _,plot in self._plots:
            plot.clear()
        self._render_power()
        start,stop = p.start_sample/recording.sample_rate_hz, p.stop_sample/recording.sample_rate_hz
        for label,a,b,color in (("L-STF",0,8,(0,180,255,40)),("L-LTF",8,16,(255,180,0,40)),("L-SIG",16,20,(255,0,180,40)),
                          ("DATA",20,(stop-start)*1e6,(0,255,100,30))):
            if b > a:
                overlay = pg.LinearRegionItem(((start+a*1e-6)*1000,(start+b*1e-6)*1000),movable=False,brush=color)
                overlay.setZValue(-10)
                overlay.setToolTip(label)
                self.power_plot.addItem(overlay)
        spectrum_recording = recording if self.spectrum_filter_check.isChecked() else self._capture_recording
        f,mag = recording_spectrum_trace(spectrum_recording,start_time_s=start,stop_time_s=stop)
        self.spectrum_plot.plot(f/1e6,mag,pen="y")
        self.spectrum_plot.enableAutoRange()
        power_recording = recording if self.power_filter_check.isChecked() else self._capture_recording
        average, peak = packet_power(power_recording,start,stop)
        displayed = replace(p,packet_power_dbm=average,peak_power_dbm=peak)
        results = visible_results(displayed,recording,diagnostics=self.diagnostics_check.isChecked(),
            statistics=self._result.measurement_statistics,conditions=configuration.measurement_conditions(self))
        self.summary_table.setRowCount(len(results))
        for row,result in enumerate(results):
            values = result.row()
            for col,value in enumerate(values):
                if col==0 and result.measurement_id=="carrier_frequency_error":
                    value = "Carrier Frequency\nError"
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(result.tooltip())
                item.setData(QtCore.Qt.ItemDataRole.UserRole,result.measurement_id)
                color = dedicated_status_color(values[3].lower())
                if color is not None:
                    item.setForeground(QtGui.QBrush(color))
                self.summary_table.setItem(row,col,item)
        self.summary_table.resizeRowsToContents()
        training = p.rf_details.get("training",{})
        if training:
            self.flatness_plot.plot(training["subcarriers"],training["deviation_db"],pen="y",name="Measured LTF energy")
            self.flatness_plot.plot(training["subcarriers"],training["upper_db"],pen="r",name="IEEE 2024 upper limit")
            self.flatness_plot.plot(training["subcarriers"],training["lower_db"],pen="c",name="IEEE 2024 lower limit")
        self.flatness_plot.setTitle("Received LTF; see Measurement Conditions")
        spectrum = p.rf_details.get("spectrum",{})
        if spectrum:
            frequency = (recording.center_frequency_hz+spectrum["offset_hz"])/1e6
            self.mask_plot.plot(frequency,spectrum["psd_dbm_mhz"],pen="y",name="Measured PSD (digital)")
            self.mask_plot.plot(frequency,spectrum["upper_dbm_mhz"],pen="r",name="IEEE 2024 upper limit (both sides)")
        self.mask_plot.setTitle("Equivalent Digital Measurement; partial span / VBW unavailable")
        carriers = np.array([*range(-26,-21),*range(-20,-7),*range(-6,0),*range(1,7),*range(8,21),*range(22,27)])
        for i,region in enumerate((p.signal,p.data)):
            if region is None:
                continue
            title = region.name+" - "+region.modulation
            self.modulation_tabs.setTabText(i,title)
            self.symbol_tabs.setTabText(i,title)
            plot = self.constellation_plots[i]
            plot_complex_symbol_distribution(
                plot,region.measured.ravel(),density=self.density_check.isChecked(),
                density_spread=self.density_spread_combo.currentData())
            plot_unit_circle(plot)
            plot.setAspectLocked(True,1)
            set_iq_plane_range(plot)
            # Physical subcarrier coordinates: leave DC/pilot/null columns blank.
            grid = np.full((len(region.measured),53),np.nan)
            grid[:,carriers+26] = 100*abs(region.error)
            image = pg.ImageItem(grid,axisOrder="row-major")
            image.setLookupTable(pg.colormap.get("viridis").getLookupTable())
            image.setLevels((0,max(.1,float(np.nanmax(grid)))))
            image.setRect(QtCore.QRectF(-26.5,-.5,53,len(region.measured)))
            self.resource_plots[i].addItem(image)
            self.resource_plots[i].setRange(xRange=(-26.5,26.5),yRange=(-.5,len(region.measured)-.5),padding=0)
            self.resource_plots[i].setTitle(f"EVM (%), dark → bright: 0 → {max(.1,float(np.nanmax(grid))):.2f}")
        if p.data is not None:
            self.evm_carrier_plot.plot(carriers,p.data.evm_per_subcarrier,pen="y")
            self.evm_symbol_plot.plot(np.arange(len(p.data.evm_per_symbol)),p.data.evm_per_symbol,pen="y")
        if p.channel.size:
            h = p.channel/np.sqrt(np.mean(abs(p.channel)**2))
            self.channel_amplitude_plot.plot(p.channel_subcarriers,20*np.log10(np.maximum(abs(h),1e-15)),pen="y")
            self.channel_phase_plot.plot(p.channel_subcarriers,np.unwrap(np.angle(h)),pen="y")
        for plot in (self.evm_carrier_plot,self.evm_symbol_plot,self.channel_amplitude_plot,self.channel_phase_plot,self.flatness_plot,self.mask_plot):
            plot.enableAutoRange()
        self.packet_tabs.render_packet(p.packet)
        self._persistent_plot_ranges.finish_update(contexts={str(i+2):r.modulation if r else "none" for i,r in enumerate((p.signal,p.data))})

    def _clear_results_display(self):
        self._power_display = None
        for _,plot in self._plots:
            plot.clear()
        self.summary_table.setRowCount(0)
        self.packet_tabs.clear_packet()

    def reset(self):
        if self.shutdown_busy_reason() is not None:
            return
        self._recording = self._capture_recording = self._result = None
        self._results = ()
        self.packet_table.setRowCount(0)
        self._clear_results_display()
        self._update_actions()

    def refresh(self):
        if self.shutdown_busy_reason() is None and self._capture_recording is not None:
            self.load_recording(self._capture_recording)

    def _open_iq(self):
        if self.shutdown_busy_reason() is not None:
            return
        path,_ = QtWidgets.QFileDialog.getOpenFileName(self,"Import Wi-Fi IQ",
            file_dialog_path(self._preferences,"directories/iq"),"IQ recordings (*.npz *.iq.tar);;All files (*)")
        if path:
            try:
                self.load_recording(FileIQSource.load(path))
                remember_file_directory(self._preferences,"directories/iq",path)
            except (ValueError,OSError) as error:
                self._failed(str(error))

    def _export_iq_recording(self):
        export_iq_recording(self,self._capture_recording,self._preferences)

    def _toggle_capture(self):
        if self.shutdown_busy_reason() is not None:
            self._stop()
        else:
            self._start_capture()

    def _toggle_continuous_capture(self):
        if self.shutdown_busy_reason() is not None:
            self._stop()
        else:
            self._continuous_run_requested = True
            self._start_capture()

    def _start_capture(self):
        if self._shutdown_requested:
            return
        self._run_cancelled = False
        try:
            settings = self._capture_settings()
        except ValueError as error:
            self._failed(str(error))
            self._continuous_run_requested = False
            self._update_actions()
            return
        thread = PlutoSingleCaptureThread(self._pluto_source,settings,"Waiting for Wi-Fi IQ",self,
                                          prefer_buffered=self._continuous_run_requested)
        self._capture_thread = thread
        thread.capture_armed.connect(self.statusBar().showMessage)
        thread.capture_ready.connect(self._capture_ready)
        thread.capture_failed.connect(self._failed)
        thread.finished.connect(self._capture_finished)
        thread.finished.connect(thread.deleteLater)
        self._update_actions()
        thread.start()

    def _capture_ready(self,recording):
        if not self._shutdown_requested and not self._run_cancelled:
            try:
                self.load_recording(recording)
            except ValueError as error:
                self._failed(str(error))

    def _failed(self,message):
        self._continuous_run_requested = False
        self.statusBar().showMessage("Wi-Fi: "+str(message))
        self._update_actions()

    def _capture_finished(self):
        self._capture_thread = None
        self._work_finished()

    def _analysis_finished(self):
        self._analysis_thread = None
        self._work_finished()

    def _work_finished(self):
        self._update_actions()
        if self._capture_thread is not None or self._analysis_thread is not None:
            return
        if self._continuous_run_requested and not self._shutdown_requested:
            QtCore.QTimer.singleShot(20,self._next_capture)
        else:
            self._pluto_source.stop_stream()
            if self._shutdown_requested:
                self.shutdown_ready.emit()

    def _next_capture(self):
        if self._continuous_run_requested and not self._shutdown_requested and self._capture_thread is None and self._analysis_thread is None:
            self._start_capture()

    def _stop(self):
        was_running = self._continuous_run_requested
        self._run_cancelled = True
        self._continuous_run_requested = False
        for thread in (self._capture_thread,self._analysis_thread):
            if thread is not None:
                thread.requestInterruption()
        if was_running and self._capture_thread is None and self._analysis_thread is None:
            self._pluto_source.stop_stream()
        self._update_actions()

    def _update_actions(self):
        busy = self.shutdown_busy_reason() is not None
        for action in (self.open_iq_action,self.clear_measurement_history_action):
            action.setEnabled(not busy)
        self.export_iq_action.setEnabled(self._capture_recording is not None and not busy)
        self.refresh_analysis_action.setEnabled(self._capture_recording is not None and not busy)
        self._meas_config_dialog.setEnabled(not busy) if hasattr(self,"_meas_config_dialog") else None
        self.run_action.setText("Stop" if busy else "Single")
        self.run_continuous_action.setText("Stop Continuous" if self._continuous_run_requested else "Continuous")

    def shutdown_busy_reason(self):
        if self._capture_thread is not None:
            return "Wi-Fi capture is running"
        if self._analysis_thread is not None:
            return "Wi-Fi analysis is running"
        return "Wi-Fi Continuous is running" if self._continuous_run_requested else None

    def request_shutdown(self):
        self._shutdown_requested = True
        self._stop()

    def finalize_shutdown(self):
        if self._shutdown_finalized:
            return
        self._shutdown_finalized = True
        self._save_config()
        if self._owns_source:
            self._pluto_source.close()

    def _close_when_ready(self):
        if self._shutdown_requested and self.isWindow():
            self.close()

    def closeEvent(self,event):
        self.request_shutdown()
        if self.shutdown_busy_reason() is not None:
            event.ignore()
            return
        self.finalize_shutdown()
        event.accept()
