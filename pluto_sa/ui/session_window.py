"""RTSA window extension for session restore and analyzer preset behavior."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pyqtgraph.Qt import QtCore, QtWidgets

from pluto_sa.config.session_state import (
    RTSA_APPLICATION,
    RTSA_DEVICE_KEY,
    RTSA_ORGANIZATION,
    PROFILED_ANALYZER_MODES,
    RTSASessionState,
    apply_config_values,
    apply_mode_config_values,
    capture_config_values,
    capture_mode_config_values,
    clear_session_state,
    load_session_state,
    save_session_state,
)
from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.modes.analyzer_mode import AnalyzerMode
from pluto_sa.sdr.pluto_receiver import PlutoReceiver
from pluto_sa.sdr.continuous_acquisition import ContinuousIQAcquisition
from pluto_sa.ui.main_window import (
    GRAPH_VIEW_BOTH,
    GRAPH_VIEW_OPTIONS,
    MAX_MARKER_FREQUENCY_HZ,
    MAX_MARKER_STEP_HZ,
    MIN_MARKER_FREQUENCY_HZ,
    MIN_MARKER_STEP_HZ,
    MIN_TRACE_AVERAGE_COUNT,
    MAX_TRACE_AVERAGE_COUNT,
    PLOT_SPACING,
    PLOT_WIDTH,
    STATUS_PANEL_HEIGHT,
    TRACE_TYPE_LIVE,
    TRACE_TYPE_OPTIONS,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    RealtimeSpectrumWindow,
)


SYSTEM_FRAME_EXTRA_HEIGHT = 80
SESSION_WINDOW_HEIGHT = WINDOW_HEIGHT + SYSTEM_FRAME_EXTRA_HEIGHT
SESSION_SIDE_PANEL_HEIGHT = SESSION_WINDOW_HEIGHT - 24
SESSION_PLOT_HEIGHT = (
    SESSION_SIDE_PANEL_HEIGHT - STATUS_PANEL_HEIGHT - (PLOT_SPACING * 2)
) // 2
SESSION_DUAL_PLOT_HEIGHT = SESSION_PLOT_HEIGHT * 2 + PLOT_SPACING


class SessionRealtimeSpectrumWindow(RealtimeSpectrumWindow):
    """Add PC-local session persistence and a measurement-style Preset."""

    def __init__(self, *args, **kwargs) -> None:
        self._session_settings = QtCore.QSettings(RTSA_ORGANIZATION, RTSA_APPLICATION)
        self._session_acquisition_started = False
        self._mode_session_states: dict[AnalyzerMode, RTSASessionState] = {}
        self._shared_center_freq_hz: int | None = None
        self._mode_profile_restore_in_progress = False
        self._last_profiled_analyzer_mode = AnalyzerMode.REALTIME_SA
        super().__init__(*args, **kwargs)
        self._resize_for_system_frame()
        self._install_system_frame()
        self._apply_display_mode()
        self._restore_saved_session_on_startup()

    def _resize_for_system_frame(self) -> None:
        self.setFixedSize(WINDOW_WIDTH, SESSION_WINDOW_HEIGHT)
        if hasattr(self, "left_panel"):
            self.left_panel.setFixedHeight(SESSION_SIDE_PANEL_HEIGHT)
        if hasattr(self, "control_panel"):
            self.control_panel.setFixedHeight(SESSION_SIDE_PANEL_HEIGHT)

    def _apply_display_mode(self) -> None:
        """Keep the fixed-size plot geometry aligned with the taller window."""

        super()._apply_display_mode()
        if not hasattr(self, "waterfall_plot") or not hasattr(self, "spectrum_plot"):
            return
        if self.graph_view_mode == "waterfall_only":
            self.waterfall_plot.setFixedSize(PLOT_WIDTH, SESSION_DUAL_PLOT_HEIGHT)
            self.spectrum_plot.setFixedSize(PLOT_WIDTH, SESSION_PLOT_HEIGHT)
        elif self.graph_view_mode == "spectrum_only":
            self.spectrum_plot.setFixedSize(PLOT_WIDTH, SESSION_DUAL_PLOT_HEIGHT)
            self.waterfall_plot.setFixedSize(PLOT_WIDTH, SESSION_PLOT_HEIGHT)
        else:
            self.waterfall_plot.setFixedSize(PLOT_WIDTH, SESSION_PLOT_HEIGHT)
            self.spectrum_plot.setFixedSize(PLOT_WIDTH, SESSION_PLOT_HEIGHT)

    def _install_system_frame(self) -> None:
        """Add SYSTEM -> System -> Preset / Device navigation."""

        system_group = QtWidgets.QGroupBox("SYSTEM")
        self._apply_groupbox_title_font(system_group)
        system_layout = QtWidgets.QVBoxLayout(system_group)
        self.system_menu_button = self._make_control_button("System")
        system_layout.addWidget(self.system_menu_button)

        main_layout = self.main_menu_page.layout()
        insert_index = max(0, main_layout.count() - 1)
        main_layout.insertWidget(insert_index, system_group)

        self.system_page = QtWidgets.QWidget()
        system_page_layout = QtWidgets.QVBoxLayout(self.system_page)
        system_page_layout.setContentsMargins(0, 0, 0, 0)
        system_page_layout.setSpacing(10)
        self.preset_button = self._make_control_button("Preset")
        self.device_button = self._make_control_button("Device")
        system_page_layout.addWidget(self.preset_button)
        system_page_layout.addWidget(self.device_button)
        system_page_layout.addStretch(1)
        self.control_stack.addWidget(self.system_page)

        self.system_menu_button.clicked.connect(
            lambda: self._show_control_page("System", self.system_page)
        )
        self.preset_button.clicked.connect(self._on_restore_defaults_clicked)
        self.device_button.clicked.connect(self._on_device_clicked)
        self._install_control_panel_event_filters()

    def _on_device_clicked(self) -> None:
        """Select another Pluto, initialize it, and restart the current mode."""

        # Keep startup and in-application device identity rules identical.
        # The local import avoids making the application entry point a module
        # dependency while session_window itself is imported by that entry point.
        from pluto_sa.main import _choose_pluto_target

        accepted, selector = _choose_pluto_target(self, force_prompt=True)
        if not accepted:
            return
        if selector == self.config.sdr_uri:
            return

        old_receiver = self.receiver
        old_selector = self.config.sdr_uri
        self.timer.stop()
        old_acquisition = getattr(self, "iq_acquisition", None)
        stopped = (
            old_acquisition.stop()
            if old_acquisition is not None
            else old_receiver.stop()
        )
        if not stopped:
            self._restart_timer_for_current_mode()
            QtWidgets.QMessageBox.critical(
                self,
                "ADALM-Pluto Connection Error",
                "The current receiver is still stopping. Device selection was not changed.",
            )
            return
        self._reset_all_measurement_state(
            stop_receiver=False,
            stop_sweep=True,
            reset_markers=False,
        )
        self._session_acquisition_started = False
        self.config.sdr_uri = selector
        try:
            new_receiver = PlutoReceiver(
                self.config,
                owner_application="Pluto RTSA",
            )
        except Exception as error:
            self.config.sdr_uri = old_selector
            self._update_device_window_title()
            self._session_settings.setValue(RTSA_DEVICE_KEY, old_selector or "")
            self._session_settings.sync()
            self.receiver = old_receiver
            if old_acquisition is not None:
                self.iq_acquisition = old_acquisition
            self.sweep_controller.receiver = old_receiver
            self.start_initial_acquisition(force=True)
            QtWidgets.QMessageBox.critical(
                self,
                "ADALM-Pluto Connection Error",
                f"Could not initialize the selected receiver.\n\n{error}",
            )
            return

        self.receiver = new_receiver
        self.iq_acquisition = ContinuousIQAcquisition(new_receiver)
        self._update_device_window_title()
        self.sweep_controller.receiver = new_receiver
        self.sweep_controller.config = self.config
        old_receiver.close()
        self._reset_plot_state()
        self.start_initial_acquisition(force=True)
        self._page_history.clear()
        self._show_control_page("Main Menu", self.main_menu_page, push_history=False)
        self._refresh_status_label()

    def _restore_saved_session_on_startup(self) -> None:
        state = load_session_state(self._session_settings)
        if state is None:
            self._shared_center_freq_hz = int(self.config.center_freq_hz)
            self._mode_session_states[self.config.analyzer_mode] = self._capture_mode_session_state()
            return
        try:
            self._load_mode_session_states(state)
            target_mode = self._normalize_profiled_mode(
                state.profiled_analyzer_mode or state.analyzer_mode
            )
            target_state = self._mode_session_states.get(target_mode)
            if target_state is None:
                target_state = self._legacy_mode_state(state, target_mode)
                self._mode_session_states[target_mode] = target_state
            active_state = self._state_with_shared_center(target_state)
            if state.analyzer_mode == AnalyzerMode.CALIBRATION:
                active_state = RTSASessionState(
                    analyzer_mode=AnalyzerMode.CALIBRATION,
                    config_values=active_state.config_values,
                    realtime_graph_view_mode=active_state.realtime_graph_view_mode,
                    persistence_enabled=active_state.persistence_enabled,
                    traces=active_state.traces,
                    markers=active_state.markers,
                )
            self._mode_profile_restore_in_progress = True
            try:
                self._apply_session_state(active_state, start_if_needed=False)
            finally:
                self._mode_profile_restore_in_progress = False
            self._last_profiled_analyzer_mode = target_mode
        except Exception as exc:
            print(f"[RTSA] Saved session restore failed; using defaults: {exc}")
            clear_session_state(self._session_settings)
            self._mode_session_states.clear()
            default_state = self._make_default_session_state()
            self._shared_center_freq_hz = int(
                default_state.config_values["center_freq_hz"]
            )
            self._mode_session_states[AnalyzerMode.REALTIME_SA] = (
                self._state_as_mode_profile(default_state)
            )
            self._mode_profile_restore_in_progress = True
            try:
                self._apply_session_state(default_state, start_if_needed=False)
            finally:
                self._mode_profile_restore_in_progress = False

    @staticmethod
    def _normalize_profiled_mode(mode: AnalyzerMode) -> AnalyzerMode:
        if mode == AnalyzerMode.TIME_ANALYZER:
            return AnalyzerMode.HIGH_SPEED_TIME_ANALYZER
        if mode not in PROFILED_ANALYZER_MODES:
            return AnalyzerMode.REALTIME_SA
        return mode

    def _load_mode_session_states(self, state: RTSASessionState) -> None:
        shared_center = state.shared_center_freq_hz
        if shared_center is None:
            shared_center = state.config_values.get(
                "center_freq_hz",
                self.config.center_freq_hz,
            )
        self._shared_center_freq_hz = int(shared_center)
        self._mode_session_states = {
            self._normalize_profiled_mode(mode_state.analyzer_mode): mode_state
            for mode_state in state.mode_states
            if self._normalize_profiled_mode(mode_state.analyzer_mode)
            in PROFILED_ANALYZER_MODES
        }
        if not self._mode_session_states:
            target_mode = self._normalize_profiled_mode(state.analyzer_mode)
            self._mode_session_states[target_mode] = self._legacy_mode_state(
                state,
                target_mode,
            )

    def _legacy_mode_state(
        self,
        state: RTSASessionState,
        target_mode: AnalyzerMode,
    ) -> RTSASessionState:
        return RTSASessionState(
            analyzer_mode=target_mode,
            config_values={
                key: value
                for key, value in state.config_values.items()
                if key != "center_freq_hz"
            },
            realtime_graph_view_mode=state.realtime_graph_view_mode,
            persistence_enabled=state.persistence_enabled,
            traces=state.traces,
            markers=state.markers,
        )

    def _make_default_session_state(
        self,
        analyzer_mode: AnalyzerMode = AnalyzerMode.REALTIME_SA,
    ) -> RTSASessionState:
        analyzer_mode = self._normalize_profiled_mode(analyzer_mode)
        defaults = SpectrumConfig(
            analyzer_mode=analyzer_mode,
            sdr_uri=self.config.sdr_uri,
        )
        if self._shared_center_freq_hz is not None:
            defaults.center_freq_hz = int(self._shared_center_freq_hz)
        traces = tuple(
            {
                "is_visible": index == 0,
                "trace_type": TRACE_TYPE_LIVE,
                "hold_enabled": False,
                "average_count": 10,
            }
            for index in range(4)
        )
        markers = tuple(
            {
                "is_enabled": False,
                "trace_name": "Trace1",
                "frequency_hz": int(defaults.center_freq_hz),
                "step_hz": 1_000_000,
                "time_sec": 0.0,
                "time_step_sec": 0.1,
                "continuous_peak_enabled": False,
            }
            for _index in range(4)
        )
        return RTSASessionState(
            analyzer_mode=analyzer_mode,
            config_values=capture_config_values(defaults),
            realtime_graph_view_mode=GRAPH_VIEW_BOTH,
            persistence_enabled=False,
            traces=traces,
            markers=markers,
        )

    def _capture_mode_session_state(
        self,
        *,
        analyzer_mode: AnalyzerMode | None = None,
        config_for_save: SpectrumConfig | None = None,
    ) -> RTSASessionState:
        target_mode = self._normalize_profiled_mode(
            analyzer_mode or self.config.analyzer_mode
        )
        config_for_save = config_for_save or self.config
        graph_view_mode = self.graph_view_mode
        if graph_view_mode not in GRAPH_VIEW_OPTIONS:
            graph_view_mode = GRAPH_VIEW_BOTH

        traces = tuple(
            {
                "is_visible": bool(trace.is_visible),
                "trace_type": str(trace.trace_type),
                "hold_enabled": bool(trace.hold_enabled),
                "average_count": int(trace.average_count),
            }
            for trace in self.trace_states
        )
        markers = tuple(
            {
                "is_enabled": bool(marker.is_enabled),
                "trace_name": str(marker.trace_name),
                "frequency_hz": int(marker.frequency_hz),
                "step_hz": int(marker.step_hz),
                "time_sec": float(marker.time_sec),
                "time_step_sec": float(marker.time_step_sec),
                "continuous_peak_enabled": bool(marker.continuous_peak_enabled),
            }
            for marker in self.marker_states
        )
        return RTSASessionState(
            analyzer_mode=target_mode,
            config_values=capture_mode_config_values(config_for_save),
            realtime_graph_view_mode=graph_view_mode,
            persistence_enabled=bool(self.persistence_enabled),
            traces=traces,
            markers=markers,
        )

    @staticmethod
    def _state_as_mode_profile(state: RTSASessionState) -> RTSASessionState:
        return RTSASessionState(
            analyzer_mode=state.analyzer_mode,
            config_values={
                key: value
                for key, value in state.config_values.items()
                if key != "center_freq_hz"
            },
            realtime_graph_view_mode=state.realtime_graph_view_mode,
            persistence_enabled=state.persistence_enabled,
            traces=state.traces,
            markers=state.markers,
        )

    def _state_with_shared_center(self, state: RTSASessionState) -> RTSASessionState:
        center_freq_hz = (
            int(self.config.center_freq_hz)
            if self._shared_center_freq_hz is None
            else int(self._shared_center_freq_hz)
        )
        return RTSASessionState(
            analyzer_mode=state.analyzer_mode,
            config_values={**state.config_values, "center_freq_hz": center_freq_hz},
            realtime_graph_view_mode=state.realtime_graph_view_mode,
            persistence_enabled=state.persistence_enabled,
            traces=state.traces,
            markers=state.markers,
        )

    def _capture_session_state(self) -> RTSASessionState:
        active_mode = self.config.analyzer_mode
        current_mode = active_mode
        config_for_save = self.config
        if (
            current_mode == AnalyzerMode.CALIBRATION
            and self._calibration_mode_saved_config is not None
        ):
            config_for_save = self._calibration_mode_saved_config
            current_mode = self._normalize_profiled_mode(
                config_for_save.analyzer_mode
            )
            self._shared_center_freq_hz = int(config_for_save.center_freq_hz)
            current_state = self._mode_session_states.get(current_mode)
            if current_state is None:
                current_state = self._capture_mode_session_state(
                    analyzer_mode=current_mode,
                    config_for_save=config_for_save,
                )
                self._mode_session_states[current_mode] = current_state
        else:
            current_mode = self._normalize_profiled_mode(current_mode)
            self._shared_center_freq_hz = int(config_for_save.center_freq_hz)
            current_state = self._capture_mode_session_state(
                analyzer_mode=current_mode,
                config_for_save=config_for_save,
            )
            self._mode_session_states[current_mode] = current_state
        active_state = self._state_with_shared_center(current_state)
        return RTSASessionState(
            analyzer_mode=active_mode,
            config_values=active_state.config_values,
            realtime_graph_view_mode=active_state.realtime_graph_view_mode,
            persistence_enabled=active_state.persistence_enabled,
            traces=active_state.traces,
            markers=active_state.markers,
            shared_center_freq_hz=self._shared_center_freq_hz,
            mode_states=tuple(
                self._mode_session_states[mode]
                for mode in PROFILED_ANALYZER_MODES
                if mode in self._mode_session_states
            ),
            profiled_analyzer_mode=current_mode,
        )

    def _apply_shared_center_to_config(self, center_freq_hz: int) -> None:
        """Move a mode-local start/stop window to the one shared center."""

        center_freq_hz = int(center_freq_hz)
        self.config.center_freq_hz = center_freq_hz
        if (
            self.config.use_start_stop_freq
            and self.config.display_start_freq_hz is not None
            and self.config.display_stop_freq_hz is not None
        ):
            width_hz = max(
                1,
                int(self.config.display_stop_freq_hz)
                - int(self.config.display_start_freq_hz),
            )
            self.config.display_span_hz = width_hz
            self.config.display_start_freq_hz = center_freq_hz - width_hz // 2
            self.config.display_stop_freq_hz = (
                self.config.display_start_freq_hz + width_hz
            )

    def _apply_mode_profile_before_switch(self, state: RTSASessionState) -> None:
        current_selector = self.config.sdr_uri
        apply_mode_config_values(self.config, state.config_values)
        self.config.sdr_uri = current_selector
        center_freq_hz = (
            self.config.center_freq_hz
            if self._shared_center_freq_hz is None
            else self._shared_center_freq_hz
        )
        self._apply_shared_center_to_config(int(center_freq_hz))
        graph_view_mode = str(state.realtime_graph_view_mode)
        if graph_view_mode not in GRAPH_VIEW_OPTIONS:
            graph_view_mode = GRAPH_VIEW_BOTH
        self.graph_view_mode = graph_view_mode
        self._saved_realtime_graph_view_mode = graph_view_mode
        self._sync_amplitude_scale_from_config()
        self.receiver.set_gain_db(int(self.config.rx_gain_db))

    def _restore_mode_profile_after_switch(self, state: RTSASessionState) -> None:
        self._restore_trace_state(state.traces)
        self._restore_marker_state(state.markers)
        self._restore_persistence_state(bool(state.persistence_enabled))
        self._sync_config_backed_controls()
        self._refresh_status_label()
        self._update_trace_menu_buttons()
        self._update_marker_menu_buttons()
        self._update_graph_view_controls()
        self._update_persistence_controls()
        self._apply_display_mode()

    def _change_analyzer_mode(self, analyzer_mode: AnalyzerMode) -> None:
        """Switch modes through independent profiles with one shared center."""

        if self._mode_profile_restore_in_progress:
            super()._change_analyzer_mode(analyzer_mode)
            return
        if analyzer_mode == AnalyzerMode.TIME_ANALYZER:
            analyzer_mode = AnalyzerMode.HIGH_SPEED_TIME_ANALYZER
        previous_mode = self.config.analyzer_mode
        if previous_mode == analyzer_mode:
            return

        if previous_mode in PROFILED_ANALYZER_MODES:
            self._shared_center_freq_hz = int(self.config.center_freq_hz)
            self._mode_session_states[previous_mode] = (
                self._capture_mode_session_state(analyzer_mode=previous_mode)
            )
            self._last_profiled_analyzer_mode = previous_mode

        # Stop the old mode before restoring the target profile.  In
        # particular, _apply_mode_profile_before_switch writes Internal Gain;
        # doing that while an RTSA/HSTA refill is outstanding can block mode
        # switching for seconds or leave stale IQ queued for the new mode.
        self._quiesce_acquisition_for_mode_switch()

        target_state: RTSASessionState | None = None
        if analyzer_mode in PROFILED_ANALYZER_MODES:
            target_state = self._mode_session_states.get(analyzer_mode)
            if target_state is None:
                target_state = self._state_as_mode_profile(
                    self._make_default_session_state(analyzer_mode)
                )
                self._mode_session_states[analyzer_mode] = target_state
            self._apply_mode_profile_before_switch(target_state)
            if previous_mode == AnalyzerMode.CALIBRATION:
                # The selected mode profile supersedes Calibration's temporary
                # return snapshot. The snapshot's mode was already saved on entry.
                self._calibration_mode_saved_config = None

        if analyzer_mode == AnalyzerMode.CALIBRATION:
            save_session_state(self._session_settings, self._capture_session_state())

        super()._change_analyzer_mode(analyzer_mode)
        self._session_acquisition_started = True

        if target_state is not None:
            self._last_profiled_analyzer_mode = analyzer_mode
            self._restore_mode_profile_after_switch(target_state)
            save_session_state(self._session_settings, self._capture_session_state())

    def _apply_session_state(
        self,
        state: RTSASessionState,
        *,
        start_if_needed: bool,
    ) -> None:
        """Apply saved user settings without changing the selected Pluto."""

        current_selector = self.config.sdr_uri
        previous_mode = self.config.analyzer_mode
        target_mode = state.analyzer_mode
        if target_mode == AnalyzerMode.TIME_ANALYZER:
            target_mode = AnalyzerMode.HIGH_SPEED_TIME_ANALYZER

        self._session_acquisition_started = False
        self.timer.stop()
        self._reset_all_measurement_state(
            stop_receiver=True,
            stop_sweep=True,
            reset_markers=False,
        )
        self._calibration_mode_saved_config = None

        apply_config_values(self.config, state.config_values)
        self.config.sdr_uri = current_selector
        self._apply_shared_center_to_config(
            int(
                state.config_values.get(
                    "center_freq_hz",
                    self._shared_center_freq_hz or self.config.center_freq_hz,
                )
            )
        )
        self._sync_amplitude_scale_from_config()
        self.receiver.set_gain_db(int(self.config.rx_gain_db))

        graph_view_mode = str(state.realtime_graph_view_mode)
        if graph_view_mode not in GRAPH_VIEW_OPTIONS:
            graph_view_mode = GRAPH_VIEW_BOTH
        self.graph_view_mode = graph_view_mode
        self._saved_realtime_graph_view_mode = graph_view_mode

        if previous_mode != target_mode:
            self._change_analyzer_mode(target_mode)
            self._session_acquisition_started = True
        else:
            self.config.analyzer_mode = target_mode
            self._rebuild_realtime_runtime_after_mode_change()
            self.receiver.set_gain_db(int(self.config.rx_gain_db))
            self._reset_plot_state()
            self._apply_analyzer_mode_ui_constraints()

        self._restore_trace_state(state.traces)
        self._restore_marker_state(state.markers)
        self._restore_persistence_state(bool(state.persistence_enabled))
        self._sync_config_backed_controls()
        self._refresh_status_label()
        self._update_trace_menu_buttons()
        self._update_marker_menu_buttons()
        self._update_graph_view_controls()
        self._update_persistence_controls()
        self._apply_display_mode()

        if target_mode == AnalyzerMode.CALIBRATION:
            self._show_control_page("Calibrate", self.calibration_page, push_history=False)
        else:
            self._page_history.clear()
            self._show_control_page("Main Menu", self.main_menu_page, push_history=False)

        if start_if_needed and not self._session_acquisition_started:
            self.start_initial_acquisition(force=True)

    def _restore_trace_state(self, saved_traces: tuple[dict[str, Any], ...]) -> None:
        for index, trace in enumerate(self.trace_states):
            if index >= len(saved_traces):
                break
            saved = saved_traces[index]
            trace.is_visible = bool(saved.get("is_visible", trace.is_visible))
            trace_type = str(saved.get("trace_type", trace.trace_type))
            trace.trace_type = trace_type if trace_type in TRACE_TYPE_OPTIONS else TRACE_TYPE_LIVE
            trace.hold_enabled = bool(saved.get("hold_enabled", trace.hold_enabled))
            average_count = int(saved.get("average_count", trace.average_count))
            trace.average_count = max(
                MIN_TRACE_AVERAGE_COUNT,
                min(MAX_TRACE_AVERAGE_COUNT, average_count),
            )
            trace.display_db = None
            trace.max_hold_power = None
            trace.average_power = None
            self._update_trace_control_state(index)
        self._update_trace_curves()

    def _restore_marker_state(self, saved_markers: tuple[dict[str, Any], ...]) -> None:
        trace_names = set(self.marker_trace_options)
        for index, marker in enumerate(self.marker_states):
            if index >= len(saved_markers):
                break
            saved = saved_markers[index]
            marker.is_enabled = bool(saved.get("is_enabled", marker.is_enabled))
            trace_name = str(saved.get("trace_name", marker.trace_name))
            marker.trace_name = trace_name if trace_name in trace_names else "Trace1"
            marker.frequency_hz = max(
                MIN_MARKER_FREQUENCY_HZ,
                min(MAX_MARKER_FREQUENCY_HZ, int(saved.get("frequency_hz", marker.frequency_hz))),
            )
            marker.step_hz = max(
                MIN_MARKER_STEP_HZ,
                min(MAX_MARKER_STEP_HZ, int(saved.get("step_hz", marker.step_hz))),
            )
            marker.time_sec = max(0.0, float(saved.get("time_sec", marker.time_sec)))
            marker.time_step_sec = max(
                1e-9,
                float(saved.get("time_step_sec", marker.time_step_sec)),
            )
            marker.continuous_peak_enabled = bool(
                saved.get("continuous_peak_enabled", marker.continuous_peak_enabled)
            )
            marker.sweep_snapshot_power_db = None
            self._update_marker_control_state(index)
        self._update_marker_items()

    def _restore_persistence_state(self, enabled: bool) -> None:
        supported = self.config.analyzer_mode in (
            AnalyzerMode.REALTIME_SA,
            AnalyzerMode.WIDEBAND_REALTIME_SA,
        )
        self.persistence_enabled = bool(enabled and supported)
        if self.persistence_enabled:
            self._initialize_persistence_buffer()
        else:
            self._apply_persistence_visibility()

    def start_initial_acquisition(self, *, force: bool = False) -> None:
        """Start the correct acquisition path for the restored analyzer mode."""

        if self._session_acquisition_started and not force:
            return
        if self.config.analyzer_mode == AnalyzerMode.SWEEP_SA:
            self._start_sweep_continuous()
        elif self.config.analyzer_mode == AnalyzerMode.WIDEBAND_REALTIME_SA:
            self._start_wideband_continuous()
        elif self.config.analyzer_mode == AnalyzerMode.HIGH_SPEED_TIME_ANALYZER:
            self._start_high_speed_time_analyzer_continuous()
        elif self.config.analyzer_mode == AnalyzerMode.TIME_ANALYZER:
            self._start_time_analyzer_continuous()
        else:
            self._start_realtime_continuous()
        self._session_acquisition_started = True

    def _on_restore_defaults_clicked(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self,
            "Preset",
            "Restore analyzer settings to defaults?\n\nThe selected Pluto receiver will be kept.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._mode_session_states.clear()
        self._shared_center_freq_hz = None
        state = self._make_default_session_state()
        self._shared_center_freq_hz = int(state.config_values["center_freq_hz"])
        self._mode_session_states[state.analyzer_mode] = self._state_as_mode_profile(
            state
        )
        self._mode_profile_restore_in_progress = True
        try:
            self._apply_session_state(state, start_if_needed=True)
        finally:
            self._mode_profile_restore_in_progress = False
        save_session_state(self._session_settings, self._capture_session_state())

    def closeEvent(self, event) -> None:
        try:
            save_session_state(self._session_settings, self._capture_session_state())
        except Exception as exc:
            print(f"[RTSA] Failed to save session state: {exc}")
        super().closeEvent(event)
