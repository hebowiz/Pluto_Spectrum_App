from pluto_rtsa.config.session_state import (
    RTSA_SESSION_KEY,
    RTSASessionState,
    apply_config_values,
    apply_mode_config_values,
    capture_config_values,
    capture_mode_config_values,
    decode_session_state,
    encode_session_state,
    load_session_state,
    load_session_state_file,
    save_session_state,
    save_session_state_file,
)
from pluto_common.config.spectrum_config import SpectrumConfig
from pluto_common.config.analyzer_mode import AnalyzerMode
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from pluto_common.control_panel import ControlPanelNavigator
import pluto_rtsa.main as main_module
import pluto_rtsa.ui.session_window as session_window_module


class FakeSettings:
    def __init__(self):
        self.values = {}
        self.sync_count = 0

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):
        self.values[key] = value

    def remove(self, key):
        self.values.pop(key, None)

    def sync(self):
        self.sync_count += 1


def test_rtsa_system_frame_uses_state_then_device_and_state_hierarchy():
    pg.mkQApp("RTSA system state controls")
    window = QtWidgets.QMainWindow()
    window.main_menu_page = QtWidgets.QWidget()
    main_layout = QtWidgets.QVBoxLayout(window.main_menu_page)
    main_layout.addStretch(1)
    window.control_stack = QtWidgets.QStackedWidget()
    window.control_stack.addWidget(window.main_menu_page)
    window._make_control_button = lambda text: QtWidgets.QPushButton(text)
    window._apply_groupbox_title_font = lambda _group: None
    window._show_control_page = lambda *_args, **_kwargs: None
    window._on_state_recall_clicked = lambda: None
    window._on_state_save_clicked = lambda: None
    window._on_restore_defaults_clicked = lambda: None
    window._on_device_clicked = lambda: None
    window._install_control_panel_event_filters = lambda: None

    session_window_module.SessionRealtimeSpectrumWindow._install_system_frame(window)

    assert [
        window.system_group.layout().itemAt(index).widget().text()
        for index in range(window.system_group.layout().count())
        if window.system_group.layout().itemAt(index).widget() is not None
    ] == ["State", "Device"]
    assert [
        window.state_page.layout().itemAt(index).widget().text()
        for index in range(window.state_page.layout().count())
        if window.state_page.layout().itemAt(index).widget() is not None
    ] == ["Recall", "Save", "Preset"]
    assert not hasattr(window, "file_button")
    window.close()


def test_shared_control_scroll_resets_child_and_restores_parent():
    app = pg.mkQApp("shared control scroll history")
    panel = QtWidgets.QWidget()
    panel.resize(240, 300)
    layout = QtWidgets.QVBoxLayout(panel)
    title = QtWidgets.QLabel("Main Menu")
    stack = QtWidgets.QStackedWidget()
    main_page = QtWidgets.QWidget()
    main_page.setMinimumHeight(900)
    child_page = QtWidgets.QWidget()
    child_page.setMinimumHeight(700)
    stack.addWidget(main_page)
    stack.addWidget(child_page)
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(stack)
    back = QtWidgets.QPushButton("Back")
    layout.addWidget(title)
    layout.addWidget(scroll, 1)
    layout.addWidget(back)
    navigator = ControlPanelNavigator(
        panel=panel,
        title_label=title,
        stack=stack,
        main_page=main_page,
        back_button=back,
        scroll_area=scroll,
    )
    panel.show()
    app.processEvents()
    main_position = min(120, scroll.verticalScrollBar().maximum())
    assert main_position > 0
    scroll.verticalScrollBar().setValue(main_position)

    navigator.show_page("Child", child_page)
    app.processEvents()
    assert scroll.verticalScrollBar().value() == 0

    scroll.verticalScrollBar().setValue(60)
    navigator.navigate_back()
    app.processEvents()
    assert scroll.verticalScrollBar().value() == main_position
    panel.close()


def test_session_state_round_trip():
    config = SpectrumConfig()
    config.center_freq_hz = 2_480_000_000
    config.rbw_hz = 250_000.0
    config.rx_gain_db = 12
    config.realtime_fft_parameter_mode = "Advanced"
    state = RTSASessionState(
        analyzer_mode=AnalyzerMode.SWEEP_SA,
        config_values=capture_config_values(config),
        realtime_graph_view_mode="waterfall_only",
        persistence_enabled=True,
        traces=({"is_visible": True, "trace_type": "Average", "average_count": 32},),
        markers=({"is_enabled": True, "frequency_hz": 2_480_000_000},),
    )

    restored = decode_session_state(encode_session_state(state))

    assert restored.analyzer_mode == AnalyzerMode.SWEEP_SA
    assert restored.config_values["center_freq_hz"] == 2_480_000_000
    assert restored.config_values["rbw_hz"] == 250_000.0
    assert restored.config_values["realtime_fft_parameter_mode"] == "Advanced"
    assert restored.realtime_graph_view_mode == "waterfall_only"
    assert restored.persistence_enabled is True
    assert restored.traces[0]["average_count"] == 32
    assert restored.markers[0]["is_enabled"] is True


def test_session_state_round_trip_preserves_independent_mode_profiles():
    realtime = SpectrumConfig(analyzer_mode=AnalyzerMode.REALTIME_SA)
    realtime.rbw_hz = 10_000.0
    sweep = SpectrumConfig(analyzer_mode=AnalyzerMode.SWEEP_SA)
    sweep.rbw_hz = 250_000.0
    sweep.sweep_time_ms = 345.0
    state = RTSASessionState(
        analyzer_mode=AnalyzerMode.SWEEP_SA,
        config_values={
            **capture_mode_config_values(sweep),
            "center_freq_hz": 915_000_000,
        },
        shared_center_freq_hz=915_000_000,
        mode_states=(
            RTSASessionState(
                analyzer_mode=AnalyzerMode.REALTIME_SA,
                config_values=capture_mode_config_values(realtime),
            ),
            RTSASessionState(
                analyzer_mode=AnalyzerMode.SWEEP_SA,
                config_values=capture_mode_config_values(sweep),
            ),
        ),
    )

    restored = decode_session_state(encode_session_state(state))
    profiles = {item.analyzer_mode: item for item in restored.mode_states}

    assert restored.shared_center_freq_hz == 915_000_000
    assert profiles[AnalyzerMode.REALTIME_SA].config_values["rbw_hz"] == 10_000.0
    assert profiles[AnalyzerMode.SWEEP_SA].config_values["rbw_hz"] == 250_000.0
    assert profiles[AnalyzerMode.SWEEP_SA].config_values["sweep_time_ms"] == 345.0
    assert all("center_freq_hz" not in item.config_values for item in profiles.values())


def test_schema_one_session_remains_loadable_for_profile_migration():
    restored = decode_session_state(
        '{"schema_version":1,"analyzer_mode":"Sweep SA",'
        '"config":{"center_freq_hz":915000000,"rbw_hz":250000},'
        '"realtime_graph_view_mode":"spectrum_only",'
        '"persistence_enabled":false,"traces":[],"markers":[]}'
    )

    assert restored.analyzer_mode == AnalyzerMode.SWEEP_SA
    assert restored.shared_center_freq_hz == 915_000_000
    assert restored.config_values["rbw_hz"] == 250_000
    assert restored.mode_states == ()


def test_mode_config_restore_keeps_shared_center_and_device():
    saved = SpectrumConfig(analyzer_mode=AnalyzerMode.WIDEBAND_REALTIME_SA)
    saved.center_freq_hz = 915_000_000
    saved.rbw_hz = 100_000.0
    saved.rx_gain_db = 13
    values = capture_mode_config_values(saved)

    target = SpectrumConfig(sdr_uri="serial:RX01")
    target.center_freq_hz = 2_440_000_000
    apply_mode_config_values(target, values)

    assert "center_freq_hz" not in values
    assert target.center_freq_hz == 2_440_000_000
    assert target.sdr_uri == "serial:RX01"
    assert target.rbw_hz == 100_000.0
    assert target.rx_gain_db == 13


def test_shared_center_moves_start_stop_window_without_changing_span():
    config = SpectrumConfig()
    config.use_start_stop_freq = True
    config.display_start_freq_hz = 2_400_000_000
    config.display_stop_freq_hz = 2_420_000_000
    window = type("WindowHarness", (), {"config": config})()

    session_window_module.SessionRealtimeSpectrumWindow._apply_shared_center_to_config(
        window,
        2_450_000_000,
    )

    assert config.center_freq_hz == 2_450_000_000
    assert config.display_span_hz == 20_000_000
    assert config.display_start_freq_hz == 2_440_000_000
    assert config.display_stop_freq_hz == 2_460_000_000


def test_config_restore_does_not_change_device_or_internal_defaults():
    saved = SpectrumConfig()
    saved.center_freq_hz = 915_000_000
    saved.rx_gain_db = 7
    values = capture_config_values(saved)

    target = SpectrumConfig(sdr_uri="serial:ABC")
    target.calibration_offset_db = -55.0
    target.capture_buffer_blocks = 123
    apply_config_values(target, values)

    assert target.center_freq_hz == 915_000_000
    assert target.rx_gain_db == 7
    assert target.sdr_uri == "serial:ABC"
    assert target.calibration_offset_db == -55.0
    assert target.capture_buffer_blocks == 123


def test_qsettings_compatible_save_and_load():
    settings = FakeSettings()
    state = RTSASessionState(
        analyzer_mode=AnalyzerMode.REALTIME_SA,
        config_values=capture_config_values(SpectrumConfig()),
    )

    save_session_state(settings, state)
    restored = load_session_state(settings)

    assert RTSA_SESSION_KEY in settings.values
    assert settings.sync_count == 1
    assert restored is not None
    assert restored.analyzer_mode == AnalyzerMode.REALTIME_SA


def test_session_state_file_round_trip(tmp_path):
    state = RTSASessionState(
        analyzer_mode=AnalyzerMode.WIDEBAND_REALTIME_SA,
        config_values=capture_config_values(SpectrumConfig()),
        shared_center_freq_hz=2_440_000_000,
        profiled_analyzer_mode=AnalyzerMode.WIDEBAND_REALTIME_SA,
    )
    path = tmp_path / "all-modes.rtsastate.json"

    save_session_state_file(path, state)

    assert load_session_state_file(path) == state


def test_complete_state_capture_includes_every_mode_profile():
    active = RTSASessionState(
        analyzer_mode=AnalyzerMode.SWEEP_SA,
        config_values={"center_freq_hz": 915_000_000},
        shared_center_freq_hz=915_000_000,
        mode_states=(
            RTSASessionState(
                analyzer_mode=AnalyzerMode.SWEEP_SA,
                config_values={"sweep_time_ms": 321.0},
            ),
        ),
        profiled_analyzer_mode=AnalyzerMode.SWEEP_SA,
    )
    window = type("WindowHarness", (), {})()
    window._capture_session_state = lambda: active
    window._make_default_session_state = lambda mode: RTSASessionState(
        analyzer_mode=mode,
        config_values={"rbw_hz": 1_000_000.0},
    )
    window._state_as_mode_profile = (
        session_window_module.SessionRealtimeSpectrumWindow._state_as_mode_profile
    )

    captured = (
        session_window_module.SessionRealtimeSpectrumWindow._capture_complete_session_state(
            window
        )
    )

    assert captured.profiled_analyzer_mode == AnalyzerMode.SWEEP_SA
    assert tuple(profile.analyzer_mode for profile in captured.mode_states) == (
        AnalyzerMode.REALTIME_SA,
        AnalyzerMode.WIDEBAND_REALTIME_SA,
        AnalyzerMode.SWEEP_SA,
        AnalyzerMode.HIGH_SPEED_TIME_ANALYZER,
    )
    sweep_profile = next(
        profile
        for profile in captured.mode_states
        if profile.analyzer_mode == AnalyzerMode.SWEEP_SA
    )
    assert sweep_profile.config_values["sweep_time_ms"] == 321.0


def test_device_change_replaces_receiver_and_restarts_current_mode(monkeypatch):
    events = []

    class Receiver:
        def __init__(self, config, name="new", **_kwargs):
            self.config = config
            self.name = name

        def close(self):
            events.append(f"close:{self.name}")

        def stop(self):
            events.append(f"stop:{self.name}")
            return True

    class Timer:
        def stop(self):
            events.append("timer:stop")

    class SweepController:
        def __init__(self, receiver):
            self.receiver = receiver
            self.config = None

    config = SpectrumConfig(sdr_uri="serial:old")
    config.analyzer_mode = AnalyzerMode.HIGH_SPEED_TIME_ANALYZER
    old_receiver = Receiver(config, "old")
    window = type("WindowHarness", (), {})()
    window.config = config
    window.receiver = old_receiver
    window.sweep_controller = SweepController(old_receiver)
    window.timer = Timer()
    window._session_settings = FakeSettings()
    window._session_acquisition_started = True
    window._page_history = [("System", object())]
    window.main_menu_page = object()
    window._reset_all_measurement_state = lambda **kwargs: events.append(
        ("reset", kwargs)
    )
    window.start_initial_acquisition = lambda **kwargs: events.append(
        ("start", config.analyzer_mode, kwargs)
    )
    window._reset_plot_state = lambda: events.append("plot:reset")
    window._update_device_window_title = lambda: events.append("title:update")
    window._show_control_page = lambda *args, **kwargs: events.append(
        ("page", args[0], kwargs)
    )
    window._refresh_status_label = lambda: events.append("status:refresh")
    window._restart_timer_for_current_mode = lambda: events.append("timer:restart")

    monkeypatch.setattr(
        main_module,
        "_choose_pluto_target",
        lambda *_args, **_kwargs: (True, "serial:new"),
    )
    monkeypatch.setattr(session_window_module, "PlutoReceiver", Receiver)

    session_window_module.SessionRealtimeSpectrumWindow._on_device_clicked(window)

    assert config.sdr_uri == "serial:new"
    assert window.receiver is not old_receiver
    assert window.sweep_controller.receiver is window.receiver
    assert window.sweep_controller.config is config
    assert "stop:old" in events
    assert "close:old" in events
    assert ("start", AnalyzerMode.HIGH_SPEED_TIME_ANALYZER, {"force": True}) in events
    assert window._page_history == []


def test_config_control_sync_updates_all_config_backed_selection_groups():
    events = []
    window = type("WindowHarness", (), {})()
    method_names = (
        "_update_analyzer_mode_controls",
        "_update_realtime_sa_controls",
        "_update_sweep_controls",
        "_update_sweep_detector_selection_page",
        "_update_wideband_chunk_width_selection_page",
        "_update_graph_view_controls",
        "_update_persistence_controls",
        "_update_trigger_controls",
        "_update_control_button_value_labels",
    )
    for method_name in method_names:
        setattr(window, method_name, lambda name=method_name: events.append(name))

    from pluto_rtsa.ui.main_window import RealtimeSpectrumWindow

    RealtimeSpectrumWindow._sync_config_backed_controls(window)

    assert events == list(method_names)
