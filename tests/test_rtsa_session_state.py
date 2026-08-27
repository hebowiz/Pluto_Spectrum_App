from pluto_sa.config.session_state import (
    RTSA_SESSION_KEY,
    RTSASessionState,
    apply_config_values,
    capture_config_values,
    decode_session_state,
    encode_session_state,
    load_session_state,
    save_session_state,
)
from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.modes.analyzer_mode import AnalyzerMode


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


def test_session_state_round_trip():
    config = SpectrumConfig()
    config.center_freq_hz = 2_480_000_000
    config.rbw_hz = 250_000.0
    config.rx_gain_db = 12
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
    assert restored.realtime_graph_view_mode == "waterfall_only"
    assert restored.persistence_enabled is True
    assert restored.traces[0]["average_count"] == 32
    assert restored.markers[0]["is_enabled"] is True


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
