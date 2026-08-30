"""Persist user-facing RTSA session state between launches."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.modes.analyzer_mode import AnalyzerMode


RTSA_ORGANIZATION = "PlutoSpectrumApp"
RTSA_APPLICATION = "PlutoRTSA"
RTSA_DEVICE_KEY = "pluto_rx/selector"
RTSA_SESSION_KEY = "session/state_json"
RTSA_SESSION_SCHEMA_VERSION = 2

PROFILED_ANALYZER_MODES = (
    AnalyzerMode.REALTIME_SA,
    AnalyzerMode.WIDEBAND_REALTIME_SA,
    AnalyzerMode.SWEEP_SA,
    AnalyzerMode.HIGH_SPEED_TIME_ANALYZER,
)

# Explicitly persist user-facing analyzer settings only. Hardware selection,
# calibration constants, queue/buffer tuning, profiling switches, and other
# implementation details intentionally keep their code-defined defaults.
PERSISTED_CONFIG_FIELDS = (
    "center_freq_hz",
    "center_freq_step_mhz",
    "display_span_hz",
    "use_start_stop_freq",
    "display_start_freq_hz",
    "display_stop_freq_hz",
    "rbw_hz",
    "rx_gain_db",
    "ref_level_dbm",
    "display_range_db",
    "ext_att_db",
    "ext_gain_db",
    "remove_dc_offset",
    "fft_size",
    "realtime_fft_parameter_mode",
    "realtime_min_display_bins",
    "realtime_overlap_ratio",
    "realtime_fft_rate_limit_hz",
    "waterfall_history",
    "persistence_decay_mode",
    "wideband_chunk_width_hz",
    "sweep_points",
    "sweep_time_ms",
    "sweep_detector_mode",
    "time_analyzer_sample_rate_hz",
    "time_analyzer_rf_bandwidth_hz",
    "time_analyzer_time_span_s",
    "time_analyzer_display_points",
    "hsta_trigger_kind",
    "hsta_trigger_run_mode",
    "hsta_trigger_slope",
    "hsta_trigger_level_dbm",
    "hsta_trigger_hysteresis_db",
    "hsta_trigger_position_percent",
    "hsta_trigger_auto_timeout_s",
)

MODE_LOCAL_CONFIG_FIELDS = tuple(
    field_name for field_name in PERSISTED_CONFIG_FIELDS if field_name != "center_freq_hz"
)


@dataclass(frozen=True)
class RTSASessionState:
    """Serializable user session without runtime measurement buffers."""

    analyzer_mode: AnalyzerMode
    config_values: dict[str, Any]
    realtime_graph_view_mode: str = "both"
    persistence_enabled: bool = False
    traces: tuple[dict[str, Any], ...] = ()
    markers: tuple[dict[str, Any], ...] = ()
    shared_center_freq_hz: int | None = None
    mode_states: tuple["RTSASessionState", ...] = field(default_factory=tuple)
    profiled_analyzer_mode: AnalyzerMode | None = None

    def _to_profile_mapping(self) -> dict[str, Any]:
        return {
            "analyzer_mode": self.analyzer_mode.value,
            "config": {
                key: value
                for key, value in self.config_values.items()
                if key in MODE_LOCAL_CONFIG_FIELDS
            },
            "realtime_graph_view_mode": str(self.realtime_graph_view_mode),
            "persistence_enabled": bool(self.persistence_enabled),
            "traces": [dict(item) for item in self.traces],
            "markers": [dict(item) for item in self.markers],
        }

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": RTSA_SESSION_SCHEMA_VERSION,
            "analyzer_mode": self.analyzer_mode.value,
            "config": dict(self.config_values),
            "realtime_graph_view_mode": str(self.realtime_graph_view_mode),
            "persistence_enabled": bool(self.persistence_enabled),
            "traces": [dict(item) for item in self.traces],
            "markers": [dict(item) for item in self.markers],
            "shared_center_freq_hz": (
                None
                if self.shared_center_freq_hz is None
                else int(self.shared_center_freq_hz)
            ),
            "mode_states": [state._to_profile_mapping() for state in self.mode_states],
            "profiled_analyzer_mode": (
                None
                if self.profiled_analyzer_mode is None
                else self.profiled_analyzer_mode.value
            ),
        }

    @classmethod
    def _from_profile_mapping(cls, payload: Mapping[str, Any]) -> "RTSASessionState":
        analyzer_mode = AnalyzerMode(str(payload["analyzer_mode"]))
        if analyzer_mode == AnalyzerMode.TIME_ANALYZER:
            analyzer_mode = AnalyzerMode.HIGH_SPEED_TIME_ANALYZER
        if analyzer_mode not in PROFILED_ANALYZER_MODES:
            raise ValueError(f"unsupported profiled analyzer mode: {analyzer_mode.value}")
        raw_config = payload.get("config", {})
        if not isinstance(raw_config, Mapping):
            raise ValueError("RTSA mode config must be an object")
        raw_traces = payload.get("traces", [])
        raw_markers = payload.get("markers", [])
        if not isinstance(raw_traces, list) or not isinstance(raw_markers, list):
            raise ValueError("RTSA trace/marker state must be arrays")
        return cls(
            analyzer_mode=analyzer_mode,
            config_values={
                field_name: raw_config[field_name]
                for field_name in MODE_LOCAL_CONFIG_FIELDS
                if field_name in raw_config
            },
            realtime_graph_view_mode=str(payload.get("realtime_graph_view_mode", "both")),
            persistence_enabled=bool(payload.get("persistence_enabled", False)),
            traces=tuple(dict(item) for item in raw_traces if isinstance(item, Mapping)),
            markers=tuple(dict(item) for item in raw_markers if isinstance(item, Mapping)),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RTSASessionState":
        version = int(payload.get("schema_version", 0))
        if version not in (1, RTSA_SESSION_SCHEMA_VERSION):
            raise ValueError(f"unsupported RTSA session schema version: {version}")
        analyzer_mode = AnalyzerMode(str(payload["analyzer_mode"]))
        raw_config = payload.get("config", {})
        if not isinstance(raw_config, Mapping):
            raise ValueError("RTSA session config must be an object")
        config_values = {
            field_name: raw_config[field_name]
            for field_name in PERSISTED_CONFIG_FIELDS
            if field_name in raw_config
        }
        raw_traces = payload.get("traces", [])
        raw_markers = payload.get("markers", [])
        if not isinstance(raw_traces, list) or not isinstance(raw_markers, list):
            raise ValueError("RTSA trace/marker state must be arrays")
        traces = tuple(dict(item) for item in raw_traces if isinstance(item, Mapping))
        markers = tuple(dict(item) for item in raw_markers if isinstance(item, Mapping))
        mode_states: tuple[RTSASessionState, ...] = ()
        shared_center_freq_hz: int | None = None
        if version >= 2:
            raw_mode_states = payload.get("mode_states", [])
            if not isinstance(raw_mode_states, list):
                raise ValueError("RTSA mode states must be an array")
            mode_states = tuple(
                cls._from_profile_mapping(item)
                for item in raw_mode_states
                if isinstance(item, Mapping)
            )
            raw_shared_center = payload.get("shared_center_freq_hz")
            if raw_shared_center is not None:
                shared_center_freq_hz = int(raw_shared_center)
        if shared_center_freq_hz is None and "center_freq_hz" in config_values:
            shared_center_freq_hz = int(config_values["center_freq_hz"])
        profiled_analyzer_mode: AnalyzerMode | None = None
        raw_profiled_mode = payload.get("profiled_analyzer_mode")
        if raw_profiled_mode is not None:
            candidate = AnalyzerMode(str(raw_profiled_mode))
            if candidate == AnalyzerMode.TIME_ANALYZER:
                candidate = AnalyzerMode.HIGH_SPEED_TIME_ANALYZER
            if candidate in PROFILED_ANALYZER_MODES:
                profiled_analyzer_mode = candidate
        return cls(
            analyzer_mode=analyzer_mode,
            config_values=config_values,
            realtime_graph_view_mode=str(payload.get("realtime_graph_view_mode", "both")),
            persistence_enabled=bool(payload.get("persistence_enabled", False)),
            traces=traces,
            markers=markers,
            shared_center_freq_hz=shared_center_freq_hz,
            mode_states=mode_states,
            profiled_analyzer_mode=profiled_analyzer_mode,
        )


def capture_config_values(config: SpectrumConfig) -> dict[str, Any]:
    """Return the whitelisted user-facing values from ``SpectrumConfig``."""

    return {
        field_name: getattr(config, field_name)
        for field_name in PERSISTED_CONFIG_FIELDS
    }


def capture_mode_config_values(config: SpectrumConfig) -> dict[str, Any]:
    """Capture mode-local values; center frequency is deliberately shared."""

    return {
        field_name: getattr(config, field_name)
        for field_name in MODE_LOCAL_CONFIG_FIELDS
    }


def apply_config_values(config: SpectrumConfig, values: Mapping[str, Any]) -> None:
    """Apply persisted values while leaving device/runtime fields untouched."""

    for field_name in PERSISTED_CONFIG_FIELDS:
        if field_name in values:
            setattr(config, field_name, values[field_name])


def apply_mode_config_values(config: SpectrumConfig, values: Mapping[str, Any]) -> None:
    """Apply one mode profile without changing the shared center or device."""

    for field_name in MODE_LOCAL_CONFIG_FIELDS:
        if field_name in values:
            setattr(config, field_name, values[field_name])


def encode_session_state(state: RTSASessionState) -> str:
    return json.dumps(state.to_mapping(), ensure_ascii=False, separators=(",", ":"))


def decode_session_state(raw: str) -> RTSASessionState:
    payload = json.loads(str(raw))
    if not isinstance(payload, Mapping):
        raise ValueError("RTSA session state must be a JSON object")
    return RTSASessionState.from_mapping(payload)


def load_session_state(settings) -> RTSASessionState | None:
    """Load session state from a QSettings-compatible object."""

    raw = settings.value(RTSA_SESSION_KEY, "")
    if raw is None or not str(raw).strip():
        return None
    try:
        return decode_session_state(str(raw))
    except Exception as exc:
        print(f"[RTSA] Ignoring invalid saved session state: {exc}")
        return None


def save_session_state(settings, state: RTSASessionState) -> None:
    """Store one versioned JSON payload in QSettings."""

    settings.setValue(RTSA_SESSION_KEY, encode_session_state(state))
    settings.sync()


def clear_session_state(settings) -> None:
    settings.remove(RTSA_SESSION_KEY)
    settings.sync()
