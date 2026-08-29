"""Persist user-facing RTSA session state between launches."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.modes.analyzer_mode import AnalyzerMode


RTSA_ORGANIZATION = "PlutoSpectrumApp"
RTSA_APPLICATION = "PlutoRTSA"
RTSA_DEVICE_KEY = "pluto_rx/selector"
RTSA_SESSION_KEY = "session/state_json"
RTSA_SESSION_SCHEMA_VERSION = 1

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


@dataclass(frozen=True)
class RTSASessionState:
    """Serializable user session without runtime measurement buffers."""

    analyzer_mode: AnalyzerMode
    config_values: dict[str, Any]
    realtime_graph_view_mode: str = "both"
    persistence_enabled: bool = False
    traces: tuple[dict[str, Any], ...] = ()
    markers: tuple[dict[str, Any], ...] = ()

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": RTSA_SESSION_SCHEMA_VERSION,
            "analyzer_mode": self.analyzer_mode.value,
            "config": dict(self.config_values),
            "realtime_graph_view_mode": str(self.realtime_graph_view_mode),
            "persistence_enabled": bool(self.persistence_enabled),
            "traces": [dict(item) for item in self.traces],
            "markers": [dict(item) for item in self.markers],
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RTSASessionState":
        version = int(payload.get("schema_version", 0))
        if version != RTSA_SESSION_SCHEMA_VERSION:
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
        return cls(
            analyzer_mode=analyzer_mode,
            config_values=config_values,
            realtime_graph_view_mode=str(payload.get("realtime_graph_view_mode", "both")),
            persistence_enabled=bool(payload.get("persistence_enabled", False)),
            traces=traces,
            markers=markers,
        )


def capture_config_values(config: SpectrumConfig) -> dict[str, Any]:
    """Return the whitelisted user-facing values from ``SpectrumConfig``."""

    return {
        field_name: getattr(config, field_name)
        for field_name in PERSISTED_CONFIG_FIELDS
    }


def apply_config_values(config: SpectrumConfig, values: Mapping[str, Any]) -> None:
    """Apply persisted values while leaving device/runtime fields untouched."""

    for field_name in PERSISTED_CONFIG_FIELDS:
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
