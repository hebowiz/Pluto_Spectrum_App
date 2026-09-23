"""Wi-Fi capture results; logical packet coordinates and RF samples stay separate."""
from dataclasses import dataclass, field
import numpy as np
from pluto_protocol.model import PacketAnalysisResult
from .results import MeasurementResult


@dataclass(frozen=True)
class OFDMRegion:
    name: str
    modulation: str
    measured: np.ndarray
    reference: np.ndarray
    error: np.ndarray
    evm_rms_percent: float
    evm_peak_percent: float
    evm_per_subcarrier: np.ndarray
    evm_per_symbol: np.ndarray
    pilot_error_rms_percent: float


@dataclass(frozen=True)
class WiFiPacketResult:
    packet: PacketAnalysisResult
    start_sample: int
    stop_sample: int
    detection_confidence: float
    packet_power_dbm: float
    peak_power_dbm: float
    cfo_hz: float | None
    signal: OFDMRegion | None
    data: OFDMRegion | None
    channel: np.ndarray
    channel_subcarriers: np.ndarray
    common_phase_error_rad: np.ndarray
    symbol_clock_error_ppm: float | None = None
    measurements: tuple[MeasurementResult,...] = ()
    rf_details: dict = field(default_factory=dict)

    @property
    def integrity(self):
        values = {s.key: s.value for s in self.packet.summary}
        return {"preamble_detected": bool(values.get("preamble", False)),
                "lsig_complete": bool(values.get("lsig_complete", False)),
                "lsig_parity_valid": values.get("lsig_parity"),
                "data_complete": bool(values.get("data_complete", False)),
                "psdu_complete": bool(values.get("psdu_complete", False)),
                "fcs_valid": self.packet.integrity.crc_valid}

    @property
    def measurement_eligible(self):
        return self.data is not None and bool(self.packet.decode_context.get("phy_valid"))


@dataclass(frozen=True)
class WiFiCaptureResult:
    packets: tuple[WiFiPacketResult, ...]
    issues: tuple[str, ...] = ()
    measurement_statistics: dict = field(default_factory=dict)

    @property
    def counts(self):
        return {"detected": len(self.packets),
                "complete": sum(p.integrity["psdu_complete"] for p in self.packets),
                "measurement_eligible": sum(p.measurement_eligible for p in self.packets),
                "decode_success": sum(bool(p.packet.decode_context.get("phy_valid")) for p in self.packets),
                "fcs_valid": sum(p.packet.integrity.crc_valid is True for p in self.packets)}
