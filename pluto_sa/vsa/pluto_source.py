"""Finite Pluto IQ acquisition source for the VSA application."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from pluto_sa.config.input_frontend import InputPowerCorrection
from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.modes.analyzer_mode import AnalyzerMode
from pluto_sa.sdr.pluto_receiver import PlutoReceiver
from pluto_sa.sdr.trigger import (
    AcquisitionMetadata,
    TriggerConfig,
    TriggerKind,
    TriggerRearmMode,
    TriggerRunMode,
)
from pluto_sa.sdr.trigger_acquisition import TriggerAcquisitionController
from pluto_sa.vsa.model import IQRecording
from pluto_sa.vsa.sources import IQSourceCapabilities, recording_from_acquisition


@dataclass(frozen=True)
class PlutoCaptureSettings:
    """R&S-style finite capture settings resolved for Pluto hardware."""

    center_frequency_hz: float = 2_441_000_000.0
    symbol_rate_hz: float = 1_000_000.0
    samples_per_symbol: int = 8
    capture_length_s: float = 0.003
    rf_bandwidth_hz: float = 8_000_000.0
    sdr_uri: str | None = None
    swap_iq: bool = False
    power_correction: InputPowerCorrection = InputPowerCorrection()

    def __post_init__(self) -> None:
        if not np.isfinite(self.center_frequency_hz) or self.center_frequency_hz <= 0.0:
            raise ValueError("center_frequency_hz must be positive")
        if not np.isfinite(self.symbol_rate_hz) or self.symbol_rate_hz <= 0.0:
            raise ValueError("symbol_rate_hz must be positive")
        if int(self.samples_per_symbol) not in {2, 4, 8, 16, 32, 64, 128}:
            raise ValueError("samples_per_symbol must be an R&S capture oversampling value")
        if not np.isfinite(self.capture_length_s) or self.capture_length_s <= 0.0:
            raise ValueError("capture_length_s must be positive")
        if not np.isfinite(self.rf_bandwidth_hz) or self.rf_bandwidth_hz <= 0.0:
            raise ValueError("rf_bandwidth_hz must be positive")

    @property
    def requested_sample_rate_hz(self) -> int:
        return int(round(self.symbol_rate_hz * int(self.samples_per_symbol)))

    @property
    def capture_samples(self) -> int:
        return max(1, int(round(self.capture_length_s * self.requested_sample_rate_hz)))

    @property
    def nominal_usable_bandwidth_hz(self) -> float:
        return min(0.8 * self.requested_sample_rate_hz, self.rf_bandwidth_hz)


class PlutoLiveSource:
    """Own a reusable Pluto receiver and produce immutable finite captures."""

    capabilities = IQSourceCapabilities(
        finite_capture=True,
        continuous_stream=True,
        hardware_trigger=False,
        writable_frontend=True,
    )

    def __init__(
        self,
        receiver_factory: Callable[[SpectrumConfig], PlutoReceiver] = PlutoReceiver,
    ) -> None:
        self._receiver_factory = receiver_factory
        self._receiver: PlutoReceiver | None = None
        self._active_config: SpectrumConfig | None = None

    def capture_single(self, settings: PlutoCaptureSettings) -> IQRecording:
        config = self._spectrum_config(settings)
        if self._receiver is None:
            self._receiver = self._receiver_factory(config)
        elif self._active_config != config:
            self._receiver.reconfigure(config)
        self._active_config = config

        block = self._receiver.capture_iq_block(
            settings.capture_samples,
            source="VSA Pluto Single",
            fresh=True,
        )
        actual_sample_rate_hz = self._receiver.get_current_sample_rate_hz()
        actual_rf_bandwidth_hz = self._receiver.get_current_rf_bandwidth_hz()
        usable_bandwidth_hz = min(
            0.8 * float(actual_sample_rate_hz),
            float(actual_rf_bandwidth_hz),
        )
        trigger = TriggerConfig(
            kind=TriggerKind.FREE_RUN,
            run_mode=TriggerRunMode.SINGLE,
            rearm_mode=TriggerRearmMode.STOP_ON_TRIGGER,
            pretrigger_samples=0,
            posttrigger_samples=block.sample_count - 1,
        )
        controller = TriggerAcquisitionController(
            trigger,
            AcquisitionMetadata(
                sample_rate_hz=float(actual_sample_rate_hz),
                center_freq_hz=float(settings.center_frequency_hz),
                rf_bandwidth_hz=usable_bandwidth_hz,
                gain_db=float(settings.power_correction.internal_gain_db),
                source="VSA Pluto Single",
            ),
        )
        records = controller.feed(block)
        if len(records) != 1:
            raise RuntimeError("Pluto single capture did not produce one complete record")
        recording = recording_from_acquisition(
            records[0],
            calibration_offset_db=settings.power_correction.calibration_offset_db,
            frequency_dependent_offset_db=(
                settings.power_correction.frequency_dependent_offset_db
            ),
            input_correction_db=settings.power_correction.input_correction_db,
            amplitude_calibrated=False,
        )
        iq = recording.iq
        if settings.swap_iq:
            iq = (iq.imag + 1j * iq.real).astype(np.complex64)
        return replace(
            recording,
            iq=iq,
            usable_bandwidth_hz=usable_bandwidth_hz,
            metadata={
                **dict(recording.metadata),
                "pluto_live_capture": True,
                "capture_oversampling": int(settings.samples_per_symbol),
                "capture_length_s": float(settings.capture_length_s),
                "requested_sample_rate_hz": settings.requested_sample_rate_hz,
                "actual_sample_rate_hz": int(actual_sample_rate_hz),
                "actual_rf_bandwidth_hz": int(actual_rf_bandwidth_hz),
                "internal_gain_db": float(settings.power_correction.internal_gain_db),
                "external_attenuation_db": float(
                    settings.power_correction.external_attenuation_db
                ),
                "external_gain_db": float(settings.power_correction.external_gain_db),
                "nominal_pluto_amplitude": True,
                "swap_iq": bool(settings.swap_iq),
            },
        )

    def close(self) -> None:
        if self._receiver is not None:
            self._receiver.close()
            self._receiver = None
            self._active_config = None

    @staticmethod
    def _spectrum_config(settings: PlutoCaptureSettings) -> SpectrumConfig:
        correction = settings.power_correction
        return SpectrumConfig(
            analyzer_mode=AnalyzerMode.HIGH_SPEED_TIME_ANALYZER,
            center_freq_hz=int(round(settings.center_frequency_hz)),
            calibration_offset_db=float(correction.calibration_offset_db),
            rx_gain_db=int(round(correction.internal_gain_db)),
            ext_att_db=float(correction.external_attenuation_db),
            ext_gain_db=float(correction.external_gain_db),
            sdr_uri=(settings.sdr_uri.strip() if settings.sdr_uri else None),
            time_analyzer_sample_rate_hz=settings.requested_sample_rate_hz,
            time_analyzer_rf_bandwidth_hz=int(round(settings.rf_bandwidth_hz)),
            time_analyzer_time_span_s=float(settings.capture_length_s),
        )
