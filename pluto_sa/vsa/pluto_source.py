"""Finite Pluto IQ acquisition source for the VSA application."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Callable

import numpy as np

from pluto_sa.config.input_frontend import InputPowerCorrection
from pluto_sa.config.spectrum_config import SpectrumConfig
from pluto_sa.modes.analyzer_mode import AnalyzerMode
from pluto_sa.sdr.pluto_receiver import PlutoReceiver
from pluto_sa.sdr.continuous_acquisition import (
    ContinuousIQAcquisition,
    resolve_record_stream_block_samples,
)
from pluto_sa.sdr.trigger import (
    AcquisitionMetadata,
    TriggerConfig,
    TriggerKind,
    TriggerRearmMode,
    TriggerRunMode,
    TriggerSlope,
    power_trigger_display_dbm_to_dbfs,
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
    lo_offset_hz: float = 0.0
    analysis_bandwidth_hz: float | None = None
    sdr_uri: str | None = None
    swap_iq: bool = False
    power_correction: InputPowerCorrection = InputPowerCorrection()
    trigger_source: TriggerKind = TriggerKind.FREE_RUN
    trigger_level_dbm: float = -20.0
    trigger_slope: TriggerSlope = TriggerSlope.RISING
    trigger_offset_s: float = 0.0
    trigger_hysteresis_db: float = 3.0

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
        if not np.isfinite(self.lo_offset_hz):
            raise ValueError("lo_offset_hz must be finite")
        if self.analysis_bandwidth_hz is not None and (
            not np.isfinite(self.analysis_bandwidth_hz)
            or self.analysis_bandwidth_hz <= 0.0
        ):
            raise ValueError("analysis_bandwidth_hz must be positive when set")
        if self.trigger_source not in {TriggerKind.FREE_RUN, TriggerKind.POWER_LEVEL}:
            raise ValueError("VSA Pluto capture supports Free Run or I/Q Power trigger")
        if not np.isfinite(self.trigger_level_dbm):
            raise ValueError("trigger_level_dbm must be finite")
        if not np.isfinite(self.trigger_offset_s):
            raise ValueError("trigger_offset_s must be finite")
        if self.trigger_offset_s < -self.capture_length_s:
            raise ValueError("negative Trigger Offset cannot exceed Capture Length")
        if not np.isfinite(self.trigger_hysteresis_db) or self.trigger_hysteresis_db < 0.0:
            raise ValueError("trigger_hysteresis_db must be finite and non-negative")

    @property
    def requested_sample_rate_hz(self) -> int:
        return int(round(self.symbol_rate_hz * int(self.samples_per_symbol)))

    @property
    def capture_samples(self) -> int:
        return max(1, int(round(self.capture_length_s * self.requested_sample_rate_hz)))

    @property
    def nominal_usable_bandwidth_hz(self) -> float:
        return min(0.8 * self.requested_sample_rate_hz, self.rf_bandwidth_hz)

    @property
    def hardware_lo_frequency_hz(self) -> float:
        return float(self.center_frequency_hz) + float(self.lo_offset_hz)

    @property
    def trigger_offset_samples(self) -> int:
        return int(round(self.trigger_offset_s * self.requested_sample_rate_hz))

    @property
    def default_trigger_prestore_samples(self) -> int:
        """Return the zero-offset safety history retained before a trigger.

        A software power trigger declares an edge only after an above-threshold
        sample arrives.  Starting the record at that sample clips the leading
        ramp or preamble of the first burst, although later bursts in the same
        capture remain complete.  An explicitly non-zero Trigger Offset still
        remains authoritative.
        """

        return min(
            max(0, self.capture_samples - 1),
            16 * int(self.samples_per_symbol),
        )


class CaptureCancelledError(RuntimeError):
    """Raised when the operator stops a pending Pluto acquisition."""


class PlutoLiveSource:
    """Own a reusable Pluto receiver and produce immutable finite captures."""

    capabilities = IQSourceCapabilities(
        finite_capture=True,
        continuous_stream=True,
        hardware_trigger=False,
        writable_frontend=True,
    )
    _STREAM_CAPTURE_BLOCK_SAMPLES = 65_536
    _STREAM_BUFFER_BLOCKS = 8

    def __init__(
        self,
        receiver_factory: Callable[[SpectrumConfig], PlutoReceiver] | None = None,
    ) -> None:
        self._receiver_factory = receiver_factory or (
            lambda config: PlutoReceiver(config, owner_application="Pluto VSA")
        )
        self._receiver: PlutoReceiver | None = None
        self._acquisition: ContinuousIQAcquisition | None = None
        self._active_config: SpectrumConfig | None = None
        self._actual_sample_rate_hz: int | None = None
        self._actual_rf_bandwidth_hz: int | None = None

    def is_stream_ready(self, settings: PlutoCaptureSettings) -> bool:
        """Return whether matching Pluto IQ is already arriving in the ring."""
        acquisition = self._acquisition
        if (
            self._receiver is None
            or acquisition is None
            or self._active_config != self._spectrum_config(settings)
            or not acquisition.is_running
        ):
            return False
        plan = acquisition.plan
        return bool(
            plan is not None
            and plan.source == "vsa_continuous"
            and plan.max_blocks is None
            and plan.block_size
            == resolve_record_stream_block_samples(
                settings.capture_samples,
                base_block_samples=self._STREAM_CAPTURE_BLOCK_SAMPLES,
            )
        )

    def capture_single(
        self,
        settings: PlutoCaptureSettings,
        *,
        cancelled: Callable[[], bool] | None = None,
        armed: Callable[[], None] | None = None,
        fresh: bool = True,
        prefer_buffered: bool = False,
    ) -> IQRecording:
        cancelled = cancelled or (lambda: False)
        if cancelled():
            raise CaptureCancelledError("Pluto capture cancelled")
        prepare_started_at = time.perf_counter()
        producer_reused = self.is_stream_ready(settings)
        buffered_start = bool(prefer_buffered and producer_reused)
        config = self._spectrum_config(settings)
        if self._receiver is None:
            self._receiver = self._receiver_factory(config)
            self._acquisition = ContinuousIQAcquisition(self._receiver)
            self._actual_sample_rate_hz = None
            self._actual_rf_bandwidth_hz = None
        elif self._active_config != config:
            assert self._acquisition is not None
            self._acquisition.stop()
            self._receiver.reconfigure(config)
            self._actual_sample_rate_hz = None
            self._actual_rf_bandwidth_hz = None
        self._active_config = config

        # Hardware getters take the receiver I/Q lock.  Once the continuous
        # producer is running, that lock is held across blocking sdr.rx() calls
        # and can starve this capture worker for multiple refills.  The values
        # cannot change without the reconfiguration path above, so cache them
        # while the producer is stopped/new and reuse them for every re-arm.
        if self._actual_sample_rate_hz is None:
            self._actual_sample_rate_hz = int(
                self._receiver.get_current_sample_rate_hz()
            )
        if self._actual_rf_bandwidth_hz is None:
            self._actual_rf_bandwidth_hz = int(
                self._receiver.get_current_rf_bandwidth_hz()
            )
        actual_sample_rate_hz = self._actual_sample_rate_hz
        actual_rf_bandwidth_hz = self._actual_rf_bandwidth_hz
        prepare_ms = (time.perf_counter() - prepare_started_at) * 1e3
        usable_bandwidth_hz = min(
            0.8 * float(actual_sample_rate_hz),
            float(actual_rf_bandwidth_hz),
        )
        capture_source = (
            "VSA Pluto I/Q Power Trigger"
            if settings.trigger_source is TriggerKind.POWER_LEVEL
            else "VSA Pluto Single"
        )
        metadata = AcquisitionMetadata(
            sample_rate_hz=float(actual_sample_rate_hz),
            center_freq_hz=float(settings.hardware_lo_frequency_hz),
            rf_bandwidth_hz=usable_bandwidth_hz,
            gain_db=float(settings.power_correction.internal_gain_db),
            source=capture_source,
        )
        (
            record,
            output_start_offset,
            stream_block_samples,
            stream_diagnostics,
        ) = self._capture_record(
            settings,
            metadata,
            cancelled=cancelled,
            armed=armed,
            fresh=bool(fresh),
            prefer_buffered=buffered_start,
        )
        recording = recording_from_acquisition(
            record,
            calibration_offset_db=settings.power_correction.calibration_offset_db,
            frequency_dependent_offset_db=(
                settings.power_correction.frequency_dependent_offset_db
            ),
            input_correction_db=settings.power_correction.input_correction_db,
            amplitude_calibrated=False,
        )
        trigger_sample_index = recording.trigger_sample_index
        if output_start_offset or recording.sample_count != settings.capture_samples:
            output_stop = output_start_offset + settings.capture_samples
            trigger_sample_index = (
                recording.trigger_sample_index
                if (
                    recording.trigger_sample_index is not None
                    and recording.start_sample_index + output_start_offset
                    <= recording.trigger_sample_index
                    < recording.start_sample_index + output_stop
                )
                else None
            )
            recording = replace(
                recording,
                iq=recording.iq[output_start_offset:output_stop],
                start_sample_index=recording.start_sample_index + output_start_offset,
                trigger_sample_index=trigger_sample_index,
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
                "requested_center_frequency_hz": float(
                    settings.center_frequency_hz
                ),
                "hardware_lo_frequency_hz": float(
                    settings.hardware_lo_frequency_hz
                ),
                "lo_offset_hz": float(settings.lo_offset_hz),
                "experimental_lo_offset": bool(settings.lo_offset_hz),
                "requested_analysis_bandwidth_hz": (
                    None
                    if settings.analysis_bandwidth_hz is None
                    else float(settings.analysis_bandwidth_hz)
                ),
                "actual_sample_rate_hz": int(actual_sample_rate_hz),
                "actual_rf_bandwidth_hz": int(actual_rf_bandwidth_hz),
                "internal_gain_db": float(settings.power_correction.internal_gain_db),
                "external_attenuation_db": float(
                    settings.power_correction.external_attenuation_db
                ),
                "external_gain_db": float(settings.power_correction.external_gain_db),
                "nominal_pluto_amplitude": True,
                "swap_iq": bool(settings.swap_iq),
                "acquisition_trigger_source": settings.trigger_source.value,
                "acquisition_trigger_level_dbm": float(settings.trigger_level_dbm),
                "acquisition_trigger_slope": settings.trigger_slope.value,
                "acquisition_trigger_offset_s": float(settings.trigger_offset_s),
                "acquisition_default_prestore_samples": int(
                    settings.default_trigger_prestore_samples
                    if (
                        settings.trigger_source is TriggerKind.POWER_LEVEL
                        and settings.trigger_offset_samples == 0
                    )
                    else 0
                ),
                "acquisition_trigger_sample_index": int(record.trigger_sample_index),
                "continuous_rx_block_samples": int(stream_block_samples),
                "continuous_rx_buffered_start": buffered_start,
                "continuous_rx_producer_reused": bool(producer_reused),
                "continuous_rx_prepare_ms": float(prepare_ms),
                "trigger_offset_in_rx_block": int(record.trigger.offset_in_block),
                "trigger_samples_to_rx_block_end": int(
                    stream_block_samples - record.trigger.offset_in_block
                ),
                **stream_diagnostics,
            },
        )

    def _capture_record(
        self,
        settings: PlutoCaptureSettings,
        metadata: AcquisitionMetadata,
        *,
        cancelled: Callable[[], bool],
        armed: Callable[[], None] | None,
        fresh: bool,
        prefer_buffered: bool,
    ):
        assert self._receiver is not None
        assert self._acquisition is not None
        if settings.trigger_source is TriggerKind.FREE_RUN:
            trigger = TriggerConfig(
                kind=TriggerKind.FREE_RUN,
                run_mode=TriggerRunMode.SINGLE,
                rearm_mode=TriggerRearmMode.STOP_ON_TRIGGER,
                pretrigger_samples=0,
                posttrigger_samples=settings.capture_samples - 1,
            )
            output_start_offset = 0
        else:
            offset_samples = int(round(settings.trigger_offset_s * metadata.sample_rate_hz))
            default_prestore_samples = (
                settings.default_trigger_prestore_samples if offset_samples == 0 else 0
            )
            output_start_from_trigger = offset_samples - default_prestore_samples
            pretrigger_samples = max(0, -output_start_from_trigger)
            posttrigger_samples = max(
                0,
                settings.capture_samples + output_start_from_trigger - 1,
            )
            correction = settings.power_correction
            trigger = TriggerConfig(
                kind=TriggerKind.POWER_LEVEL,
                run_mode=TriggerRunMode.SINGLE,
                rearm_mode=TriggerRearmMode.STOP_ON_TRIGGER,
                slope=settings.trigger_slope,
                level_dbfs=power_trigger_display_dbm_to_dbfs(
                    settings.trigger_level_dbm,
                    calibration_offset_db=correction.calibration_offset_db,
                    frequency_dependent_offset_db=(
                        correction.frequency_dependent_offset_db
                    ),
                    input_correction_db=correction.input_correction_db,
                ),
                hysteresis_db=settings.trigger_hysteresis_db,
                pretrigger_samples=pretrigger_samples,
                posttrigger_samples=posttrigger_samples,
            )
            output_start_offset = pretrigger_samples + output_start_from_trigger

        controller = TriggerAcquisitionController(trigger, metadata)
        stream_block_samples = resolve_record_stream_block_samples(
            settings.capture_samples,
            base_block_samples=self._STREAM_CAPTURE_BLOCK_SAMPLES,
        )
        cursor_setup_started_at = time.perf_counter()
        cursor = self._acquisition.start(
            block_size=stream_block_samples,
            source="vsa_continuous",
            fresh=fresh,
            cursor_start="newest" if prefer_buffered else "latest",
        )
        cursor_setup_ms = (time.perf_counter() - cursor_setup_started_at) * 1e3
        first_block_wait_started_at = time.perf_counter()
        first_block_wait_ms: float | None = None
        armed_reported = False
        observed_blocks = 0
        max_refill_ratio = 0.0
        while True:
            if cancelled():
                raise CaptureCancelledError("Pluto capture cancelled")
            read_result = self._acquisition.read(cursor, max_blocks=32)
            cursor = read_result.cursor
            if read_result.overrun:
                raise RuntimeError(
                    "VSA Pluto stream consumer overrun: "
                    f"missed {read_result.missed_blocks} IQ block(s)"
                )
            if not read_result.blocks:
                time.sleep(0.001)
                continue
            if first_block_wait_ms is None:
                first_block_wait_ms = (
                    time.perf_counter() - first_block_wait_started_at
                ) * 1e3
            if not armed_reported:
                # A running Python thread is not proof that libiio has queued
                # and filled its first DMA buffer.  Only expose the armed state
                # after the first actual IQ block arrives.
                armed_reported = True
                if armed is not None:
                    armed()
            for block in read_result.blocks:
                if cancelled():
                    raise CaptureCancelledError("Pluto capture cancelled")
                observed_blocks += 1
                expected_refill_s = block.sample_count / metadata.sample_rate_hz
                if expected_refill_s > 0.0:
                    max_refill_ratio = max(
                        max_refill_ratio,
                        float(block.capture_elapsed_s) / expected_refill_s,
                    )
                # The hardware producer has one stable source name so Free
                # Run and Power Trigger can share it without restarting.
                # Records retain the user-facing acquisition source.
                records = controller.feed(replace(block, source=metadata.source))
                if records:
                    return (
                        records[0],
                        output_start_offset,
                        stream_block_samples,
                        {
                            "continuous_rx_observed_blocks": int(observed_blocks),
                            "continuous_rx_max_refill_ratio": float(max_refill_ratio),
                            "continuous_rx_cursor_setup_ms": float(cursor_setup_ms),
                            "continuous_rx_first_block_wait_ms": float(
                                first_block_wait_ms or 0.0
                            ),
                            "continuous_rx_kernel_buffers_requested": int(
                                getattr(self._receiver, "rx_kernel_buffers_requested", 0)
                            ),
                            "continuous_rx_kernel_buffers_applied": (
                                None
                                if getattr(
                                    self._receiver,
                                    "rx_kernel_buffers_applied",
                                    None,
                                )
                                is None
                                else int(self._receiver.rx_kernel_buffers_applied)
                            ),
                        },
                    )

    def close(self) -> None:
        if self._receiver is not None:
            if self._acquisition is not None:
                self._acquisition.stop()
            self._receiver.close()
            self._receiver = None
            self._acquisition = None
            self._active_config = None
            self._actual_sample_rate_hz = None
            self._actual_rf_bandwidth_hz = None

    def stop_stream(self) -> None:
        """Stop background IQ production without closing the Pluto receiver."""

        if self._acquisition is not None:
            self._acquisition.stop()

    @staticmethod
    def _spectrum_config(settings: PlutoCaptureSettings) -> SpectrumConfig:
        correction = settings.power_correction
        return SpectrumConfig(
            analyzer_mode=AnalyzerMode.HIGH_SPEED_TIME_ANALYZER,
            center_freq_hz=int(round(settings.hardware_lo_frequency_hz)),
            calibration_offset_db=float(correction.calibration_offset_db),
            rx_gain_db=int(round(correction.internal_gain_db)),
            ext_att_db=float(correction.external_attenuation_db),
            ext_gain_db=float(correction.external_gain_db),
            sdr_uri=(settings.sdr_uri.strip() if settings.sdr_uri else None),
            time_analyzer_sample_rate_hz=settings.requested_sample_rate_hz,
            time_analyzer_rf_bandwidth_hz=int(round(settings.rf_bandwidth_hz)),
            time_analyzer_time_span_s=float(settings.capture_length_s),
            capture_buffer_blocks=PlutoLiveSource._STREAM_BUFFER_BLOCKS,
        )
