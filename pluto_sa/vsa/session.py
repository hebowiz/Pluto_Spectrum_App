"""VSA session state independent of the Qt application shell."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter

import numpy as np

from pluto_sa.vsa.analysis import VSAAnalyzer, capture_power_traces
from pluto_sa.vsa.channel import extract_analysis_channel
from pluto_sa.vsa.dc import apply_robust_dc_removal
from pluto_sa.vsa.model import (
    IQRecording,
    ModulationFamily,
    SignalDescription,
    VSAAnalysisResult,
    VSASettings,
)
from pluto_sa.vsa.pattern import (
    carrier_correct_recording,
    DemodulationSettings,
    IQPowerTriggerSettings,
    PatternAnalyzer,
    PatternSearchResult,
    PatternSearchSettings,
    ResultRangeSettings,
    SynchronizationSource,
)


@dataclass
class VSASession:
    name: str = "VSA 1"
    recording: IQRecording | None = None
    signal: SignalDescription | None = None
    settings: VSASettings = field(default_factory=VSASettings)
    result: VSAAnalysisResult | None = None
    pattern_search: PatternSearchSettings | None = None
    iq_power_trigger: IQPowerTriggerSettings = field(
        default_factory=IQPowerTriggerSettings
    )
    result_range: ResultRangeSettings = field(default_factory=ResultRangeSettings)
    demodulation: DemodulationSettings = field(default_factory=DemodulationSettings)
    pattern_result: PatternSearchResult | None = None
    pattern_range_result: VSAAnalysisResult | None = None
    carrier_corrected_result: VSAAnalysisResult | None = None
    carrier_corrected_pattern_range_result: VSAAnalysisResult | None = None
    pattern_error: str | None = None
    analysis_timings_ms: dict[str, float] = field(default_factory=dict)
    capture_time_s: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    capture_power_dbfs: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    capture_power_dbm: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    revision: int = 0
    _analyzer: VSAAnalyzer = field(default_factory=VSAAnalyzer, repr=False)
    _pattern_analyzer: PatternAnalyzer = field(default_factory=PatternAnalyzer, repr=False)
    _prepared_recording: IQRecording | None = field(default=None, repr=False)
    _prepared_settings: VSASettings | None = field(default=None, repr=False)

    def analysis_snapshot(self) -> "VSASession":
        """Return an isolated analysis job sharing only immutable input data."""
        return VSASession(
            name=self.name,
            recording=self.recording,
            signal=self.signal,
            settings=self.settings,
            pattern_search=self.pattern_search,
            iq_power_trigger=self.iq_power_trigger,
            result_range=self.result_range,
            demodulation=self.demodulation,
            capture_time_s=self.capture_time_s,
            capture_power_dbfs=self.capture_power_dbfs,
            capture_power_dbm=self.capture_power_dbm,
            revision=self.revision,
            _prepared_recording=self._prepared_recording,
            _prepared_settings=self._prepared_settings,
        )

    def adopt_analysis_results(self, completed: "VSASession") -> None:
        """Publish results produced by a matching background snapshot."""
        if completed.revision != self.revision:
            raise ValueError("analysis snapshot revision is stale")
        self.result = completed.result
        self.pattern_result = completed.pattern_result
        self.pattern_range_result = completed.pattern_range_result
        self.carrier_corrected_result = completed.carrier_corrected_result
        self.carrier_corrected_pattern_range_result = (
            completed.carrier_corrected_pattern_range_result
        )
        self.pattern_error = completed.pattern_error
        self.analysis_timings_ms = dict(completed.analysis_timings_ms)
        self.capture_time_s = completed.capture_time_s
        self.capture_power_dbfs = completed.capture_power_dbfs
        self.capture_power_dbm = completed.capture_power_dbm

    def _invalidate_results(self) -> None:
        self.result = None
        self.pattern_result = None
        self.pattern_range_result = None
        self.carrier_corrected_result = None
        self.carrier_corrected_pattern_range_result = None
        self.pattern_error = None
        self.analysis_timings_ms = {}
        self.capture_time_s = np.empty(0, dtype=np.float64)
        self.capture_power_dbfs = np.empty(0, dtype=np.float64)
        self.capture_power_dbm = np.empty(0, dtype=np.float64)
        self._prepared_recording = None
        self._prepared_settings = None

    def set_recording(self, recording: IQRecording) -> None:
        self.recording = recording
        self._invalidate_results()
        self.revision += 1

    def set_signal(self, signal: SignalDescription) -> None:
        self.signal = signal
        self._invalidate_results()
        self.revision += 1

    def update_settings(self, **changes: object) -> None:
        self.settings = replace(self.settings, **changes)
        self._invalidate_results()
        self.revision += 1

    def configure_pattern_analysis(
        self,
        search: PatternSearchSettings | None,
        result_range: ResultRangeSettings | None = None,
        demodulation: DemodulationSettings | None = None,
        iq_power_trigger: IQPowerTriggerSettings | None = None,
    ) -> None:
        self.pattern_search = search
        if iq_power_trigger is not None:
            self.iq_power_trigger = iq_power_trigger
        elif search is not None:
            self.iq_power_trigger = search.iq_power_trigger
        if result_range is not None:
            self.result_range = result_range
        if demodulation is not None:
            self.demodulation = demodulation
        self._invalidate_results()
        self.revision += 1

    def _prepare_analysis_recording(self) -> tuple[IQRecording, VSASettings]:
        """Apply source-independent channel and DC processing exactly once."""
        if self.recording is None:
            raise RuntimeError("no IQ recording is loaded")
        if (
            self._prepared_recording is not None
            and self._prepared_settings is not None
        ):
            return self._prepared_recording, self._prepared_settings
        prepared = self.recording
        if self.settings.analysis_bandwidth_hz is not None:
            prepared = extract_analysis_channel(
                self.recording,
                center_frequency_hz=(
                    self.recording.center_frequency_hz
                    if self.settings.analysis_center_frequency_hz is None
                    else self.settings.analysis_center_frequency_hz
                ),
                bandwidth_hz=self.settings.analysis_bandwidth_hz,
            )
        remove_dc = self.settings.remove_dc and bool(
            prepared.metadata.get("dc_removal_recommended", False)
        ) and not bool(prepared.metadata.get("experimental_lo_offset", False))
        if remove_dc:
            prepared = apply_robust_dc_removal(prepared)
        prepared_settings = replace(
            self.settings,
            remove_dc=False,
            analysis_center_frequency_hz=None,
            analysis_bandwidth_hz=None,
        )
        self._prepared_recording = prepared
        self._prepared_settings = prepared_settings
        return prepared, prepared_settings

    def _publish_demodulation_result(
        self,
        pattern_recording: IQRecording,
        prepared_settings: VSASettings,
    ) -> None:
        """Build the common corrected/full/range products for one sync result."""
        if self.pattern_result is None:
            return
        stage_started = perf_counter()
        corrected_recording = carrier_correct_recording(
            pattern_recording,
            self.pattern_result,
            compensate_drift=self.demodulation.compensate_carrier_frequency_drift,
        )
        selected = pattern_recording.iq[
            self.pattern_result.result_start_sample : self.pattern_result.result_stop_sample
        ]
        range_recording = replace(
            pattern_recording,
            iq=selected,
            start_sample_index=(
                pattern_recording.start_sample_index
                + self.pattern_result.result_start_sample
            ),
            trigger_sample_index=None,
            source=f"{pattern_recording.source} | Result Range",
        )
        corrected_selected = corrected_recording.iq[
            self.pattern_result.result_start_sample : self.pattern_result.result_stop_sample
        ]
        corrected_range_recording = replace(
            corrected_recording,
            iq=corrected_selected,
            start_sample_index=(
                corrected_recording.start_sample_index
                + self.pattern_result.result_start_sample
            ),
            trigger_sample_index=None,
            source=f"{corrected_recording.source} | Result Range",
        )
        self.analysis_timings_ms["post_prepare"] = (
            perf_counter() - stage_started
        ) * 1e3
        stage_started = perf_counter()
        self.carrier_corrected_result = self._analyzer.analyze(
            corrected_recording, self.signal, prepared_settings
        )
        self.pattern_range_result = self._analyzer.analyze(
            range_recording, self.signal, prepared_settings
        )
        self.carrier_corrected_pattern_range_result = self._analyzer.analyze(
            corrected_range_recording, self.signal, prepared_settings
        )
        self.analysis_timings_ms["post_analysis"] = (
            perf_counter() - stage_started
        ) * 1e3

    def _publish_selected_range_result(
        self,
        pattern_recording: IQRecording,
        prepared_settings: VSASettings,
    ) -> None:
        """Build only the corrected selected-range product needed by metrics."""

        if self.pattern_result is None:
            return
        corrected_recording = carrier_correct_recording(
            pattern_recording,
            self.pattern_result,
            compensate_drift=self.demodulation.compensate_carrier_frequency_drift,
        )
        corrected_selected = corrected_recording.iq[
            self.pattern_result.result_start_sample : self.pattern_result.result_stop_sample
        ]
        corrected_range_recording = replace(
            corrected_recording,
            iq=corrected_selected,
            start_sample_index=(
                corrected_recording.start_sample_index
                + self.pattern_result.result_start_sample
            ),
            trigger_sample_index=None,
            source=f"{corrected_recording.source} | Result Range",
        )
        stage_started = perf_counter()
        self.carrier_corrected_pattern_range_result = self._analyzer.analyze(
            corrected_range_recording, self.signal, prepared_settings
        )
        # Keep analyze()'s non-optional return contract for deferred sessions.
        self.result = self.carrier_corrected_pattern_range_result
        self.analysis_timings_ms["selected_range_analysis"] = (
            perf_counter() - stage_started
        ) * 1e3

    def generate_display_products(self) -> None:
        """Materialize corrected/range plot products after deferred analysis."""

        if self.pattern_result is None or self.recording is None or self.signal is None:
            return
        if (
            self.carrier_corrected_result is not None
            and self.pattern_range_result is not None
            and self.carrier_corrected_pattern_range_result is not None
        ):
            return
        analysis_recording, prepared_settings = self._prepare_analysis_recording()
        self._publish_demodulation_result(analysis_recording, prepared_settings)

    def analyze_base_only(self) -> VSAAnalysisResult:
        """Analyze IQ without invoking automatic pattern/data synchronization."""

        if self.recording is None:
            raise RuntimeError("no IQ recording is loaded")
        if self.signal is None:
            raise RuntimeError("no signal description is configured")
        total_started = perf_counter()
        if self.capture_time_s.size != self.recording.sample_count:
            (
                self.capture_time_s,
                self.capture_power_dbfs,
                self.capture_power_dbm,
            ) = capture_power_traces(self.recording)
        stage_started = perf_counter()
        analysis_recording, prepared_settings = self._prepare_analysis_recording()
        self.analysis_timings_ms = {
            "preprocess": (perf_counter() - stage_started) * 1e3,
        }
        stage_started = perf_counter()
        self.result = self._analyzer.analyze(
            analysis_recording, self.signal, prepared_settings
        )
        self.analysis_timings_ms["base_analysis"] = (
            perf_counter() - stage_started
        ) * 1e3
        self.analysis_timings_ms["total_dsp"] = (
            perf_counter() - total_started
        ) * 1e3
        self.pattern_result = None
        self.pattern_range_result = None
        self.carrier_corrected_result = None
        self.carrier_corrected_pattern_range_result = None
        self.pattern_error = None
        return self.result

    def analyze(self, *, generate_display_products: bool = True) -> VSAAnalysisResult:
        if self.recording is None:
            raise RuntimeError("no IQ recording is loaded")
        if self.signal is None:
            raise RuntimeError("no signal description is configured")
        total_started = perf_counter()
        if self.capture_time_s.size != self.recording.sample_count:
            (
                self.capture_time_s,
                self.capture_power_dbfs,
                self.capture_power_dbm,
            ) = capture_power_traces(self.recording)
        stage_started = perf_counter()
        analysis_recording, prepared_settings = self._prepare_analysis_recording()
        self.analysis_timings_ms = {
            "preprocess": (perf_counter() - stage_started) * 1e3,
        }
        if generate_display_products:
            stage_started = perf_counter()
            self.result = self._analyzer.analyze(
                analysis_recording,
                self.signal,
                prepared_settings,
            )
            self.analysis_timings_ms["base_analysis"] = (
                perf_counter() - stage_started
            ) * 1e3
        else:
            self.result = None
            self.analysis_timings_ms["base_analysis"] = 0.0
        self.pattern_result = None
        self.pattern_range_result = None
        self.carrier_corrected_result = None
        self.carrier_corrected_pattern_range_result = None
        self.pattern_error = None
        detected_data_without_pattern = (
            self.pattern_search is None
            and self.signal.modulation.family.uses_iq_constellation
            and self.demodulation.coarse_synchronization
            in {SynchronizationSource.AUTO, SynchronizationSource.DETECTED_DATA}
        )
        if self.pattern_search is not None or detected_data_without_pattern:
            pattern_recording = analysis_recording
            try:
                stage_started = perf_counter()
                if (
                    self.demodulation.coarse_synchronization
                    is SynchronizationSource.DETECTED_DATA
                    or self.pattern_search is None
                ):
                    self.pattern_result = self._pattern_analyzer.detect_data(
                        pattern_recording,
                        self.signal,
                        self.pattern_search,
                        self.result_range,
                        self.demodulation,
                        self.iq_power_trigger,
                    )
                else:
                    self.pattern_result = self._pattern_analyzer.search(
                        pattern_recording,
                        self.signal,
                        self.pattern_search,
                        self.result_range,
                        self.demodulation,
                    )
                self.analysis_timings_ms["pattern_search"] = (
                    perf_counter() - stage_started
                ) * 1e3
                if (
                    self.pattern_search is not None
                    and self.pattern_search.meas_only_if_pattern_symbols_correct
                    and self.pattern_result.pattern_symbol_errors > 0
                    and self.pattern_result.metadata.get("pattern_match_valid", True)
                ):
                    raise ValueError(
                        "pattern waveform matched but Pattern Symbols Correct is false"
                    )
                if generate_display_products:
                    self._publish_demodulation_result(
                        pattern_recording, prepared_settings
                    )
                else:
                    self._publish_selected_range_result(
                        pattern_recording, prepared_settings
                    )
            except ValueError as error:
                self.pattern_error = str(error)
                allow_detected_fallback = (
                    self.pattern_search is not None
                    and self.signal.modulation.family.uses_iq_constellation
                    and self.demodulation.coarse_synchronization
                    is SynchronizationSource.AUTO
                )
                if allow_detected_fallback:
                    try:
                        stage_started = perf_counter()
                        self.pattern_result = self._pattern_analyzer.detect_data(
                            pattern_recording,
                            self.signal,
                            self.pattern_search,
                            self.result_range,
                            self.demodulation,
                            self.iq_power_trigger,
                        )
                        self.analysis_timings_ms["detected_data_sync"] = (
                            perf_counter() - stage_started
                        ) * 1e3
                        if generate_display_products:
                            self._publish_demodulation_result(
                                pattern_recording, prepared_settings
                            )
                        else:
                            self._publish_selected_range_result(
                                pattern_recording, prepared_settings
                            )
                    except ValueError as fallback_error:
                        self.pattern_error = f"{error}; {fallback_error}"
                        self.pattern_result = None
                if (
                    self.pattern_result is None
                    and self.pattern_search is not None
                    and self.pattern_search.meas_only_if_pattern_symbols_correct
                ):
                    self.pattern_result = None
                    self.pattern_range_result = None
                    self.carrier_corrected_result = None
                    self.carrier_corrected_pattern_range_result = None
                    self.analysis_timings_ms["total_dsp"] = (
                        perf_counter() - total_started
                    ) * 1e3
                    raise
        self.analysis_timings_ms["total_dsp"] = (
            perf_counter() - total_started
        ) * 1e3
        if self.result is None:
            raise RuntimeError("VSA analysis did not produce a result")
        return self.result
