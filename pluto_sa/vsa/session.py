"""VSA session state independent of the Qt application shell."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from pluto_sa.vsa.analysis import VSAAnalyzer
from pluto_sa.vsa.channel import extract_analysis_channel
from pluto_sa.vsa.model import IQRecording, SignalDescription, VSAAnalysisResult, VSASettings
from pluto_sa.vsa.pattern import (
    DemodulationSettings,
    PatternAnalyzer,
    PatternSearchResult,
    PatternSearchSettings,
    ResultRangeSettings,
)


@dataclass
class VSASession:
    name: str = "VSA 1"
    recording: IQRecording | None = None
    signal: SignalDescription | None = None
    settings: VSASettings = field(default_factory=VSASettings)
    result: VSAAnalysisResult | None = None
    pattern_search: PatternSearchSettings | None = None
    result_range: ResultRangeSettings = field(default_factory=ResultRangeSettings)
    demodulation: DemodulationSettings = field(default_factory=DemodulationSettings)
    pattern_result: PatternSearchResult | None = None
    pattern_range_result: VSAAnalysisResult | None = None
    pattern_error: str | None = None
    revision: int = 0
    _analyzer: VSAAnalyzer = field(default_factory=VSAAnalyzer, repr=False)
    _pattern_analyzer: PatternAnalyzer = field(default_factory=PatternAnalyzer, repr=False)

    def set_recording(self, recording: IQRecording) -> None:
        self.recording = recording
        self.result = None
        self.pattern_result = None
        self.pattern_range_result = None
        self.pattern_error = None
        self.revision += 1

    def set_signal(self, signal: SignalDescription) -> None:
        self.signal = signal
        self.result = None
        self.pattern_result = None
        self.pattern_range_result = None
        self.pattern_error = None
        self.revision += 1

    def update_settings(self, **changes: object) -> None:
        self.settings = replace(self.settings, **changes)
        self.result = None
        self.pattern_result = None
        self.pattern_range_result = None
        self.pattern_error = None
        self.revision += 1

    def configure_pattern_analysis(
        self,
        search: PatternSearchSettings | None,
        result_range: ResultRangeSettings | None = None,
        demodulation: DemodulationSettings | None = None,
    ) -> None:
        self.pattern_search = search
        if result_range is not None:
            self.result_range = result_range
        if demodulation is not None:
            self.demodulation = demodulation
        self.pattern_result = None
        self.pattern_range_result = None
        self.pattern_error = None
        self.revision += 1

    def analyze(self) -> VSAAnalysisResult:
        if self.recording is None:
            raise RuntimeError("no IQ recording is loaded")
        if self.signal is None:
            raise RuntimeError("no signal description is configured")
        self.result = self._analyzer.analyze(self.recording, self.signal, self.settings)
        self.pattern_result = None
        self.pattern_range_result = None
        self.pattern_error = None
        if self.pattern_search is not None:
            pattern_recording = self.recording
            if self.settings.analysis_bandwidth_hz is not None:
                pattern_recording = extract_analysis_channel(
                    self.recording,
                    center_frequency_hz=(
                        self.recording.center_frequency_hz
                        if self.settings.analysis_center_frequency_hz is None
                        else self.settings.analysis_center_frequency_hz
                    ),
                    bandwidth_hz=self.settings.analysis_bandwidth_hz,
                )
            if self.settings.remove_dc:
                pattern_recording = replace(
                    pattern_recording,
                    iq=np.asarray(pattern_recording.iq) - np.mean(pattern_recording.iq),
                )
            try:
                self.pattern_result = self._pattern_analyzer.search(
                    pattern_recording,
                    self.signal,
                    self.pattern_search,
                    self.result_range,
                    self.demodulation,
                )
                if (
                    self.pattern_search.meas_only_if_pattern_symbols_correct
                    and self.pattern_result.pattern_symbol_errors > 0
                ):
                    raise ValueError(
                        "pattern waveform matched but Pattern Symbols Correct is false"
                    )
                selected = pattern_recording.iq[
                    self.pattern_result.result_start_sample :
                    self.pattern_result.result_stop_sample
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
                self.pattern_range_result = self._analyzer.analyze(
                    range_recording,
                    self.signal,
                    replace(
                        self.settings,
                        remove_dc=False,
                        analysis_center_frequency_hz=None,
                        analysis_bandwidth_hz=None,
                    ),
                )
            except ValueError as error:
                self.pattern_error = str(error)
                if self.pattern_search.meas_only_if_pattern_symbols_correct:
                    self.pattern_result = None
                    self.pattern_range_result = None
                    raise
        return self.result
