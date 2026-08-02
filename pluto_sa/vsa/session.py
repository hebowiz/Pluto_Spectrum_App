"""VSA session state independent of the Qt application shell."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from pluto_sa.vsa.analysis import VSAAnalyzer
from pluto_sa.vsa.model import IQRecording, SignalDescription, VSAAnalysisResult, VSASettings


@dataclass
class VSASession:
    name: str = "VSA 1"
    recording: IQRecording | None = None
    signal: SignalDescription | None = None
    settings: VSASettings = field(default_factory=VSASettings)
    result: VSAAnalysisResult | None = None
    revision: int = 0
    _analyzer: VSAAnalyzer = field(default_factory=VSAAnalyzer, repr=False)

    def set_recording(self, recording: IQRecording) -> None:
        self.recording = recording
        self.result = None
        self.revision += 1

    def set_signal(self, signal: SignalDescription) -> None:
        self.signal = signal
        self.result = None
        self.revision += 1

    def update_settings(self, **changes: object) -> None:
        self.settings = replace(self.settings, **changes)
        self.result = None
        self.revision += 1

    def analyze(self) -> VSAAnalysisResult:
        if self.recording is None:
            raise RuntimeError("no IQ recording is loaded")
        if self.signal is None:
            raise RuntimeError("no signal description is configured")
        self.result = self._analyzer.analyze(self.recording, self.signal, self.settings)
        return self.result
