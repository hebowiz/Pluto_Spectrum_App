"""Vector signal analysis contracts and offline processing."""

from pluto_vsa.analysis import VSAAnalyzer
from pluto_vsa.channel import extract_analysis_channel
from pluto_vsa.model import (
    CompositeSignalDescription,
    CompositeVSAAnalysisResult,
    IQRecording,
    ModulationFamily,
    ModulationKind,
    ModulationSegment,
    SignalDescription,
    VSAAnalysisResult,
    VSASegmentAnalysis,
    VSASettings,
)
from pluto_vsa.session import VSASession
from pluto_vsa.pattern import (
    BitOrdering,
    carrier_correct_recording,
    DemodulationSettings,
    KnownPattern,
    MatchSelectionPolicy,
    MeasurementFilterMode,
    PatternAnalyzer,
    PatternSearchMode,
    PatternSearchResult,
    PatternSearchSettings,
    ResultRangeAlignment,
    ResultRangeReference,
    ResultRangeSettings,
    SynchronizationSource,
)

__all__ = [
    "CompositeSignalDescription",
    "CompositeVSAAnalysisResult",
    "IQRecording",
    "BitOrdering",
    "carrier_correct_recording",
    "DemodulationSettings",
    "KnownPattern",
    "MatchSelectionPolicy",
    "MeasurementFilterMode",
    "ModulationFamily",
    "ModulationKind",
    "ModulationSegment",
    "PatternAnalyzer",
    "PatternSearchMode",
    "PatternSearchResult",
    "PatternSearchSettings",
    "ResultRangeAlignment",
    "ResultRangeReference",
    "ResultRangeSettings",
    "SignalDescription",
    "VSAAnalysisResult",
    "VSAAnalyzer",
    "VSASettings",
    "VSASegmentAnalysis",
    "VSASession",
    "SynchronizationSource",
    "extract_analysis_channel",
]
