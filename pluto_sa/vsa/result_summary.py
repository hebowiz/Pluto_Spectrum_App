"""Canonical Result Summary item definitions shared by UI and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pluto_sa.vsa.model import ModulationFamily


class ResultSummaryCategory(str, Enum):
    COMMON = "Common Measurement Results"
    PSK = "PSK Measurement Results"
    FSK = "FSK Measurement Results"
    DIAGNOSTICS = "Synchronization Diagnostics"


@dataclass(frozen=True)
class ResultSummaryItem:
    item_id: str
    label: str
    category: ResultSummaryCategory
    families: frozenset[ModulationFamily]
    implemented: bool
    default_visible: bool
    description: str

    def applies_to(self, family: ModulationFamily) -> bool:
        return family in self.families


_BOTH = frozenset((ModulationFamily.FSK, ModulationFamily.PSK))
_PSK = frozenset((ModulationFamily.PSK,))
_FSK = frozenset((ModulationFamily.FSK,))


def _item(
    item_id: str,
    label: str,
    category: ResultSummaryCategory,
    families: frozenset[ModulationFamily],
    *,
    implemented: bool,
    default: bool = False,
    description: str,
) -> ResultSummaryItem:
    return ResultSummaryItem(
        item_id=item_id,
        label=label,
        category=category,
        families=families,
        implemented=implemented,
        default_visible=default,
        description=description,
    )


RESULT_SUMMARY_ITEMS = (
    _item("modulation", "Modulation", ResultSummaryCategory.COMMON, _BOTH,
          implemented=True, default=True, description="Configured modulation type."),
    _item("power", "Power", ResultSummaryCategory.COMMON, _BOTH,
          implemented=True, default=True, description="Linear mean power over the analyzed result data."),
    _item("carrier_frequency_error", "Carrier Frequency Error", ResultSummaryCategory.COMMON, _BOTH,
          implemented=True, default=True, description="Estimated carrier frequency offset (CFO)."),
    _item("evm_rms", "EVM RMS", ResultSummaryCategory.PSK, _PSK,
          implemented=True, default=True, description="RMS error between physical absolute IQ decision points and the absolute reference sequence."),
    _item("differential_symbol_evm_rms", "Differential Symbol EVM RMS", ResultSummaryCategory.PSK, _PSK,
          implemented=True, default=True, description="RMS error of measured adjacent-symbol differential vectors against their ideal phase-shift symbols."),
    _item("bluetooth_devm_rms", "Bluetooth DEVM RMS", ResultSummaryCategory.PSK, _PSK,
          implemented=True, default=True, description="Bluetooth Appendix C differential EVM calculated from adjacent changes after removing the ideal reference sequence."),
    _item("evm_peak", "EVM Peak", ResultSummaryCategory.PSK, _PSK,
          implemented=False, description="Peak error-vector magnitude in the evaluation range."),
    _item("mer_rms", "MER RMS", ResultSummaryCategory.PSK, _PSK,
          implemented=False, description="RMS modulation error ratio."),
    _item("mer_peak", "MER Peak", ResultSummaryCategory.PSK, _PSK,
          implemented=False, description="Worst modulation error ratio."),
    _item("phase_error_rms", "Phase Error RMS", ResultSummaryCategory.PSK, _PSK,
          implemented=False, description="RMS phase difference from the reference vector."),
    _item("phase_error_peak", "Phase Error Peak", ResultSummaryCategory.PSK, _PSK,
          implemented=False, description="Peak phase difference from the reference vector."),
    _item("magnitude_error_rms_psk", "Magnitude Error RMS", ResultSummaryCategory.PSK, _PSK,
          implemented=False, description="RMS magnitude difference from the reference vector."),
    _item("magnitude_error_peak_psk", "Magnitude Error Peak", ResultSummaryCategory.PSK, _PSK,
          implemented=False, description="Peak magnitude difference from the reference vector."),
    _item("symbol_rate_error", "Symbol Rate Error", ResultSummaryCategory.PSK, _PSK,
          implemented=True, default=True, description="Measured symbol-rate difference in ppm."),
    _item("iq_skew", "I/Q Skew", ResultSummaryCategory.PSK, _PSK,
          implemented=False, description="Constant timing difference between I and Q."),
    _item("rho", "Rho", ResultSummaryCategory.PSK, _PSK,
          implemented=False, description="Normalized correlation of measured and reference waveforms."),
    _item("iq_offset", "I/Q Offset", ResultSummaryCategory.PSK, _PSK,
          implemented=False, description="Origin/DC offset of the measured I/Q signal."),
    _item("iq_imbalance", "I/Q Imbalance", ResultSummaryCategory.PSK, _PSK,
          implemented=False, description="Combined gain imbalance and quadrature error."),
    _item("gain_imbalance", "Gain Imbalance", ResultSummaryCategory.PSK, _PSK,
          implemented=False, description="Gain difference between I and Q paths."),
    _item("quadrature_error", "Quadrature Error", ResultSummaryCategory.PSK, _PSK,
          implemented=False, description="Departure of I/Q phase separation from 90 degrees."),
    _item("amplitude_droop", "Amplitude Droop", ResultSummaryCategory.PSK, _PSK,
          implemented=False, description="Signal amplitude decrease over time."),
    _item("frequency_error_rms", "Frequency Error RMS", ResultSummaryCategory.FSK, _FSK,
          implemented=True, default=True, description="FSK frequency-model RMS residual normalized to measured deviation."),
    _item("frequency_error_peak", "Frequency Error Peak", ResultSummaryCategory.FSK, _FSK,
          implemented=False, description="Peak FSK frequency error normalized to measured deviation."),
    _item("magnitude_error_rms_fsk", "Magnitude Error RMS", ResultSummaryCategory.FSK, _FSK,
          implemented=False, description="RMS magnitude difference for reconstructed FSK waveforms."),
    _item("magnitude_error_peak_fsk", "Magnitude Error Peak", ResultSummaryCategory.FSK, _FSK,
          implemented=False, description="Peak magnitude difference for reconstructed FSK waveforms."),
    _item("fsk_deviation_error", "FSK Deviation Error", ResultSummaryCategory.FSK, _FSK,
          implemented=True, default=True, description="Measured deviation minus configured reference deviation."),
    _item("fsk_measured_deviation", "FSK Meas Deviation", ResultSummaryCategory.FSK, _FSK,
          implemented=True, default=True, description="Frequency deviation estimated from the measured FSK signal."),
    _item("fsk_reference_deviation", "FSK Ref Deviation", ResultSummaryCategory.FSK, _FSK,
          implemented=True, description="User-configured reference frequency deviation."),
    _item("carrier_frequency_drift", "Carrier Frequency Drift", ResultSummaryCategory.FSK, _FSK,
          implemented=True, default=True, description="Linear FSK carrier drift in Hz per symbol."),
    _item("pattern_symbols_correct", "Pattern Symbols Correct", ResultSummaryCategory.DIAGNOSTICS, _BOTH,
          implemented=True, default=True, description="Whether every configured pattern symbol matched."),
    _item("pattern_match_variant", "Pattern Match", ResultSummaryCategory.DIAGNOSTICS, _FSK,
          implemented=True, default=True, description="Whether the configured FSK pattern or its bitwise complement matched."),
    _item("iq_correlation", "I/Q Correlation", ResultSummaryCategory.DIAGNOSTICS, _BOTH,
          implemented=True, default=True, description="Normalized pattern waveform correlation."),
    _item("match_selection", "Selected Result", ResultSummaryCategory.DIAGNOSTICS, _BOTH,
          implemented=True, default=True, description="One-based selected packet index and eligible packet count."),
    _item("result_symbols", "Result Symbols", ResultSummaryCategory.DIAGNOSTICS, _BOTH,
          implemented=True, default=True, description="Number of demodulated symbols in the current result."),
    _item("pattern_error", "Pattern Error", ResultSummaryCategory.DIAGNOSTICS, _BOTH,
          implemented=True, default=True, description="Pattern-search failure reason."),
    _item("estimated_carrier", "Estimated Carrier", ResultSummaryCategory.DIAGNOSTICS, _BOTH,
          implemented=True, description="Analysis center plus the estimated CFO."),
    _item("display", "Display", ResultSummaryCategory.DIAGNOSTICS, _BOTH,
          implemented=True, description="Whether raw or carrier-corrected results are displayed."),
    _item("psk_carrier_drift", "PSK Carrier Drift", ResultSummaryCategory.DIAGNOSTICS, _PSK,
          implemented=True, description="PSK synchronization drift estimate."),
    _item("sync_evm_rms", "Sync EVM RMS", ResultSummaryCategory.DIAGNOSTICS, _PSK,
          implemented=True, description="Complex-EVM objective used by the PSK synchronizer."),
    _item("fractional_timing", "Fractional Timing", ResultSummaryCategory.DIAGNOSTICS, _FSK,
          implemented=True, description="Estimated and applied fractional FSK symbol timing."),
    _item("frequency_fit_rms", "Frequency Fit RMS", ResultSummaryCategory.DIAGNOSTICS, _FSK,
          implemented=True, description="Residual of the FSK synchronization frequency model."),
    _item("timing_confidence", "Timing Confidence", ResultSummaryCategory.DIAGNOSTICS, _FSK,
          implemented=True, description="Confidence score for FSK fractional timing."),
    _item("deviation_error_percent", "Deviation Error (%)", ResultSummaryCategory.DIAGNOSTICS, _FSK,
          implemented=True, description="Diagnostic relative FSK deviation error."),
    _item("drift_model", "Drift Model", ResultSummaryCategory.DIAGNOSTICS, _FSK,
          implemented=True, description="FSK drift-model quality-gate result."),
    _item("applied_drift", "Applied Drift", ResultSummaryCategory.DIAGNOSTICS, _FSK,
          implemented=True, description="FSK drift actually applied by compensation."),
)

RESULT_SUMMARY_BY_ID = {item.item_id: item for item in RESULT_SUMMARY_ITEMS}
DEFAULT_RESULT_SUMMARY_IDS = frozenset(
    item.item_id for item in RESULT_SUMMARY_ITEMS
    if item.implemented and item.default_visible
)


def normalize_result_summary_ids(values: object) -> set[str]:
    """Validate persisted IDs, ignoring entries added by newer application versions."""

    if values is None:
        return set(DEFAULT_RESULT_SUMMARY_IDS)
    if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
        raise ValueError("visible Result Summary items must be an array of strings")
    return {
        value for value in values
        if value in RESULT_SUMMARY_BY_ID and RESULT_SUMMARY_BY_ID[value].implemented
    }
