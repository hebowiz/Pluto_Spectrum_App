import numpy as np

from tools.validate_snapshot_phase import analyze_block


def _tone(samples: int, *, phase_jump_after: int | None = None) -> np.ndarray:
    sample_rate_hz = 12_000_000.0
    n = np.arange(samples, dtype=np.float64)
    iq = np.exp(2j * np.pi * 1_000_000.0 * n / sample_rate_hz)
    if phase_jump_after is not None:
        iq[phase_jump_after:] *= np.exp(1j * np.deg2rad(30.0))
    return (256.0 * iq).astype(np.complex64)


def test_phase_continuity_analysis_accepts_continuous_cw() -> None:
    result = analyze_block(_tone(20_000), 12_000_000.0)

    assert result["phase_outlier_count"] == 0
    assert result["sample_slip_candidate_count"] == 0
    assert result["phase_residual_max_deg"] < 0.01
    assert abs(result["frequency_from_phase_hz"] - 1_000_000.0) < 0.1


def test_phase_continuity_analysis_detects_single_phase_jump() -> None:
    result = analyze_block(
        _tone(20_000, phase_jump_after=10_000),
        12_000_000.0,
    )

    assert result["phase_outlier_count"] == 1
    assert result["sample_slip_candidate_count"] == 1
    assert result["phase_residual_max_deg"] > 29.0
