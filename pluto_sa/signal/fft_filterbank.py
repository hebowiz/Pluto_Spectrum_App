"""Gaussian FFT analysis filter bank shared by RTSA-style modes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pluto_sa.signal.measurement_filter import design_iq_rbw_filter

FFT_FILTERBANK_PROCESSING_SIGNATURE = "gaussian_fft_filterbank_v1"


@dataclass(frozen=True)
class GaussianFFTFilterBankDesign:
    """Resolved per-bin Gaussian analysis-filter characteristics."""

    sample_rate_hz: float
    fft_size: int
    bin_width_hz: float
    requested_rbw_hz: float
    effective_rbw_hz: float
    noise_equivalent_bandwidth_hz: float
    support_samples: int
    rbw_limited_by_fft_size: bool


@dataclass(frozen=True)
class AutomaticRTSAFFTDesign:
    """FFT dimensions derived from the user-facing Span/RBW conditions."""

    fft_size: int
    window_length_samples: int
    requested_display_bins: int
    available_display_bins: int
    limited_by_fft_size: bool


def _next_power_of_two(value: int) -> int:
    resolved = max(1, int(value))
    return 1 << int(np.ceil(np.log2(resolved)))


def resolve_automatic_rtsa_fft_design(
    *,
    sample_rate_hz: float,
    rbw_hz: float,
    guard_ratio: float,
    minimum_display_bins: int = 1024,
    maximum_fft_size: int = 16_384,
) -> AutomaticRTSAFFTDesign:
    """Choose NFFT from RBW support and useful on-screen bin density.

    The RBW defines the non-zero analysis-window length. NFFT is independently
    enlarged, by zero padding when necessary, so the guarded display span has
    enough frequency bins. The maximum is a processing-policy ceiling rather
    than a change to the requested Span/RBW measurement condition.
    """
    if not np.isfinite(sample_rate_hz) or float(sample_rate_hz) <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if not np.isfinite(rbw_hz) or float(rbw_hz) <= 0.0:
        raise ValueError("rbw_hz must be positive")
    guard = float(guard_ratio)
    if not 0.0 <= guard < 0.5:
        raise ValueError("guard_ratio must be in [0, 0.5)")
    requested_bins = max(16, int(minimum_display_bins))
    maximum_size = max(16, int(maximum_fft_size))

    coefficients, _design = design_iq_rbw_filter(
        float(sample_rate_hz),
        float(rbw_hz),
        shape="gaussian",
    )
    window_length = int(len(coefficients))
    display_fraction = 1.0 - 2.0 * guard
    density_fft_size = _next_power_of_two(
        int(np.ceil(float(requested_bins) / display_fraction))
    )
    required_size = _next_power_of_two(max(window_length, density_fft_size))
    resolved_size = min(required_size, maximum_size)
    available_bins = max(1, int(np.floor(resolved_size * display_fraction)))
    return AutomaticRTSAFFTDesign(
        fft_size=resolved_size,
        window_length_samples=min(window_length, resolved_size),
        requested_display_bins=requested_bins,
        available_display_bins=available_bins,
        limited_by_fft_size=required_size > maximum_size,
    )


def minimum_gaussian_rbw_hz(sample_rate_hz: float, fft_size: int) -> float:
    """Return the narrowest +/-4 sigma Gaussian FIR that fits one FFT frame."""
    if not np.isfinite(sample_rate_hz) or float(sample_rate_hz) <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if int(fft_size) < 3:
        raise ValueError("fft_size must be at least 3")
    max_half_width = (int(fft_size) - 1) // 2
    return float(
        4.0
        * np.sqrt(np.log(2.0))
        * float(sample_rate_hz)
        / (np.pi * float(max_half_width))
    )


def required_gaussian_fft_size(sample_rate_hz: float, rbw_hz: float) -> int:
    """Return the smallest power-of-two FFT that contains the Gaussian support."""
    if not np.isfinite(rbw_hz) or float(rbw_hz) <= 0.0:
        raise ValueError("rbw_hz must be positive")
    sigma_samples = (
        np.sqrt(np.log(2.0))
        * float(sample_rate_hz)
        / (np.pi * float(rbw_hz))
    )
    support_samples = 2 * max(1, int(np.ceil(4.0 * sigma_samples))) + 1
    return max(4, 1 << int(np.ceil(np.log2(support_samples))))


def design_gaussian_fft_filterbank(
    sample_rate_hz: float,
    fft_size: int,
    rbw_hz: float | None,
) -> tuple[np.ndarray, GaussianFFTFilterBankDesign]:
    """Build a zero-padded Gaussian analysis window for an FFT filter bank.

    Multiplication by this window followed by an FFT is a bank of frequency-
    translated copies of the same complex-IQ Gaussian FIR used by Sweep/TA.
    """
    resolved_fft_size = int(fft_size)
    if resolved_fft_size < 3:
        raise ValueError("fft_size must be at least 3")
    bin_width_hz = float(sample_rate_hz) / float(resolved_fft_size)
    requested_rbw_hz = (
        bin_width_hz if rbw_hz is None else float(rbw_hz)
    )
    if not np.isfinite(requested_rbw_hz) or requested_rbw_hz <= 0.0:
        raise ValueError("rbw_hz must be positive")
    minimum_rbw_hz = minimum_gaussian_rbw_hz(sample_rate_hz, resolved_fft_size)
    effective_request_hz = max(requested_rbw_hz, minimum_rbw_hz)
    coefficients, iq_design = design_iq_rbw_filter(
        float(sample_rate_hz),
        effective_request_hz,
        shape="gaussian",
    )
    if len(coefficients) > resolved_fft_size:
        raise RuntimeError("resolved Gaussian analysis window exceeds FFT size")

    window = np.zeros(resolved_fft_size, dtype=np.float64)
    start = (resolved_fft_size - len(coefficients)) // 2
    window[start : start + len(coefficients)] = coefficients
    window.setflags(write=False)
    design = GaussianFFTFilterBankDesign(
        sample_rate_hz=float(sample_rate_hz),
        fft_size=resolved_fft_size,
        bin_width_hz=bin_width_hz,
        requested_rbw_hz=requested_rbw_hz,
        effective_rbw_hz=float(iq_design.effective_rbw_hz),
        noise_equivalent_bandwidth_hz=float(
            iq_design.noise_equivalent_bandwidth_hz
        ),
        support_samples=int(len(coefficients)),
        rbw_limited_by_fft_size=bool(requested_rbw_hz < minimum_rbw_hz),
    )
    return window, design
