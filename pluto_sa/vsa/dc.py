"""Software DC estimation for zero-IF VSA recordings."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pluto_sa.vsa.model import IQRecording


def estimate_robust_dc_offset(
    iq: np.ndarray,
    *,
    low_cluster_fraction: float = 0.20,
    max_estimation_samples: int = 200_000,
) -> complex:
    """Estimate a constant I/Q DC vector without averaging an entire burst.

    A two-dimensional histogram supplies the densest I/Q location.  The final
    estimate is the mean of the samples nearest that location, which normally
    represents the receiver-noise/no-signal cluster in a finite Pluto capture.
    This deliberately cannot separate a continuous wanted carrier located
    exactly at zero IF; intentional LO offset is the appropriate tool for that
    ambiguous case.
    """
    samples = np.asarray(iq).reshape(-1)
    finite = np.isfinite(samples.real) & np.isfinite(samples.imag)
    samples = samples[finite].astype(np.complex128, copy=False)
    if samples.size == 0:
        return 0.0j
    if not 0.0 < float(low_cluster_fraction) <= 1.0:
        raise ValueError("low_cluster_fraction must be in (0, 1]")
    if int(max_estimation_samples) <= 0:
        raise ValueError("max_estimation_samples must be positive")
    if samples.size > int(max_estimation_samples):
        indices = np.linspace(
            0,
            samples.size - 1,
            int(max_estimation_samples),
            dtype=np.int64,
        )
        samples = samples[indices]

    median = complex(np.median(samples.real), np.median(samples.imag))
    real_low, real_high = np.percentile(samples.real, (0.5, 99.5))
    imag_low, imag_high = np.percentile(samples.imag, (0.5, 99.5))
    if real_high > real_low and imag_high > imag_low:
        bin_count = min(128, max(32, int(round(np.sqrt(samples.size)))))
        histogram, real_edges, imag_edges = np.histogram2d(
            samples.real,
            samples.imag,
            bins=bin_count,
            range=((real_low, real_high), (imag_low, imag_high)),
        )
        real_bin, imag_bin = np.unravel_index(
            int(np.argmax(histogram)), histogram.shape
        )
        initial = complex(
            0.5 * (real_edges[real_bin] + real_edges[real_bin + 1]),
            0.5 * (imag_edges[imag_bin] + imag_edges[imag_bin + 1]),
        )
    else:
        initial = median
    cluster_count = max(1, int(np.ceil(samples.size * low_cluster_fraction)))
    if cluster_count >= samples.size:
        cluster = samples
    else:
        distance = np.abs(samples - initial)
        cluster_indices = np.argpartition(distance, cluster_count - 1)[:cluster_count]
        cluster = samples[cluster_indices]
    estimate = np.mean(cluster, dtype=np.complex128)
    # Grow the dense seed to cover the complete noise cluster.  Using only the
    # nearest fraction would bias the answer toward the center of one histogram
    # bin, while a generous robust radius still excludes a separated burst.
    for _ in range(4):
        residual = np.abs(cluster - estimate)
        scale = float(np.median(residual))
        if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
            break
        expanded = samples[np.abs(samples - estimate) <= 6.0 * scale]
        if expanded.size < cluster.size:
            break
        cluster = expanded
        updated = np.mean(cluster, dtype=np.complex128)
        if abs(updated - estimate) <= 1e-9 * max(1.0, abs(estimate)):
            estimate = updated
            break
        estimate = updated
    return complex(estimate) if np.isfinite(estimate) else median


def apply_robust_dc_removal(recording: IQRecording) -> IQRecording:
    """Return a full-rate recording with the robust constant DC vector removed.

    The operation is idempotent for recordings carrying our processing
    metadata.  Marking the exported recording as no longer recommending DC
    removal prevents a later file analysis from applying the same correction
    for a second time.
    """
    if bool(recording.metadata.get("software_dc_removal_applied", False)):
        return recording
    dc_offset = estimate_robust_dc_offset(recording.iq)
    return replace(
        recording,
        iq=(np.asarray(recording.iq) - dc_offset).astype(
            np.complex64, copy=False
        ),
        metadata={
            **dict(recording.metadata),
            "dc_removal_recommended": False,
            "software_dc_removal_applied": True,
            "software_dc_estimator": "low-cluster robust location",
            "software_dc_offset_real": float(dc_offset.real),
            "software_dc_offset_imag": float(dc_offset.imag),
        },
    )
