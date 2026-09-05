"""Single-channel selection for source-independent VSA IQ recordings."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, firwin

from pluto_sa.vsa.model import IQRecording


def validate_analysis_channel_capture(
    *,
    sample_rate_hz: float,
    usable_bandwidth_hz: float,
    lo_offset_hz: float,
    analysis_bandwidth_hz: float | None,
) -> None:
    """Validate the shared VSA analysis-filter/offset-LO geometry."""

    if analysis_bandwidth_hz is None:
        if lo_offset_hz != 0.0:
            raise ValueError("LO Offset requires Enable Analysis Channel")
        return
    bandwidth_hz = float(analysis_bandwidth_hz)
    input_rate_hz = float(sample_rate_hz)
    usable_hz = min(input_rate_hz, float(usable_bandwidth_hz))
    if bandwidth_hz >= input_rate_hz:
        raise ValueError(
            "Analysis Bandwidth must be smaller than the input sample rate"
        )
    if abs(float(lo_offset_hz)) + 0.5 * bandwidth_hz > 0.5 * usable_hz:
        raise ValueError(
            "LO Offset and Analysis Bandwidth exceed the usable Pluto "
            "capture bandwidth"
        )
    if lo_offset_hz != 0.0 and abs(float(lo_offset_hz)) <= 0.5 * bandwidth_hz:
        raise ValueError(
            "LO Offset must exceed half the Analysis Bandwidth so the "
            "Analysis Filter can reject the Pluto DC spur"
        )


def extract_requested_analysis_channel(recording: IQRecording) -> IQRecording:
    """Apply the analysis channel requested by a Pluto live capture."""

    bandwidth = recording.metadata.get("requested_analysis_bandwidth_hz")
    if bandwidth is None:
        return recording
    center = recording.metadata.get(
        "requested_center_frequency_hz", recording.center_frequency_hz
    )
    return extract_analysis_channel(
        recording,
        center_frequency_hz=float(center),
        bandwidth_hz=float(bandwidth),
    )


def _default_decimation(sample_rate_hz: float, bandwidth_hz: float) -> int:
    """Keep approximately four output samples per analysis-bandwidth period."""
    desired_rate_hz = 4.0 * float(bandwidth_hz)
    candidate = max(1, int(round(float(sample_rate_hz) / desired_rate_hz)))
    while candidate > 1 and float(sample_rate_hz) / candidate < 2.5 * bandwidth_hz:
        candidate -= 1
    return candidate


def _filter_taps(sample_rate_hz: float, bandwidth_hz: float) -> np.ndarray:
    # Roughly eight taps per input-sample/channel-bandwidth ratio gives useful
    # adjacent-channel rejection without making short interactive captures
    # unnecessarily expensive.  Keep an odd length for an integer group delay.
    count = int(np.ceil(8.0 * float(sample_rate_hz) / float(bandwidth_hz)))
    count = min(2049, max(65, count))
    if count % 2 == 0:
        count += 1
    return firwin(
        count,
        cutoff=0.5 * float(bandwidth_hz),
        fs=float(sample_rate_hz),
        window=("kaiser", 8.0),
        scale=True,
    )


def extract_analysis_channel(
    recording: IQRecording,
    *,
    center_frequency_hz: float,
    bandwidth_hz: float,
) -> IQRecording:
    """DDC, low-pass filter and integer-decimate one manually selected channel.

    ``center_frequency_hz`` is absolute when the source recording has an RF
    center, and relative to zero for baseband/generated recordings.  The
    returned recording is centered on the selected channel and retains the
    original amplitude-correction convention.
    """
    input_rate_hz = float(recording.sample_rate_hz)
    selected_center_hz = float(center_frequency_hz)
    selected_bandwidth_hz = float(bandwidth_hz)
    if not np.isfinite(selected_center_hz):
        raise ValueError("analysis center frequency must be finite")
    if not np.isfinite(selected_bandwidth_hz) or selected_bandwidth_hz <= 0.0:
        raise ValueError("analysis bandwidth must be positive")
    if selected_bandwidth_hz >= input_rate_hz:
        raise ValueError(
            "analysis bandwidth must be smaller than the input sample rate"
        )

    offset_hz = selected_center_hz - float(recording.center_frequency_hz)
    usable_bandwidth_hz = min(
        input_rate_hz,
        float(recording.usable_bandwidth_hz or input_rate_hz),
    )
    if abs(offset_hz) + 0.5 * selected_bandwidth_hz > 0.5 * usable_bandwidth_hz:
        raise ValueError(
            "selected analysis channel exceeds the usable capture bandwidth"
        )

    source_indices = recording.start_sample_index + np.arange(
        recording.sample_count, dtype=np.float64
    )
    oscillator = np.exp(-2j * np.pi * offset_hz * source_indices / input_rate_hz)
    translated = np.asarray(recording.iq, dtype=np.complex128) * oscillator
    taps = _filter_taps(input_rate_hz, selected_bandwidth_hz)
    filtered = fftconvolve(translated, taps, mode="same")
    decimation = _default_decimation(input_rate_hz, selected_bandwidth_hz)
    output_iq = filtered[::decimation].astype(np.complex64)
    output_rate_hz = input_rate_hz / decimation

    output_start_index = int(recording.start_sample_index) // decimation
    output_trigger_index: int | None = None
    if recording.trigger_sample_index is not None:
        trigger_local = recording.trigger_sample_index - recording.start_sample_index
        output_trigger_index = output_start_index + int(round(trigger_local / decimation))
        output_trigger_index = min(
            output_start_index + output_iq.size - 1,
            max(output_start_index, output_trigger_index),
        )

    return IQRecording(
        iq=output_iq,
        sample_rate_hz=output_rate_hz,
        center_frequency_hz=selected_center_hz,
        usable_bandwidth_hz=selected_bandwidth_hz,
        source=f"{recording.source} | Analysis channel",
        full_scale=recording.full_scale,
        calibration_offset_db=recording.calibration_offset_db,
        frequency_dependent_offset_db=recording.frequency_dependent_offset_db,
        input_correction_db=recording.input_correction_db,
        amplitude_calibrated=recording.amplitude_calibrated,
        start_sample_index=output_start_index,
        trigger_sample_index=output_trigger_index,
        discontinuity_reason=recording.discontinuity_reason,
        metadata={
            **dict(recording.metadata),
            "analysis_channel_applied": True,
            "analysis_center_frequency_hz": selected_center_hz,
            "analysis_center_offset_hz": offset_hz,
            "analysis_bandwidth_hz": selected_bandwidth_hz,
            "analysis_input_sample_rate_hz": input_rate_hz,
            "analysis_decimation": decimation,
            "analysis_filter_taps": int(taps.size),
            "source_start_sample_index": int(recording.start_sample_index),
        },
    )
