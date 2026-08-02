"""Measure CW phase continuity inside and between Pluto RX snapshot buffers."""

from __future__ import annotations

import argparse
import json
import time

import adi
import numpy as np


def analyze_block(iq: np.ndarray, sample_rate_hz: float) -> dict:
    values = np.asarray(iq, dtype=np.complex128)
    centered = values - np.mean(values)
    magnitude = np.abs(centered)
    products = centered[1:] * np.conj(centered[:-1])
    mean_step_rad = float(np.angle(np.sum(products)))
    frequency_hz = mean_step_rad * float(sample_rate_hz) / (2.0 * np.pi)
    phase_steps = np.angle(products)
    residual_rad = np.angle(np.exp(1j * (phase_steps - mean_step_rad)))
    residual_center_rad = float(np.median(residual_rad))
    absolute_residual_rad = np.abs(residual_rad - residual_center_rad)
    mad_rad = float(1.4826 * np.median(absolute_residual_rad))
    threshold_rad = max(np.deg2rad(10.0), 10.0 * mad_rad)
    outlier_indices = np.flatnonzero(absolute_residual_rad > threshold_rad)
    top_indices = np.argsort(absolute_residual_rad)[-5:][::-1]

    window = np.hanning(len(centered))
    spectrum_power = np.abs(np.fft.fft(centered * window)) ** 2
    peak_index = int(np.argmax(spectrum_power))
    excluded = np.ones(len(spectrum_power), dtype=bool)
    for offset in range(-3, 4):
        excluded[(peak_index + offset) % len(excluded)] = False
    noise_floor = float(np.median(spectrum_power[excluded]))
    peak_snr_db = float(
        10.0 * np.log10(max(float(spectrum_power[peak_index]), 1e-300) / max(noise_floor, 1e-300))
    )
    fft_frequency_hz = float(np.fft.fftfreq(len(centered), 1.0 / sample_rate_hz)[peak_index])

    return {
        "samples": int(len(values)),
        "mean_i": float(np.mean(values.real)),
        "mean_q": float(np.mean(values.imag)),
        "rms_magnitude": float(np.sqrt(np.mean(magnitude**2))),
        "peak_magnitude": float(np.max(magnitude)),
        "adc_clip_fraction": float(
            np.mean((np.abs(values.real) >= 2040.0) | (np.abs(values.imag) >= 2040.0))
        ),
        "frequency_from_phase_hz": frequency_hz,
        "frequency_from_fft_hz": fft_frequency_hz,
        "mean_phase_step_deg": float(np.rad2deg(mean_step_rad)),
        "phase_step_mad_deg": float(np.rad2deg(mad_rad)),
        "phase_residual_p99_deg": float(np.rad2deg(np.percentile(absolute_residual_rad, 99.0))),
        "phase_residual_p999_deg": float(np.rad2deg(np.percentile(absolute_residual_rad, 99.9))),
        "phase_residual_max_deg": float(np.rad2deg(np.max(absolute_residual_rad))),
        "phase_outlier_threshold_deg": float(np.rad2deg(threshold_rad)),
        "phase_outlier_count": int(outlier_indices.size),
        "largest_phase_residuals": [
            {
                "after_sample": int(index),
                "residual_deg": float(np.rad2deg(residual_rad[index] - residual_center_rad)),
            }
            for index in top_indices
        ],
        "fft_peak_snr_db": peak_snr_db,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="usb:1.54.5")
    parser.add_argument("--center-frequency", type=int, default=2_440_000_000)
    parser.add_argument("--sample-rate", type=int, default=12_000_000)
    parser.add_argument("--rf-bandwidth", type=int, default=12_000_000)
    parser.add_argument("--samples", type=int, default=120_000)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--gain", type=int, default=30)
    args = parser.parse_args()

    sdr = adi.Pluto(uri=args.uri)
    sdr.rx_lo = int(args.center_frequency)
    sdr.sample_rate = int(args.sample_rate)
    sdr.rx_rf_bandwidth = int(args.rf_bandwidth)
    sdr.rx_buffer_size = int(args.samples)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = int(args.gain)
    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass

    captures: list[np.ndarray] = []
    elapsed_s: list[float] = []
    try:
        for _ in range(int(args.blocks)):
            started = time.perf_counter()
            captures.append(np.asarray(sdr.rx(), dtype=np.complex64).copy())
            elapsed_s.append(time.perf_counter() - started)
    finally:
        try:
            sdr.rx_destroy_buffer()
        except Exception:
            pass

    analyses = [analyze_block(block, float(args.sample_rate)) for block in captures]
    boundary_residuals = []
    for index in range(1, len(captures)):
        expected_step = np.deg2rad(analyses[index - 1]["mean_phase_step_deg"])
        actual_step = np.angle(captures[index][0] * np.conj(captures[index - 1][-1]))
        residual = np.angle(np.exp(1j * (actual_step - expected_step)))
        boundary_residuals.append(
            {
                "before_block": index - 1,
                "after_block": index,
                "phase_residual_deg": float(np.rad2deg(residual)),
            }
        )

    print(
        json.dumps(
            {
                "uri": args.uri,
                "center_frequency_hz": int(args.center_frequency),
                "sample_rate_hz": int(args.sample_rate),
                "samples_per_block": int(args.samples),
                "expected_block_duration_ms": args.samples / args.sample_rate * 1e3,
                "capture_elapsed_ms": [value * 1e3 for value in elapsed_s],
                "blocks": analyses,
                "boundary_phase": boundary_residuals,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
