"""Deterministic offline FSK/PSK analysis used by VSA sessions."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pluto_sa.vsa.model import (
    CompositeSignalDescription,
    CompositeVSAAnalysisResult,
    IQRecording,
    ModulationFamily,
    ModulationKind,
    SignalDescription,
    VSAAnalysisResult,
    VSASegmentAnalysis,
    VSASettings,
)


_EPSILON = np.finfo(np.float64).tiny


def _constellation(kind: ModulationKind) -> np.ndarray:
    if kind is ModulationKind.BPSK:
        phases = np.array([0.0, np.pi])
    elif kind in {ModulationKind.QPSK, ModulationKind.OQPSK}:
        phases = np.pi / 4.0 + np.arange(4) * np.pi / 2.0
    elif kind is ModulationKind.PI4_DQPSK:
        phases = np.pi / 4.0 + np.arange(4) * np.pi / 2.0
    elif kind is ModulationKind.DPSK8:
        phases = np.arange(8) * np.pi / 4.0
    else:
        raise ValueError(f"{kind.value} is not a PSK modulation")
    return np.exp(1j * phases)


def _symbols_to_bits(symbols: np.ndarray, order: int) -> np.ndarray:
    bits_per_symbol = int(round(np.log2(order)))
    shifts = np.arange(bits_per_symbol - 1, -1, -1, dtype=np.int16)
    return ((symbols[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def _sample_centers(sample_count: int, samples_per_symbol: float, offset: float) -> np.ndarray:
    if samples_per_symbol < 1.0:
        raise ValueError("recording sample rate must be at least the symbol rate")
    first = samples_per_symbol / 2.0 - 0.5 + float(offset)
    centers = np.arange(first, sample_count, samples_per_symbol, dtype=np.float64)
    return centers[(centers >= 0.0) & (centers <= sample_count - 1)]


def _interpolate_complex(iq: np.ndarray, positions: np.ndarray) -> np.ndarray:
    samples = np.arange(iq.size, dtype=np.float64)
    real = np.interp(positions, samples, iq.real)
    imag = np.interp(positions, samples, iq.imag)
    return real + 1j * imag


def _spectrum(iq: np.ndarray, sample_rate_hz: float, fft_size: int) -> tuple[np.ndarray, np.ndarray]:
    used = min(iq.size, int(fft_size))
    start = max(0, (iq.size - used) // 2)
    values = np.asarray(iq[start : start + used], dtype=np.complex128)
    window = np.hanning(used) if used > 1 else np.ones(used)
    transform = np.fft.fftshift(np.fft.fft(values * window, n=int(fft_size)))
    amplitude = np.abs(transform) / max(float(np.sum(window)), 1.0)
    frequency = np.fft.fftshift(np.fft.fftfreq(int(fft_size), d=1.0 / sample_rate_hz))
    return frequency, 20.0 * np.log10(np.maximum(amplitude, _EPSILON))


def _instantaneous_frequency(iq: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    if iq.size == 1:
        return np.zeros(1, dtype=np.float64)
    frequency = np.angle(iq[1:] * np.conj(iq[:-1])) * sample_rate_hz / (2.0 * np.pi)
    return np.concatenate(([frequency[0]], frequency)).astype(np.float64)


class VSAAnalyzer:
    """Analyze one contiguous recording using an explicit signal description."""

    def analyze(
        self,
        recording: IQRecording,
        signal: SignalDescription,
        settings: VSASettings | None = None,
    ) -> VSAAnalysisResult:
        resolved = settings or VSASettings()
        iq = np.asarray(recording.iq, dtype=np.complex128)
        if resolved.remove_dc:
            iq = iq - np.mean(iq)
        sample_rate_hz = float(recording.sample_rate_hz)
        time_s = np.arange(iq.size, dtype=np.float64) / sample_rate_hz
        power_dbfs = 20.0 * np.log10(
            np.maximum(np.abs(iq) / float(recording.full_scale), _EPSILON)
        )
        power_dbm = power_dbfs + recording.dbfs_to_dbm_offset_db
        frequency_hz, spectrum_dbfs = _spectrum(
            iq / float(recording.full_scale), sample_rate_hz, resolved.fft_size
        )
        inst_frequency_hz = _instantaneous_frequency(iq, sample_rate_hz)
        samples_per_symbol = sample_rate_hz / float(signal.symbol_rate_hz)
        centers = _sample_centers(
            iq.size, samples_per_symbol, resolved.timing_offset_samples
        )

        if signal.modulation.family is ModulationFamily.FSK:
            return self._analyze_fsk(
                recording,
                signal,
                iq,
                time_s,
                power_dbfs,
                power_dbm,
                frequency_hz,
                spectrum_dbfs,
                inst_frequency_hz,
                centers,
                samples_per_symbol,
            )
        return self._analyze_psk(
            recording,
            signal,
            iq,
            time_s,
            power_dbfs,
            power_dbm,
            frequency_hz,
            spectrum_dbfs,
            inst_frequency_hz,
            centers,
        )

    def analyze_composite(
        self,
        recording: IQRecording,
        description: CompositeSignalDescription,
        settings: VSASettings | None = None,
    ) -> CompositeVSAAnalysisResult:
        """Analyze explicit non-overlapping modulation regions in one record."""
        results: list[VSASegmentAnalysis] = []
        decoded_parts: list[np.ndarray] = []
        for segment in description.segments:
            if segment.stop_sample > recording.sample_count:
                raise ValueError(
                    f"segment {segment.name!r} exceeds the IQ recording"
                )
            iq = recording.iq[segment.start_sample : segment.stop_sample]
            segment_recording = IQRecording(
                iq=iq,
                sample_rate_hz=recording.sample_rate_hz,
                center_frequency_hz=recording.center_frequency_hz,
                usable_bandwidth_hz=recording.usable_bandwidth_hz,
                source=recording.source,
                full_scale=recording.full_scale,
                calibration_offset_db=recording.calibration_offset_db,
                frequency_dependent_offset_db=recording.frequency_dependent_offset_db,
                input_correction_db=recording.input_correction_db,
                amplitude_calibrated=recording.amplitude_calibrated,
                start_sample_index=recording.start_sample_index + segment.start_sample,
                discontinuity_reason=recording.discontinuity_reason,
                metadata={
                    **dict(recording.metadata),
                    "segment_name": segment.name,
                    "segment_start_sample": segment.start_sample,
                },
            )
            local_result = self.analyze(segment_recording, segment.signal, settings)
            offset_s = segment.start_sample / recording.sample_rate_hz
            absolute_result = replace(
                local_result,
                time_s=local_result.time_s + offset_s,
                symbol_time_s=local_result.symbol_time_s + offset_s,
            )
            results.append(VSASegmentAnalysis(segment=segment, result=absolute_result))
            decoded_parts.append(absolute_result.decoded_bits)
        decoded = np.concatenate(decoded_parts) if decoded_parts else np.empty(0, dtype=np.uint8)
        return CompositeVSAAnalysisResult(
            profile_name=description.profile_name,
            segments=tuple(results),
            decoded_bits=decoded,
            metadata={
                "source": recording.source,
                "sample_rate_hz": recording.sample_rate_hz,
                "segment_count": len(results),
            },
        )

    def _analyze_fsk(
        self,
        recording: IQRecording,
        signal: SignalDescription,
        iq: np.ndarray,
        time_s: np.ndarray,
        power_dbfs: np.ndarray,
        power_dbm: np.ndarray,
        spectrum_frequency_hz: np.ndarray,
        spectrum_dbfs: np.ndarray,
        instantaneous_frequency_hz: np.ndarray,
        centers: np.ndarray,
        samples_per_symbol: float,
    ) -> VSAAnalysisResult:
        symbol_frequency = []
        half_width = max(1, int(np.floor(samples_per_symbol * 0.25)))
        for center in centers:
            index = int(round(center))
            start = max(0, index - half_width)
            stop = min(instantaneous_frequency_hz.size, index + half_width + 1)
            symbol_frequency.append(float(np.mean(instantaneous_frequency_hz[start:stop])))
        measured_frequency = np.asarray(symbol_frequency, dtype=np.float64)
        center_frequency_error = float(np.mean(measured_frequency)) if measured_frequency.size else 0.0
        centered = measured_frequency - center_frequency_error
        decoded = (centered >= 0.0).astype(np.int16)
        expected_deviation = signal.frequency_deviation_hz
        if expected_deviation is None and centered.size:
            expected_deviation = float(np.mean(np.abs(centered)))
        deviation = float(expected_deviation or 0.0)
        reference_frequency = np.where(decoded == 0, -deviation, deviation)
        measured_symbols = centered.astype(np.complex64)
        reference_symbols = reference_frequency.astype(np.complex64)
        return VSAAnalysisResult(
            time_s=time_s,
            iq=iq,
            power_dbfs=power_dbfs,
            power_dbm=power_dbm,
            spectrum_frequency_hz=spectrum_frequency_hz,
            spectrum_dbfs=spectrum_dbfs,
            instantaneous_frequency_hz=instantaneous_frequency_hz,
            symbol_time_s=centers / recording.sample_rate_hz,
            measured_symbols=measured_symbols,
            reference_symbols=reference_symbols,
            decoded_symbols=decoded,
            decoded_bits=decoded.astype(np.uint8),
            evm_rms_percent=None,
            frequency_error_hz=center_frequency_error,
            metadata={
                "modulation": signal.modulation.value,
                "samples_per_symbol": samples_per_symbol,
                "estimated_deviation_hz": deviation,
                "amplitude_calibrated": recording.amplitude_calibrated,
            },
        )

    def _analyze_psk(
        self,
        recording: IQRecording,
        signal: SignalDescription,
        iq: np.ndarray,
        time_s: np.ndarray,
        power_dbfs: np.ndarray,
        power_dbm: np.ndarray,
        spectrum_frequency_hz: np.ndarray,
        spectrum_dbfs: np.ndarray,
        instantaneous_frequency_hz: np.ndarray,
        centers: np.ndarray,
    ) -> VSAAnalysisResult:
        sampled = _interpolate_complex(iq, centers)
        rms = float(np.sqrt(np.mean(np.abs(sampled) ** 2))) if sampled.size else 1.0
        normalized = sampled / max(rms, _EPSILON)
        constellation = _constellation(signal.modulation)
        decision_input = normalized
        if signal.modulation.differential and normalized.size > 1:
            decision_input = normalized[1:] * np.conj(normalized[:-1])
            centers = centers[1:]
        distances = np.abs(decision_input[:, None] - constellation[None, :])
        decoded = np.argmin(distances, axis=1).astype(np.int16)
        reference = constellation[decoded]
        error = decision_input - reference
        evm = (
            float(100.0 * np.sqrt(np.mean(np.abs(error) ** 2) / np.mean(np.abs(reference) ** 2)))
            if reference.size
            else None
        )
        return VSAAnalysisResult(
            time_s=time_s,
            iq=iq,
            power_dbfs=power_dbfs,
            power_dbm=power_dbm,
            spectrum_frequency_hz=spectrum_frequency_hz,
            spectrum_dbfs=spectrum_dbfs,
            instantaneous_frequency_hz=instantaneous_frequency_hz,
            symbol_time_s=centers / recording.sample_rate_hz,
            measured_symbols=decision_input,
            reference_symbols=reference,
            decoded_symbols=decoded,
            decoded_bits=_symbols_to_bits(decoded, signal.modulation.order),
            evm_rms_percent=evm,
            frequency_error_hz=None,
            metadata={
                "modulation": signal.modulation.value,
                "samples_per_symbol": recording.sample_rate_hz / signal.symbol_rate_hz,
                "differential": signal.modulation.differential,
                "amplitude_calibrated": recording.amplitude_calibrated,
            },
        )
