"""Generated, file, and shared-acquisition IQ sources for the VSA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from pluto_sa.sdr.trigger import IQAcquisitionRecord
from pluto_sa.vsa.model import IQRecording, ModulationKind, SignalDescription


@dataclass(frozen=True)
class IQSourceCapabilities:
    finite_capture: bool
    continuous_stream: bool
    hardware_trigger: bool
    writable_frontend: bool


class GeneratedIQSource:
    """Deterministic waveform source for tests and the first offline UI."""

    capabilities = IQSourceCapabilities(
        finite_capture=True,
        continuous_stream=False,
        hardware_trigger=False,
        writable_frontend=False,
    )

    @staticmethod
    def fsk(
        *,
        symbol_count: int = 256,
        symbol_rate_hz: float = 1_000_000.0,
        samples_per_symbol: int = 8,
        frequency_deviation_hz: float = 250_000.0,
        gaussian_bt: float | None = 0.5,
        seed: int = 1,
    ) -> tuple[IQRecording, SignalDescription]:
        if int(symbol_count) <= 0:
            raise ValueError("symbol_count must be positive")
        if int(samples_per_symbol) < 2:
            raise ValueError("samples_per_symbol must be at least 2")
        if float(frequency_deviation_hz) <= 0.0:
            raise ValueError("frequency_deviation_hz must be positive")
        rng = np.random.default_rng(int(seed))
        symbols = rng.integers(0, 2, size=int(symbol_count), dtype=np.uint8)
        levels = np.repeat(2.0 * symbols.astype(np.float64) - 1.0, int(samples_per_symbol))
        modulation = ModulationKind.FSK2
        tx_filter = "None"
        if gaussian_bt is not None:
            if float(gaussian_bt) <= 0.0:
                raise ValueError("gaussian_bt must be positive")
            # This deterministic approximation is a test-waveform shaper, not
            # yet the reference Gaussian pulse used for measurement EVM.
            sigma_samples = max(0.5, int(samples_per_symbol) / (2.0 * np.pi * float(gaussian_bt)))
            levels = gaussian_filter1d(levels, sigma=sigma_samples, mode="nearest")
            modulation = ModulationKind.GFSK
            tx_filter = "Gaussian"
        sample_rate_hz = float(symbol_rate_hz) * int(samples_per_symbol)
        instantaneous_frequency = float(frequency_deviation_hz) * levels
        phase = 2.0 * np.pi * np.cumsum(instantaneous_frequency) / sample_rate_hz
        iq = np.exp(1j * phase).astype(np.complex64)
        recording = IQRecording(
            iq=iq,
            sample_rate_hz=sample_rate_hz,
            usable_bandwidth_hz=0.8 * sample_rate_hz,
            source="Generated FSK",
            amplitude_calibrated=True,
            metadata={"generated_symbols": symbols, "seed": int(seed)},
        )
        signal = SignalDescription(
            modulation=modulation,
            symbol_rate_hz=float(symbol_rate_hz),
            frequency_deviation_hz=float(frequency_deviation_hz),
            tx_filter=tx_filter,
            filter_parameter=gaussian_bt,
            name="Generated FSK",
        )
        return recording, signal

    @staticmethod
    def psk(
        *,
        modulation: ModulationKind = ModulationKind.QPSK,
        symbol_count: int = 256,
        symbol_rate_hz: float = 1_000_000.0,
        samples_per_symbol: int = 8,
        seed: int = 1,
    ) -> tuple[IQRecording, SignalDescription]:
        if modulation.family.value != "PSK":
            raise ValueError("modulation must be a PSK kind")
        if int(symbol_count) <= 0:
            raise ValueError("symbol_count must be positive")
        if int(samples_per_symbol) < 2:
            raise ValueError("samples_per_symbol must be at least 2")
        rng = np.random.default_rng(int(seed))
        symbols = rng.integers(0, modulation.order, size=int(symbol_count), dtype=np.int16)
        if modulation is ModulationKind.BPSK:
            alphabet = np.exp(1j * np.array([0.0, np.pi]))
        elif modulation in {ModulationKind.QPSK, ModulationKind.OQPSK, ModulationKind.PI4_DQPSK}:
            alphabet = np.exp(1j * (np.pi / 4.0 + np.arange(4) * np.pi / 2.0))
        else:
            alphabet = np.exp(1j * np.arange(8) * np.pi / 4.0)
        if modulation.differential:
            waveform_symbols = np.cumprod(alphabet[symbols])
        else:
            waveform_symbols = alphabet[symbols]
        iq = np.repeat(waveform_symbols, int(samples_per_symbol)).astype(np.complex64)
        sample_rate_hz = float(symbol_rate_hz) * int(samples_per_symbol)
        recording = IQRecording(
            iq=iq,
            sample_rate_hz=sample_rate_hz,
            usable_bandwidth_hz=0.8 * sample_rate_hz,
            source="Generated PSK",
            amplitude_calibrated=True,
            metadata={"generated_symbols": symbols, "seed": int(seed)},
        )
        signal = SignalDescription(
            modulation=modulation,
            symbol_rate_hz=float(symbol_rate_hz),
            tx_filter="None",
            name="Generated PSK",
        )
        return recording, signal


class FileIQSource:
    """Load NumPy containers or raw complex IQ without modifying the source."""

    capabilities = IQSourceCapabilities(
        finite_capture=True,
        continuous_stream=False,
        hardware_trigger=False,
        writable_frontend=False,
    )

    @staticmethod
    def load(
        path: str | Path,
        *,
        sample_rate_hz: float | None = None,
        center_frequency_hz: float = 0.0,
        raw_dtype: str = "complex64",
    ) -> IQRecording:
        resolved = Path(path)
        suffix = resolved.suffix.lower()
        metadata: dict[str, object] = {"path": str(resolved.resolve())}
        if suffix == ".npy":
            iq = np.load(resolved, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(resolved, allow_pickle=False) as container:
                key = "iq" if "iq" in container.files else container.files[0]
                iq = np.array(container[key], copy=True)
                if sample_rate_hz is None and "sample_rate_hz" in container.files:
                    sample_rate_hz = float(np.asarray(container["sample_rate_hz"]).item())
                if center_frequency_hz == 0.0 and "center_frequency_hz" in container.files:
                    center_frequency_hz = float(np.asarray(container["center_frequency_hz"]).item())
                calibration_offset_db = (
                    float(np.asarray(container["calibration_offset_db"]).item())
                    if "calibration_offset_db" in container.files
                    else 0.0
                )
                frequency_dependent_offset_db = (
                    float(np.asarray(container["frequency_dependent_offset_db"]).item())
                    if "frequency_dependent_offset_db" in container.files
                    else 0.0
                )
                input_correction_db = (
                    float(np.asarray(container["input_correction_db"]).item())
                    if "input_correction_db" in container.files
                    else 0.0
                )
                amplitude_calibrated = (
                    bool(np.asarray(container["amplitude_calibrated"]).item())
                    if "amplitude_calibrated" in container.files
                    else False
                )
                metadata["container_key"] = key
        else:
            dtype = np.dtype(raw_dtype)
            if dtype.kind != "c":
                raise ValueError("raw_dtype must be a complex NumPy dtype")
            iq = np.fromfile(resolved, dtype=dtype)
            calibration_offset_db = 0.0
            frequency_dependent_offset_db = 0.0
            input_correction_db = 0.0
            amplitude_calibrated = False
        if suffix == ".npy":
            calibration_offset_db = 0.0
            frequency_dependent_offset_db = 0.0
            input_correction_db = 0.0
            amplitude_calibrated = False
        if sample_rate_hz is None:
            raise ValueError("sample_rate_hz is required when the file has no metadata")
        return IQRecording(
            iq=iq,
            sample_rate_hz=float(sample_rate_hz),
            center_frequency_hz=float(center_frequency_hz),
            usable_bandwidth_hz=0.8 * float(sample_rate_hz),
            source=f"File: {resolved.name}",
            calibration_offset_db=calibration_offset_db,
            frequency_dependent_offset_db=frequency_dependent_offset_db,
            input_correction_db=input_correction_db,
            amplitude_calibrated=amplitude_calibrated,
            metadata=metadata,
        )

    @staticmethod
    def save_npz(path: str | Path, recording: IQRecording) -> None:
        np.savez(
            Path(path),
            iq=recording.iq,
            sample_rate_hz=np.float64(recording.sample_rate_hz),
            center_frequency_hz=np.float64(recording.center_frequency_hz),
            calibration_offset_db=np.float64(recording.calibration_offset_db),
            frequency_dependent_offset_db=np.float64(
                recording.frequency_dependent_offset_db
            ),
            input_correction_db=np.float64(recording.input_correction_db),
            amplitude_calibrated=np.bool_(recording.amplitude_calibrated),
        )


def recording_from_acquisition(
    record: IQAcquisitionRecord,
    *,
    calibration_offset_db: float = 0.0,
    frequency_dependent_offset_db: float = 0.0,
    input_correction_db: float = 0.0,
    amplitude_calibrated: bool = False,
) -> IQRecording:
    """Adapt the common Pluto trigger record without coupling DSP to Pluto."""
    return IQRecording(
        iq=record.iq,
        sample_rate_hz=record.metadata.sample_rate_hz,
        center_frequency_hz=record.metadata.center_freq_hz,
        usable_bandwidth_hz=record.metadata.rf_bandwidth_hz,
        source=record.metadata.source,
        full_scale=record.metadata.iq_full_scale,
        calibration_offset_db=float(calibration_offset_db),
        frequency_dependent_offset_db=float(frequency_dependent_offset_db),
        input_correction_db=float(input_correction_db),
        amplitude_calibrated=bool(amplitude_calibrated),
        start_sample_index=record.start_sample_index,
        trigger_sample_index=record.trigger_sample_index,
        discontinuity_reason=record.discontinuity_reason,
        metadata={
            "stream_id": record.stream_id,
            "gain_db": record.metadata.gain_db,
            "trigger_kind": record.trigger.kind.value,
            "trigger_forced": record.trigger.forced,
        },
    )
