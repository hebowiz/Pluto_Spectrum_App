import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import numpy as np
from pluto_protocol.bluetooth.hdt import HDTRate
from pluto_vsa.model import (
    IQRecording,
    ModulationKind,
    SignalDescription,
    VSAAnalysisResult,
)
from pluto_vsa.session import VSASession
from pluto_vsg.engine import BluetoothHDTWaveformEngine, BluetoothLEWaveformEngine
from pluto_vsg.model import BluetoothLEPhy
from pluto_vsg.profiles import (
    bluetooth_hdt_fields,
    bluetooth_hdt_project,
    bluetooth_le_project,
)


def _hdt_recording(rate: HDTRate, payload_length: int = 64):
    base = bluetooth_hdt_project(rate)
    settings = replace(
        base.bluetooth_hdt, payload_length_bytes=payload_length
    )
    project = replace(
        base,
        bluetooth_hdt=settings,
        fields=bluetooth_hdt_fields(settings),
    )
    generated = BluetoothHDTWaveformEngine().generate(project)
    return IQRecording(
        iq=generated.iq,
        sample_rate_hz=generated.sample_rate_hz,
        center_frequency_hz=project.center_frequency_hz,
        source=f"generated {rate.value}",
    ), generated, project


def _session_with_le_bits() -> tuple[VSASession, dict[str, object]]:
    generated = BluetoothLEWaveformEngine().generate(bluetooth_le_project(BluetoothLEPhy.LE_1M))
    artifact = generated.packet_bits
    assert artifact is not None
    count = generated.iq.size
    time_s = np.arange(count, dtype=np.float64) / generated.sample_rate_hz
    spectrum_frequency_hz = np.linspace(-2e6, 2e6, 256)
    result = VSAAnalysisResult(
        time_s=time_s,
        iq=generated.iq,
        power_dbfs=np.full(count, -12.0),
        power_dbm=np.full(count, -22.0),
        spectrum_frequency_hz=spectrum_frequency_hz,
        spectrum_dbfs=np.full(256, -70.0),
        spectrum_dbm=np.full(256, -80.0),
        instantaneous_frequency_hz=np.zeros(count),
        symbol_time_s=np.arange(artifact.bits.size) / 1e6,
        measured_symbols=np.ones(artifact.bits.size, dtype=np.complex64),
        reference_symbols=np.ones(artifact.bits.size, dtype=np.complex64),
        decoded_symbols=artifact.bits.astype(np.int16),
        decoded_bits=artifact.bits,
        evm_rms_percent=1.0,
        frequency_error_hz=2_500.0,
        metadata={"symbol_rate_error_ppm": 1.25},
    )
    session = VSASession(
        recording=IQRecording(
            iq=generated.iq,
            sample_rate_hz=generated.sample_rate_hz,
            center_frequency_hz=2_402e6,
            source="test",
        ),
        signal=SignalDescription(ModulationKind.FSK, 1e6, 250e3),
        result=result,
    )
    return session, dict(artifact.context)
